import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from autoencoder import Autoencoder, train_autoencoder, Autoencoder_layer3
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, GINConv
from torch.nn import BatchNorm1d, Linear, ReLU
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')


def _init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class GINGATCrossModel(torch.nn.Module):
    def __init__(self, num_features_xd=1, dim=64, output_dim=256, heads=4):  # nf 1
        # def __init__(self, num_features_xd=8, dim=32, output_dim=256, heads=8):            # nf 8
        super(GINGATCrossModel, self).__init__()

        mlp1 = nn.Sequential(
            Linear(num_features_xd, dim),
            ReLU(),
            Linear(dim, dim)
        )

        self.gin1 = GINConv(mlp1)
        self.bn1 = BatchNorm1d(dim)

        self.gat1 = GATConv(in_channels=dim, out_channels=dim, heads=heads, concat=True)
        self.bn2 = BatchNorm1d(dim * heads)

        mlp2 = nn.Sequential(
            Linear(dim * heads, dim * heads),
            ReLU(),
            Linear(dim * heads, dim * heads)
        )

        self.gin2 = GINConv(mlp2)
        self.bn3 = BatchNorm1d(dim * heads)

        self.gat2 = GATConv(in_channels=dim * heads, out_channels=output_dim, heads=heads, concat=False)
        self.bn4 = BatchNorm1d(output_dim)

        self.fc = Linear(dim * heads, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.gin1(x, edge_index))
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gin2(x, edge_index))

        x = self.gat2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 act_layer=nn.GELU,
                 drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SparseMoE_Self_Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=4,
                 num_heads=4,
                 top_k=2,
                 qkv_bias=False,
                 drop_ratio=0.,
                 ):
        super(SparseMoE_Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define a separate qkv layer for each expert
        self.qkv_experts = nn.ModuleList([nn.Linear(dim, dim * 3, bias=qkv_bias) for _ in range(num_experts)])

        # Define gating network
        self.gating = nn.Linear(dim, num_experts)

        self.attn_drop = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        B, N = x.shape

        # Gating network to get the importance of each expert
        gate_scores = self.gating(x)  # Shape: (B, num_experts)
        gate_scores = F.softmax(gate_scores, dim=-1)  # Normalize scores to get probabilities

        # Select top-k experts for each input
        top_k_values, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)

        # Initialize q, k, v as zeros
        q = k = v = torch.zeros(B, self.num_heads, N // self.num_heads, device=x.device)

        # Only use the top-k experts for each input
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # Shape: (B,)
            gate_score = top_k_values[:, i].unsqueeze(-1).unsqueeze(-1)

            # Gather qkv for the selected experts
            qkv = torch.stack([self.qkv_experts[idx](x[b:b + 1])
                               for b, idx in enumerate(expert_idx)], dim=0)
            qkv = qkv.reshape(B, 3, self.num_heads, N // self.num_heads).permute(1, 0, 2, 3)

            q_i, k_i, v_i = qkv[0], qkv[1], qkv[2]

            # Accumulate q, k, v weighted by gate_score
            q += gate_score * q_i
            k += gate_score * k_i
            v += gate_score * v_i

        # Self attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SparseMoE_Cross_Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=4,
                 num_heads=4,
                 top_k=2,
                 qkv_bias=False,
                 drop_ratio=0.,
                 ):
        super(SparseMoE_Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define a separate qkv layer for each expert
        self.qkv_experts = nn.ModuleList([nn.Linear(dim, dim * 3, bias=qkv_bias) for _ in range(num_experts)])

        # Define gating network
        self.gating = nn.Linear(dim, num_experts)

        self.attn_drop = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_ratio)

    def forward(self, x, y):
        B, N = x.shape

        # Gating network to get the importance of each expert
        gate_scores = self.gating(x)  # Shape: (B, num_experts)
        gate_scores = F.softmax(gate_scores, dim=-1)  # Normalize scores to get probabilities

        # Select top-k experts for each input
        top_k_values, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)

        # Initialize q, k, v as zeros
        q = k = v = torch.zeros(B, self.num_heads, N // self.num_heads, device=x.device)

        # Only use the top-k experts for each input
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # Shape: (B,)
            gate_score = top_k_values[:, i].unsqueeze(-1).unsqueeze(-1)

            # Gather qkv for the selected experts
            qkv_x = torch.stack([self.qkv_experts[idx](x[b:b + 1])
                                 for b, idx in enumerate(expert_idx)], dim=0)
            qkv_y = torch.stack([self.qkv_experts[idx](y[b:b + 1])
                                 for b, idx in enumerate(expert_idx)], dim=0)

            qkv_x = qkv_x.reshape(B, 3, self.num_heads, N // self.num_heads).permute(1, 0, 2, 3)
            qkv_y = qkv_y.reshape(B, 3, self.num_heads, N // self.num_heads).permute(1, 0, 2, 3)

            q_i, k_i, v_i = qkv_y[0], qkv_x[1], qkv_x[2]

            # Accumulate q, k, v weighted by gate_score
            q += gate_score * q_i
            k += gate_score * k_i
            v += gate_score * v_i

        # Cross attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_ratio=0.
                 ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)

        self.attn = SparseMoE_Self_Attention(dim, drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop_ratio
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Decoder(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_ratio=0.
                 ):
        super(Decoder, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SparseMoE_Self_Attention(dim, drop_ratio=drop_ratio)

        self.cross_attn = SparseMoE_Cross_Attention(dim, drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop_ratio
        )

    def forward(self, x, y):
        out = y + self.drop_path(self.cross_attn(x, self.norm1(y)))
        out = out + self.drop_path(self.mlp(self.norm2(y)))

        return out


class PositionalMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(PositionalMLP, self).__init__()
        self.mlp = Mlp(in_features=embed_dim, hidden_features=hidden_dim, out_features=embed_dim)

    def forward(self, x):
        return self.mlp(x)


class MEMOL(nn.Module):
    def __init__(self,
                 depth_e1=4,
                 depth_e2=4,
                 depth_decoder=4,
                 embed_dim=256,
                 drop_ratio=0.,
                 backbone="",
                 graph_backbone="",
                 ae_model_path=None,
                 ):
        super(MEMOL, self).__init__()

        self.depth_decoder = depth_decoder
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.rate1 = torch.nn.Parameter(torch.rand(1))
        self.rate2 = torch.nn.Parameter(torch.rand(1))
        self.rate3 = torch.nn.Parameter(torch.rand(1))

        if backbone == "CNN":
            self.img_backbone = nn.Sequential(  # 3*256*256
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),  # 64*128*128
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.02, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 64*64*64

                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1),  # 128*32*32
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.02, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 128*16*16

                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),  # 256*8*8
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.02, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 256*4*4

                nn.AdaptiveAvgPool2d((1, 1)),

                nn.Flatten()
            )
        elif backbone == "ViT":
            # Vision Transformer (ViT) Backbone from timm
            vit = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.img_backbone = nn.Sequential(
                vit.patch_embed,
                vit.pos_drop,
                *vit.blocks,
                vit.norm,
                nn.Flatten(),
                nn.Linear(224 * 224 * 3, 256)
            )

        if graph_backbone == "GIN_GAT":
            self.graph_backbone = GINGATCrossModel()

        self.ae_encoder = Autoencoder().encoder
        if ae_model_path:
            self.ae_encoder.load_state_dict(torch.load(ae_model_path))
        self.ae_encoder.eval()
        self.reduce_dim = nn.Linear(285, 256)
        self.reduce_fp_dim_1024 = nn.Linear(1024, 256)
        self.reduce_fp_dim_1536 = nn.Linear(1536, 256)
        self.reduce_fp_dim_2048 = nn.Linear(2048, 256)
        self.reduce_fp_dim_3072 = nn.Linear(3072, 256)
        self.reduce_fp_dim_4000 = nn.Linear(4000, 256)
        self.reduce_dim_512To256 = nn.Linear(512, 256)
        self.reduce_dim_1536To256 = nn.Linear(1536, 256)

        # Positional MLPs
        self.pos_mlp_e1 = PositionalMLP(embed_dim=embed_dim, hidden_dim=512)
        self.pos_mlp_e2 = PositionalMLP(embed_dim=embed_dim, hidden_dim=512)
        self.pos_mlp_e3 = PositionalMLP(embed_dim=embed_dim, hidden_dim=512)
        self.pos_mlp_final = PositionalMLP(embed_dim=embed_dim, hidden_dim=512)

        #  encoder 1
        self.norm_e1 = norm_layer(embed_dim)
        self.pos_drop_e1 = nn.Dropout(p=drop_ratio)
        self.pos_embed_e1 = nn.Parameter(torch.zeros(1, embed_dim))
        dpr_e1 = [x.item() for x in torch.linspace(0, drop_ratio, depth_e1)]
        self.encoder_e1 = nn.Sequential(*[
            Block(dim=embed_dim,
                  mlp_ratio=4,
                  drop_ratio=dpr_e1[i],
                  )
            for i in range(depth_e1)
        ])

        #  encoder 2
        self.norm_e2 = norm_layer(embed_dim)
        self.pos_drop_e2 = nn.Dropout(p=drop_ratio)
        self.pos_embed_e2 = nn.Parameter(torch.zeros(1, embed_dim))
        dpr_e2 = [x.item() for x in torch.linspace(0, drop_ratio, depth_e1)]
        self.encoder_e2 = nn.Sequential(*[
            Block(dim=embed_dim,
                  mlp_ratio=4,
                  drop_ratio=dpr_e2[i],
                  )
            for i in range(depth_e1)
        ])

        # encoder 3
        self.norm_e3 = norm_layer(embed_dim)
        self.pos_drop_e3 = nn.Dropout(p=drop_ratio)
        self.pos_embed_e3 = nn.Parameter(torch.zeros(1, embed_dim))
        dpr_e3 = [x.item() for x in torch.linspace(0, drop_ratio, depth_e1)]
        self.encoder_e3 = nn.Sequential(*[
            Block(dim=embed_dim,
                  mlp_ratio=4,
                  drop_ratio=dpr_e3[i],
                  )
            for i in range(depth_e1)
        ])

        # decoder
        self.norm_decoder = norm_layer(embed_dim)
        self.pos_drop_decoder = nn.Dropout(p=drop_ratio)
        self.decoder = Decoder(dim=embed_dim, mlp_ratio=4, drop_ratio=0., )

        # decoder2
        self.norm_decoder2 = norm_layer(embed_dim)
        self.pos_drop_decoder2 = nn.Dropout(p=drop_ratio)
        self.decoder2 = Decoder(dim=embed_dim, mlp_ratio=4, drop_ratio=0., )

        self.decoder_1d_2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=4),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=4),
            nn.BatchNorm1d(64),
            nn.ELU(),

            nn.AdaptiveMaxPool1d(1),
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*64*64

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.fc_256To2 = nn.Linear(256, 2)
        self.fc_512To2 = nn.Linear(512, 2)
        self.fc_768To2 = nn.Linear(768, 2)

    def reduce_fp_autoencoder_2048To256(self, fp):
        # Ensure the data is on the correct device
        fp = fp.to(next(self.ae_encoder.parameters()).device)
        with torch.no_grad():
            fp_reduced = self.ae_encoder(fp)
        return fp_reduced

    def forward_features_e1(self, x):
        B, _, _, _ = x.shape
        x = self.img_backbone(x)
        pos_encoding = self.pos_embed_e1
        x = self.pos_drop_e1(x + pos_encoding)
        x = self.encoder_e1(x)
        x = self.norm_e1(x)
        return x

    def forward_features_e2(self, x):
        x = self.graph_backbone(x)
        pos_encoding = self.pos_embed_e2
        x = self.pos_drop_e2(x + pos_encoding)
        x = self.encoder_e2(x)
        x = self.norm_e2(x)
        return x

    def forward_features_e3(self, x):
        pos_encoding = self.pos_embed_e3
        x = self.pos_drop_e3(x + pos_encoding)
        x = self.encoder_e3(x)
        x = self.norm_e3(x)
        return x

    def forward(self, inputs):
        image, graph, fp = inputs[0], inputs[1], inputs[2]
        image = image.to(device)
        graph = graph.to(device)
        fp = fp.to(device)

        fp = self.reduce_fp_autoencoder_2048To256(fp)

        image_feature = self.forward_features_e1(image)
        graph_feature = self.forward_features_e2(graph)
        fp = self.forward_features_e3(fp)

        image_graph_cross1 = self.decoder(image_feature, graph_feature)
        image_graph_cross2 = self.decoder(graph_feature, image_feature)
        cross_attention1 = torch.cat((image_graph_cross1, image_graph_cross2), dim=1)  # B, 512

        cross_attention1 = self.reduce_dim_512To256(cross_attention1)

        image_graph_fp_cross1 = self.decoder(cross_attention1, fp)
        image_graph_fp_cross2 = self.decoder(fp, cross_attention1)
        cross_attention2 = torch.cat((image_graph_fp_cross1, image_graph_fp_cross2), dim=1)  # B, 512
        out = self.fc_512To2(cross_attention2)

        return out, self.rate1, self.rate2, self.rate3

    def __call__(self, data, train=True):
        inputs, correct_interaction, = data[:3], data[3]
        predicted_interaction, rate1, rate2, rate3 = self.forward(inputs)
        correct_interaction = torch.squeeze(correct_interaction)

        calculated_loss = F.cross_entropy(predicted_interaction, correct_interaction.to(device))

        correct_labels = correct_interaction.to('cpu').data.numpy()
        ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
        predicted_labels = list(map(lambda x: np.argmax(x), ys))
        predicted_scores = list(map(lambda x: x[1], ys))
        return calculated_loss, correct_labels, predicted_labels, predicted_scores, rate1, rate2, rate3
