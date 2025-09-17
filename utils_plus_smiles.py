import numpy as np
import math
import torch
import torch.utils.data
from torchvision import transforms
from PIL import Image
from torch import nn
from torch_geometric.data import Batch
import pickle


class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, graph, fp, label, smiles):
        self.imgs = imgs
        self.graph = graph
        self.fp = fp
        self.label = label
        self.smiles = smiles

        self.transformImg = transforms.Compose([
            transforms.ToTensor()])

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, item):
        img_path = self.imgs[item]
        graph_feature = self.graph[item]
        fp_feature = self.fp[item]
        label_feature = self.label[item]
        smiles_feature = self.smiles[item]
        img = Image.open(img_path).convert('RGB')
        img = self.transformImg(img)

        return img, graph_feature, fp_feature, label_feature, smiles_feature


def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name)]

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        data_list = pickle.load(f)
    return data_list

def collate_fn(batch):
    imgs, graphs, fps, labels, smiles = zip(*batch)
    batch_graphs = Batch.from_data_list(graphs)
    batch_imgs = torch.stack(imgs)
    batch_fps = torch.stack(fps)
    batch_labels = torch.tensor(labels)
    return batch_imgs, batch_graphs, batch_fps, batch_labels, list(smiles)

def data_loader(batch_size, imgs, graph_name, fp_name, active_name, smiles_name):
    graph = load_pickle(graph_name)  # pickle로 저장된 그래프 데이터를 로드합니다.
    # graph = load_tensor(graph_name, torch.FloatTensor)
    fp = load_tensor(fp_name, torch.FloatTensor)
    activities = load_tensor(active_name, torch.LongTensor)
    smiles = np.load(smiles_name)

    dataset = Dataset(imgs, graph, fp, activities, smiles)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)   
    # dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)  
    return dataset, dataset_loader, smiles


def get_img_path(img_path):
    imgs = []
    with open(img_path, "r") as f:
        lines = f.read().strip().split("\n")
        for line in lines:
            imgs.append(line.split("\t")[0])
    return imgs



def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())
