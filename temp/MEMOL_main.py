# hyper tune file
import os
import shutil
import torch
import argparse
from datetime import datetime
from MEMOL_model import MEMOL
from Train import Train_model, Tester
from utils_plus_smiles import data_loader, get_img_path
import random
import numpy as np

from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, auc, average_precision_score


def model_run(args, ae_model_path, ae_model_type, depth_e1, drop_ratio, learning_rate):
    ISOTIMEFORMAT = '%Y_%m%d_%H%M'
    run_time = datetime.now().strftime(ISOTIMEFORMAT)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"dataset name：{args.dataset_name} run_time: {run_time}")

    resul_name = "result/" + args.dataset_name
    if not os.path.exists(resul_name):
        os.makedirs(resul_name)

    high_dim = ''
    train_img_path = "data/" + args.dataset_name + "/train/" + "Img_" + str(args.img_size) + "_" + str(
        args.img_size) + "/img_inf_data"
    train_image = get_img_path(train_img_path)
    train_graph_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_train_graph_nf1_Data" + high_dim + ".pkl"
    train_fp_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_train_fingerprint" + high_dim + ".npy"

    train_activities_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_train_activities.npy"
    train_smiles_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_train_smiles.npy"
    train_dataset, train_loader, train_smiles = data_loader(batch_size=args.batch_size,
                                                            imgs=train_image,
                                                            graph_name=train_graph_name,
                                                            fp_name=train_fp_name,
                                                            active_name=train_activities_name,
                                                            smiles_name=train_smiles_name)


    val_img_path = "data/" + args.dataset_name + "/val/" + "Img_" + str(args.img_size) + "_" + str(
        args.img_size) + "/img_inf_data"
    val_image = get_img_path(val_img_path)
    val_graph_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_val_graph_nf1_Data" + high_dim + ".pkl"
    val_fp_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_val_fingerprint" + high_dim + ".npy"

    val_activities_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_val_activities.npy"
    val_smiles_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_val_smiles.npy"
    val_dataset, val_loader, val_smiles = data_loader(batch_size=args.batch_size,
                                          imgs=val_image,
                                          graph_name=val_graph_name,
                                          fp_name=val_fp_name,
                                          active_name=val_activities_name,
                                          smiles_name=val_smiles_name)


    test_img_path = "data/" + args.dataset_name + "/test/" + "Img_" + str(args.img_size) + "_" + str(
        args.img_size) + "/img_inf_data"
    test_image = get_img_path(test_img_path)
    test_graph_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_test_graph_nf1_Data" + high_dim + ".pkl"
    test_fp_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_test_fingerprint" + high_dim + ".npy"

    test_activities_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_test_activities.npy"
    test_smiles_name = "data/" + args.dataset_name + "/input/" + args.dataset_name + "_test_smiles.npy"
    test_dataset, test_loader, test_smiles = data_loader(batch_size=args.batch_size,
                                                         imgs=test_image,
                                                         graph_name=test_graph_name,
                                                         fp_name=test_fp_name,
                                                         active_name=test_activities_name,
                                                         smiles_name=test_smiles_name)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = MEMOL(depth_e1=depth_e1,
                   depth_e2=args.depth_e2,
                   depth_decoder=args.depth_decoder,
                   embed_dim=args.embed_dim,
                   drop_ratio=drop_ratio,
                   backbone=args.backbone,
                   graph_backbone=args.graph_backbone,
                   ae_model_path=ae_model_path,
                   ).to(device)

    trainer = Train_model(model, learning_rate, weight_decay=1e-8)

    print("Training Start")
    best_auc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        if epoch % 10 == 0:
            trainer.optimizer.param_groups[0]['lr'] *= args.lr_decay

        total_loss = []

        for i, data_train in enumerate(train_loader):
            if data_train[0].shape[0] <= 1:
                break

            loss_train = trainer.train(data_train)
            total_loss.append(loss_train)
            if (i + 1) % 50 == 0:
                print(
                    f"Training "
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"[Batch  {i}/{len(train_loader)}] "
                    f"[batch_size {data_train[0].shape[0]}] "
                    f"[loss_train : {loss_train}]")

        model_path = "data/" + args.dataset_name + "/output/model/"
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print("Epoch: ", epoch, "     Train avg_loss: ", sum(total_loss) / len(train_loader))

        with torch.no_grad():
            # val(args, model, val_loader)
            aucs = test(args, epoch, model, test_dataset, test_loader, test_smiles, ae_model_type, depth_e1,
                        drop_ratio, learning_rate)
            if float(aucs[4]) > best_auc:
                best_auc = float(aucs[4])
                best_epoch = epoch

    return best_auc, best_epoch


def val(args, file_model, val_loader):
    model = file_model
    model.eval()
    valer = Tester(model)
    Loss, y_label, y_pred, y_score = [], [], [], []
    for i, data_list in enumerate(val_loader):
        loss, correct_labels, predicted_labels, predicted_scores, rate1, rate2, rate3 = valer.test(data_list)
        Loss.append(loss)
        for c_l in correct_labels:
            y_label.append(c_l)
        for p_l in predicted_labels:
            y_pred.append(p_l)
        for p_s in predicted_scores:
            y_score.append(p_s)

    loss_val = sum(Loss) / len(val_loader)
    AUC_val = roc_auc_score(y_label, y_score)

    AUPRC = average_precision_score(y_label, y_score)

    print(
        "Validation:   Batch %d [loss : %.3f] [AUROC : %.3f] [AUPRC : %.3f]"
        % (args.batch_size, loss_val, AUC_val, AUPRC)
    )


def test(args, epoch, file_model, test_dataset, test_loader, smiles, ae_model_type, depth_e1, drop_ratio,
         learning_rate):

    model = file_model
    model.eval()
    tester = Tester(model)
    Loss, y_label, y_pred, y_score = [], [], [], []
    rate1s = []
    rate2s = []
    rate3s = []
    for i, data_list in enumerate(test_loader):
        loss, correct_labels, predicted_labels, predicted_scores, rate1, rate2, rate3 = tester.test(data_list)
        Loss.append(loss)
        for c_l in correct_labels:
            y_label.append(c_l)
        for p_l in predicted_labels:
            y_pred.append(p_l)
        for p_s in predicted_scores:
            y_score.append(p_s)
        for r1 in rate1:
            rate1s.append(r1)
        for r2 in rate2:
            rate2s.append(r2)
        for r3 in rate3:
            rate3s.append(r3)

    loss_test = sum(Loss) / len(test_loader)
    AUC_test = roc_auc_score(y_label, y_score)

    AUPRC = average_precision_score(y_label, y_score)

    directory = f"result/{args.dataset_name}/{args.dataset_name}_test0304{ae_model_type}_depth{depth_e1}_drop{drop_ratio}_lr{learning_rate}"

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = f"{directory}/{args.dataset_name}_{ae_model_type}_seed42_depth{depth_e1}_drop{drop_ratio}_lr{learning_rate}_epoch{epoch}.txt"

    with open(filename, 'w') as f:
        f.write("SMILES\tLabel\tScore\tPrediction\n")
        for i, (smiles_item, label, score, pred) in enumerate(zip(smiles, y_label, y_score, y_pred)):
            f.write(f"{smiles_item}\t{label}\t{score:.6f}\t{pred}\n")

    rr1 = sum(rate1s) / len(rate1s)
    rr2 = sum(rate2s) / len(rate2s)
    rr3 = sum(rate3s) / len(rate3s)
    print(
        "Test:  Batch %d [loss : %.3f] [AUROC : %.3f] [AUPRC : %.3f]"
        % (args.batch_size, loss_test, AUC_test, AUPRC)
    )
    print()

    AUCs = [epoch,
            len(test_dataset),
            len(test_loader),
            format(loss_test, '.3f'),
            format(AUC_test, '.3f'),
            format(AUPRC, '.3f'),
            format(rr1, '.5f'),
            format(rr2, ".5f")]
    return AUCs


def hyperparameter_setting(args, ae_model_info):
    ##### hERG #####
    #depth_e1 = 1
    #drop_ratio = 0.0
    #learning_rate = 1e-3

    ##### DILI #####
    #depth_e1 = 1
    #drop_ratio = 0.0
    #learning_rate = 5e-4

    ##### SkinReaction #####
    # depth_e1 = 4
    # drop_ratio = 0.0
    # learning_rate = 1e-3

    ##### carcino #####
    # depth_e1 = 4
    # drop_ratio = 0.3
    # learning_rate = 1e-3

    ##### DILIst #####
    depth_e1 = 2
    drop_ratio = 0.1
    learning_rate = 1e-3

    best_results = []

    ae_model_path, ae_model_type = ae_model_info[0]
    print(
        f"Test ae_model_path={ae_model_path}, depth_e1={depth_e1}, drop_ratio={drop_ratio}, learning_rate={learning_rate}")
    best_auc, best_epoch = model_run(args, ae_model_path, ae_model_type, depth_e1, drop_ratio,
                                     learning_rate)
    best_results.append((ae_model_path, depth_e1, drop_ratio, learning_rate, best_epoch, best_auc))

    best_results.sort(key=lambda x: x[5], reverse=True)

    result_file_path = f'result/{args.dataset_name}_test0304_seed42.txt'
    with open(result_file_path, 'w') as f:
        f.write("Model Path\tDepth_e1\tDrop Ratio\tLearning Rate\tEpoch\tBest AUC\n")
        for result in best_results:
            f.write(f"{result[0]}\t{result[1]}\t{result[2]}\t{result[3]}\t{result[4]}\t{result[5]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="DILIst")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--backbone', type=str, default="ViT")
    parser.add_argument('--graph_backbone', type=str, default="GIN_GAT")
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--depth_e1', type=int, default=1)
    parser.add_argument('--depth_e2', type=int, default=1)
    parser.add_argument('--depth_decoder', type=int, default=1)
    parser.add_argument('--lr_decay', type=float, default=0.85)
    parser.add_argument('--drop_ratio', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument("--device", default='cuda:0')
    opt = parser.parse_args()

    #ae_model_info = [
    #    ('data/hERG/autoencoder_layer2_ep60.pth', 'ep60'),
    #]

    #ae_model_info = [
    #    ('data/DILI/autoencoder_layer2_ep190.pth', 'ep190')
    #]

    # ae_model_info = [
    #      ('data/SkinReaction/autoencoder_layer2_ep190.pth', 'ep190')
    # ]

    ae_model_info = [
       ('data/DILIst/autoencoder_layer2_ep70.pth', 'ep70')
    ]

    # ae_model_info = [
    #     ('data/carcinogenecity/autoencoder_layer2_ep130.pth', 'ep130')
    # ]

    hyperparameter_setting(opt, ae_model_info)


def set_seed(seed):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy seed
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed
    torch.cuda.manual_seed_all(seed)  # Multi-GPU seed
    torch.backends.cudnn.deterministic = True  # CUDNN seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    set_seed(42)
    main()
