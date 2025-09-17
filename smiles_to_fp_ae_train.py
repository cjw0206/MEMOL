import torch
from autoencoder import train_autoencoder

import numpy as np
import os
from molfeat.trans.fp import FPVecTransformer

def generate_pharm2D(smiles):
    fp_trans = FPVecTransformer('pharm2D', length=1024)
    fp = fp_trans(smiles)
    return fp[0]

def generate_fcfp(smiles):
    fp_trans = FPVecTransformer('fcfp', length=1024)
    fp = fp_trans(smiles)
    return fp[0]

def smile2fingerprint(data_root, type, file_data):
    with open(file_data, "r") as f:
        data_list = f.read().strip().split("\n")

    data_list = [d for d in data_list if '.' not in d.strip().split('\t')[0]]

    fingerprints = []

    for i, data in enumerate(data_list):
        if i % 50 == 0:
            print(f'{i + 1}/{len(data_list)}')
        smile = data.strip().split("\t")[0]
        print(f"{smile}")
        fp = np.concatenate([generate_pharm2D(smile), generate_fcfp(smile)], axis=0)
        fp_array = np.array(fp)
        fingerprints.append(fp_array)

    fingerprints_array = np.array(fingerprints)
    return fingerprints_array

if __name__ == '__main__':
    dataset_name = "DILIst"
    data_root = os.path.join("data", dataset_name)
    train_file = os.path.join(data_root, f"{dataset_name}_train.txt")
    test_file = os.path.join(data_root, f"{dataset_name}_test.txt")
    val_file = os.path.join(data_root, f"{dataset_name}_val.txt")

    fingerprint_train_path = os.path.join(data_root, "input", f"{dataset_name}_train_fingerprint.npy")
    fingerprint_test_path = os.path.join(data_root, "input", f"{dataset_name}_test_fingerprint.npy")
    fingerprint_val_path = os.path.join(data_root, "input", f"{dataset_name}_val_fingerprint.npy")

    if os.path.exists(fingerprint_train_path) and os.path.exists(fingerprint_test_path) and os.path.exists(fingerprint_val_path):
        print("Loading existing fingerprints...")
        fingerprints_train = np.load(fingerprint_train_path)
        fingerprints_test = np.load(fingerprint_test_path)
        fingerprints_val = np.load(fingerprint_val_path)
    else:
        print("Generating new fingerprints...")
        fingerprints_train = smile2fingerprint(data_root, "train", train_file)
        fingerprints_test = smile2fingerprint(data_root, "test", test_file)
        fingerprints_val = smile2fingerprint(data_root, "val", val_file)

        np.save(fingerprint_train_path, fingerprints_train)
        np.save(fingerprint_test_path, fingerprints_test)
        np.save(fingerprint_val_path, fingerprints_val)

    all_fingerprints = np.concatenate([fingerprints_train, fingerprints_test, fingerprints_val], axis=0)

    '''
     for epoch in range(10, 201, 10):
        # Train and save 2-layer autoencoder
        autoencoder_model_path = os.path.join(data_root, f"autoencoder_layer2_ep{epoch}.pth")
        encoder = train_autoencoder(all_fingerprints, epochs=epoch, autoencoder_type='layer2')
        torch.save(encoder.state_dict(), autoencoder_model_path)
        print(f'2-layer Autoencoder model saved to {autoencoder_model_path}')

        # Train and save 3-layer autoencoder
        # autoencoder_model_path = os.path.join(data_root, f"autoencoder_layer3_ep{epoch}.pth")
        # encoder = train_autoencoder(all_fingerprints, epochs=epoch, autoencoder_type='layer3')
        # torch.save(encoder.state_dict(), autoencoder_model_path)
        # print(f'3-layer Autoencoder model saved to {autoencoder_model_path}')
    '''
