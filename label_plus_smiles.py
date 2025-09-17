import os
import numpy as np

def label_save(data_file, file_type):
    with open(data_file, 'r') as f:
        data_list = f.read().strip().split('\n')

    """Exclude data contains '.' in the SMILES format."""  # The '.' represents multiple chemical molecules
    data_list = [d for d in data_list if '.' not in d.strip().split('\t')[0]]
    N = len(data_list)
    print("label----dataset_size: ", N)

    active_num = 0
    inactive_num = 0
    activities = []
    smiles = []
    for i, data in enumerate(data_list):
        smile, active = data.strip().split('\t')
        
        smiles.append(np.array(smile))
        active = int(float(active))
        activities.append(np.array([active], dtype=np.float32))
        if active == 1:
            active_num += 1
        elif active == 0:
            inactive_num += 1
    print("numbers of active samples:", active_num)
    print("numbers of inactive samples:", inactive_num)

    smiles_name = os.path.join(data_root, "input", f"{dataset_name}_{file_type}_smiles.npy")
    label_name = os.path.join(data_root, "input", f"{dataset_name}_{file_type}_activities.npy")

    np.save(smiles_name, smiles)
    np.save(label_name, activities)


if __name__ == '__main__':
    dataset_name = "DILIst"
    data_root = os.path.join("data", dataset_name)
    train_file = os.path.join(data_root, f"{dataset_name}_train.txt")
    test_file = os.path.join(data_root, f"{dataset_name}_test.txt")
    val_file = os.path.join(data_root, f"{dataset_name}_val.txt")

    input_file = os.path.join(data_root, "input")
    if not os.path.exists(input_file):
        os.makedirs(input_file)

    label_save(train_file,'train')
    label_save(test_file,'test')
    label_save(val_file,'val')
