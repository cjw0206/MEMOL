import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch_geometric.data import Data
import os
import pickle
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = torch.tensor(adj_matrix.nonzero()).to(torch.long)
    x = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.float).view(-1, 1)
    return Data(x=x, edge_index=edge_index)

def smiles_to_graph_nf8(smiles):
    mol = Chem.MolFromSmiles(smiles)
    adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = torch.tensor(adj_matrix.nonzero()).to(torch.long)

    # 추가적인 원자 특성 추출
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),  # 원자 번호
            atom.GetFormalCharge(),  # 형식 전하
            atom.GetTotalNumHs(),  # 총 수소 수
            atom.GetNumExplicitHs(),  # 명시적 수소 수
            atom.GetNumImplicitHs(),  # 암시적 수소 수
            int(atom.GetIsAromatic()),  # 방향족 여부
            atom.GetDegree(),  # 연결된 원자 수 (차수)
            atom.GetMass(),  # 원자 질량
        ])

    x = torch.tensor(atom_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


graph_dict=dict()
def smile2graph(data_root, type, file_data):
    with open(file_data, "r") as f:
        data_list = f.read().strip().split("\n")

    """Exclude data contains '.' in the SMILES format."""  # The '.' represents multiple chemical molecules
    data_list = [d for d in data_list if '.' not in d.strip().split('\t')[0]]

    graphs = []

    for i, data in enumerate(data_list):
        if i % 100 == 0:
            print(f'{i + 1}/{len(data_list)}')
        smile = data.strip().split("\t")[0]

        graph = smiles_to_graph(smile)

        graphs.append(graph)

    # Define the file path for saving
    save_path = os.path.join(data_root, "input", f"{dataset_name}_{type}_graph_nf1_Data.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(graphs, f)

if __name__ == '__main__':
    dataset_name = "DILIst"
    data_root = os.path.join("data", dataset_name)
    train_file = os.path.join(data_root, f"{dataset_name}_train.txt")
    test_file = os.path.join(data_root, f"{dataset_name}_test.txt")
    val_file = os.path.join(data_root, f"{dataset_name}_val.txt")


    smile2graph(data_root, "train", train_file)
    smile2graph(data_root, "test", test_file)
    smile2graph(data_root, "val", val_file)
