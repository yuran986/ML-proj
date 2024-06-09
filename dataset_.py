import pickle as pk
import os
import torch
from torch_geometric.data import Data, Dataset
# from tools import get_aig_from_state, get_data_from_aig
import sys


class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)  # 仅包含一个图数据

    def __getitem__(self, idx):
        return self.data_list[idx]

def load_dataset_test(data_dir = "./test_dataset", label_dir = './test_dataset/test_evaluation.pkl'):
    print('Start loading dataset...')
    # 加载图信息到图信息字典 graph_dict: {state: graph}
    graph_dict = {}
    chunks_dir = os.path.join(data_dir, 'chunks')
    for chunk in os.listdir(chunks_dir):
        chunk_name = os.path.join(chunks_dir, chunk)
        graph_dict.update(load_from_file(chunk_name))
    print('Graph loaded. ')

    label_dict = load_from_file(label_dir)
    print('Label loaded. ')

    # 将图信息和label对应起来
    data_list = []
    for state in label_dict.keys():
        data = graph_dict[state]
        node_type = data['node_type'].float().unsqueeze(1)  # 将维度扩展为 (num_nodes, 1)
        num_inverted_predecessors = data['num_inverted_predecessors'].float().unsqueeze(1)
        x = torch.cat([node_type, num_inverted_predecessors], dim=1) 
        label = label_dict[state]
        data = Data(x=x, edge_index=data['edge_index'], y=torch.tensor([label]))
        data_list.append(data)
    return data_list

def load_dataset(data_dir = "./sampled_dataset", label_dir = './project_data1'):
    print('Start loading dataset...')
    # 加载图信息到图信息字典 graph_dict: {state: graph}
    graph_dict = {}
    chunks_dir = os.path.join(data_dir, 'chunks')
    for chunk in os.listdir(chunks_dir):
        chunk_name = os.path.join(chunks_dir, chunk)
        graph_dict.update(load_from_file(chunk_name))

    sample_dir = os.path.join(data_dir, 'sampled_pkl.pkl')
    sample_list = load_from_file(sample_dir)

    # 将图信息和label对应起来
    data_list = []
    for sample in sample_list:
        prj_data_dir = os.path.join(label_dir, sample)
        label_dict = load_from_file(prj_data_dir)
        for i in range(len(label_dict['input'])):
            # print(label_dict['input'][i])
            data = graph_dict[label_dict['input'][i]]
            node_type = data['node_type'].float().unsqueeze(1)  # 将维度扩展为 (num_nodes, 1)
            num_inverted_predecessors = data['num_inverted_predecessors'].float().unsqueeze(1)
            x = torch.cat([node_type, num_inverted_predecessors], dim=1) 
            label = label_dict['target'][i]
            data = Data(x=x, edge_index=data['edge_index'], y=torch.tensor([label]))
            data_list.append(data)
    return data_list

# def load_dataset_network(data_dir = "./sampled_dataset", label_dir = './project_data1'):
#     print('Start loading dataset...')
#     # 加载图信息到图信息字典 graph_dict: {state: graph}
#     graph_dict = {}
#     chunks_dir = os.path.join(data_dir, 'chunks')
#     for chunk in os.listdir(chunks_dir):
#         chunk_name = os.path.join(chunks_dir, chunk)
#         graph_dict.update(load_from_file(chunk_name))

#     # print(graph_dict.keys())
#     # if 'adder_' in graph_dict.keys():
#     #     print('adder_!')
#     # if 'adder' in graph_dict.keys():
#     #     print('adder!')
#     # sys.exit()

#     sample_dir = os.path.join(data_dir, 'sampled_pkl.pkl')
#     sample_list = load_from_file(sample_dir)

#     # 将图信息和label对应起来
#     data_list = []
#     for sample in sample_list:
#         prj_data_dir = os.path.join(label_dir, sample)
#         label_dict = load_from_file(prj_data_dir)
#         for i in range(len(label_dict['input'])):
#             # print(label_dict['input'][i])
#             data = graph_dict[label_dict['input'][i]]
#             node_type = data['node_type'].float().unsqueeze(1)  # 将维度扩展为 (num_nodes, 1)
#             num_inverted_predecessors = data['num_inverted_predecessors'].float().unsqueeze(1)
#             x = torch.cat([node_type, num_inverted_predecessors], dim=1) 
#             label = label_dict['target'][i]
#             data = Data(node_type=data['node_type'], edge_index=data['edge_index'], y=torch.tensor([label]))
#             data_list.append(data)
#     return data_list

def load_dataset_shift(data_dir = "./sampled_dataset", label_dir = './project_data2'):
    print('Start loading shift dataset...')
    # 加载图信息到图信息字典 graph_dict: {state: graph}
    graph_dict = {}
    chunks_dir = os.path.join(data_dir, 'chunks')
    for chunk in os.listdir(chunks_dir):
        chunk_name = os.path.join(chunks_dir, chunk)
        graph_dict.update(load_from_file(chunk_name))

    sample_dir = os.path.join(data_dir, 'sampled_pkl.pkl')
    sample_list = load_from_file(sample_dir)

    # 将图信息和label对应起来; data_dict: {state: (graph, label)}
    data_dict = {}
    for sample in sample_list:
        prj_data_dir = os.path.join(label_dir, sample)
        label_dict = load_from_file(prj_data_dir)
        for i in range(1, len(label_dict['input'])):
            # print(label_dict['input'][i])
            label = label_dict['target'][i-1]
            if label_dict['input'][i] in data_dict.keys():
                if label > data_dict[label_dict['input'][i]][1]:
                    data_dict[label_dict['input'][i]][1] = label
            else: 
                data = graph_dict[label_dict['input'][i]]
                data_dict[label_dict['input'][i]] = [data, label]
    
    data_list = []
    for tuple in data_dict.values():
        data, label = tuple[0], tuple[1]
        node_type = data['node_type'].float().unsqueeze(1)  # 将维度扩展为 (num_nodes, 1)
        num_inverted_predecessors = data['num_inverted_predecessors'].float().unsqueeze(1)
        x = torch.cat([node_type, num_inverted_predecessors], dim=1) 
        data = Data(x=x, edge_index=data['edge_index'], y=torch.tensor([label]))
        data_list.append(data)

    return data_list



def split_list(data_list, chunk_size):
    """将列表按块大小分割成多个子列表"""
    return [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]

def save_to_file(data, filename):
    """将数据保存到文件中"""
    with open(filename, 'wb') as f:
        pk.dump(data, f)

def load_from_file(filename):
    """从文件中加载数据"""
    with open(filename, 'rb') as f:
        return pk.load(f)
