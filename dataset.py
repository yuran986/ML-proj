import pickle as pk
import os
from tools import get_aig_from_state, get_data_from_aig
import psutil
import gc
import random

random.seed(42)

'''
dataset.py: 预处理数据，对数据进行降采样并定义获取state2label和state2data的函数等
'''

def get_sampled_pkl(directory_path, output_filename='sampled_pkl.pkl'):
    filenames = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.pkl'):
            filenames.append(filename)
    
    sampled_filenames = random_sample_list(filenames, 0.05)
    # Save the new file names to a .pkl file
    output_file_path = os.path.join(directory_path, output_filename)
    with open(output_file_path, 'wb') as f:
        pk.dump(sampled_filenames, f)
    
    return sampled_filenames

# 生成所有aig文件
def get_data_list(sampled_pkl= "./sampled_pkl.pkl", pkl_dir = "./raw_data/project_data/",output_dir = "./sampled_aig/"):
    print('Start loading data_list...')
    os.makedirs(output_dir, exist_ok=True)
    sampled_pkl = load_from_file(sampled_pkl)
    for pkl in sampled_pkl:
        pklPath = os.path.join(pkl_dir, pkl)
        pklFile = load_from_file(pklPath)
        for state, _ in zip(pklFile['input'], pklFile['target']):
            get_aig_from_state(state)

# 获取字典文件：dict: {state: label} 包含数据集中所有state
def get_state2label(project_data_dir="./raw_data/project_data", save_path="./states2label.pkl"):
    print('Start geting label_dict...')
    pkl_list = os.listdir(project_data_dir)
    state2label = {}
    for pkl in pkl_list:
        pkl_path = os.path.join(project_data_dir, pkl)
        with open(pkl_path, "rb") as f:
            file = pk.load(f)
            for state, label in zip(file['input'], file['target']):
                if state in state2label and state2label[state] != label:
                    print(f"Conflicting labels for state {state}: {state2label[state]} vs {label}")
                state2label[state] = label
    with open(save_path, "wb") as f:
        pk.dump(save_path, f)
    print("Finished saving label_dict...")
    return state2label


# 获取字典文件：dict: {state: GraphData} ，并分块存储，仅包含降采样的训练集和生成的测试集中的state
def get_state2data(data_dir = "./testset/all_aig_test", output_dir="./chunks_test"):
    print('Start saving state2data...')
    os.makedirs(output_dir, exist_ok=True)

    aigs = os.listdir(data_dir)
    chunk_size = 3000  # 每块包含的元素数量
    # 将列表分割成多个子列表
    aig_chunks = split_list(aigs, chunk_size)
    for i, aig_chunk in enumerate(aig_chunks):
        chunk = {}
        for filename in aig_chunk:
            if filename.endswith('.aig'):
                state = filename.split('.')[0]
                filepath = os.path.join(data_dir, filename)
                data = get_data_from_aig(filepath)
                chunk[state] = data

        save_path = os.path.join(output_dir, f'state2data_part_{i + 1}.pkl')
        save_to_file(chunk, save_path)
        print(f"Saved {filename} with {len(chunk)} elements")
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Memory used: {memory_info.rss / 1024 ** 2:.2f} MB")
        # 手动删除已保存的子列表并强制垃圾回收
        del chunk
        gc.collect()

    print("Finished saving state2data...")
    

def merge_chunks(output_dir="./chunks"):
    list = os.listdir(output_dir)
    data_list=[]
    for i in range(len(list)):
        filename = os.path.join(output_dir, f'data_list_part_{i + 1}.pkl')
        loaded_chunk = load_from_file(filename)
        print(f"Part {i + 1} loaded successfully with {len(loaded_chunk)} elements")
        data_list.extend(loaded_chunk)
    return data_list



def random_sample_list(input_list, sampling_ratio):
    sample_size = int(len(input_list) * sampling_ratio)
    if sample_size > len(input_list):
        raise ValueError("Sample size cannot be greater than the size of the input list")
    return random.sample(input_list, sample_size)

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
    

# if __name__ == "__main__":
    # get_state2label()
    # get_state2data()
    

