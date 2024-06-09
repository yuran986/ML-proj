import pickle as pk
import os




def get_label_dict1(project_data_dir="./raw_data/project_data", save_path="./states_dict2.pkl"):
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
    # with open(save_path, "wb") as f:
    #     pk.dump(save_path, f)
    # print("Finished saving label_dict...")
    state_list = list(state2label)
    print(len(state_list))



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
    

    
def load_data_list(output_dir):
    list = os.listdir("./chunks")
    data_list=[]
    for i in range(len(list)):
        filename = os.path.join(output_dir, f'data_list_part_{i + 1}.pkl')
        loaded_chunk = load_from_file(filename)
        print(f"Part {i + 1} loaded successfully with {len(loaded_chunk)} elements")
        data_list.extend(loaded_chunk)

def random_sample_list(input_list, sampling_ratio):
    sample_size = int(len(input_list) * sampling_ratio)
    if sample_size > len(input_list):
        raise ValueError("Sample size cannot be greater than the size of the input list")
    return random.sample(input_list, sample_size)

import random
import numpy as np
random.seed(42)

synthesisOpToPosDic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }

def get_aig_from_state(state):
    outputDir = "/home/wangyaoyao/projs/ML/project/final_aig/"
    logDir = "/home/wangyaoyao/projs/ML/project/log_output/"
    libFile = './lib/7nm/7nm.lib'

    circuitName, actions = state.split('_')
    circuitPath = './InitialAIG/train/' + circuitName + '.aig'
    
    aigFile = state + '.aig' # current AIG file
    aigFilePath = outputDir + aigFile

    logFile = './output.log'
    #Generate the AIG (initial AIG taking actionCmd) and stored in nextState
    actionCmd = ''
    for action in actions:
        actionCmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = "/home/wangyaoyao/projs/ML/oss-cad-suite/bin/yosys-abc -c \"read " + circuitPath + ";" + actionCmd + "; read_lib " + libFile + ";  write " + aigFilePath + "; print_stats\" > " + logFile
    os.system(abcRunCmd)

import shutil
import os

def copy_files(src_dir, dst_dir, file_name):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    src_file = os.path.join(src_dir, file_name)
    dst_file = os.path.join(dst_dir, file_name)
        
    if os.path.isfile(src_file):
        shutil.copy(src_file, dst_file)
    else:
        print("No such file.")


if __name__ == "__main__":
    # get_label_dict()
    # save_data_list()
    # load_data_list()
    data_dir = "/home/wangyaoyao/projs/ML/project/project_data/"
    # pkl_list = os.listdir(data_dir)
    # sampling_ratio = 0.1  # 采样比例
    # sampling_size = int(len(pkl_list)*sampling_ratio)
    # random_sampled_list = np.random.choice(pkl_list, sampling_size, replace=False)
    random_sampled_list = load_from_file( "./sampled_pkl.pkl")
    l = len(random_sampled_list)//2
    pkl_list = random_sampled_list[:l]
    print("降采样的pkl文件数量", len(pkl_list))
    save_to_file(pkl_list, "./pkl_list.pkl")
    # print(len(random_sampled_list))  # 输出: 随机采样的50%元素
    state2eval = {}
    for pkl in pkl_list:
        path = os.path.join(data_dir, pkl)
        file = load_from_file(path)
        for state,eval in zip(file['input'],file['target']):
            state2eval[state] = eval
    print(len(state2eval.keys()))

    states = list(state2eval.keys()) # 需要获取aig的states
    src_directory1 = "/home/wangyaoyao/projs/ML/project/sampled_aig/"
    src_directory2 = "/home/wangyaoyao/projs/ML/project/resampled_aig/"
    dst_directory = "/home/wangyaoyao/projs/ML/project/final_aig/"

    resampled_states = os.listdir(src_directory2) # resampled_aig下有的
    sampled_states = load_from_file("./sampled_states.pkl") # sampled_aig下有的
    print(len(sampled_states))
    print(sampled_states[0])
    sampled_states = dict(sampled_states)
    for i, state in enumerate(states):
        filename = state+'.aig'
        if state in resampled_states:
            copy_files(src_directory2,dst_directory, filename)
        elif state in sampled_states:
            copy_files(src_directory1,dst_directory, filename)
        else:
            get_aig_from_state(state)
        print(i)

    # path = "/home/wangyaoyao/projs/ML/project/resampled_aig/"
    # src_directory = "/home/wangyaoyao/projs/ML/project/sampled_aig/"
    # dst_directory = "/home/wangyaoyao/projs/ML/project/resampled_aig/"
    
    # resampled_states = os.listdir(path) # resampled_aig下有的
    # # state2eval = load_from_file("./resample_state2eval.pkl") # 字典
    # print(len(state2eval.keys()))
    # sampled_states = load_from_file("./sampled_states.pkl") # sampled_aig下有的
    # print(len(sampled_states))
    # print(sampled_states[0])
    # sampled_states = dict(sampled_states)
    # states = list(state2eval.keys())
    # for state in states[38177:]:
    #     print(state)
    #     if state in resampled_states:
    #         continue
    #     if state in sampled_states:
    #         filename = state+'.aig'
    #         copy_files(src_directory,dst_directory, filename)
    #         print("copy!")
    #     else:
    #         get_aig_from_state(state)
    

    


