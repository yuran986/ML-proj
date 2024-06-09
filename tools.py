import pickle as pk
import os
import torch
import abc_py as abcPy
import numpy as np
from torch_geometric.data import Data

'''
test.py: 所有获取单个aig文件或单个图信息的函数
'''

synthesisOpToPosDic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }

# state -> aigFile, return aigFilePath
def get_aig_from_state(state, outputDir = "/mnt/e/Code/ML/data/sampled_aig", logFile="ouput.log"):
    logDir = "/mnt/e/Code/ML/log_output/"
    libFile = './lib/7nm/7nm.lib'

    circuitName, actions = state.split('_')
    circuitPath = './raw_data/InitialAIG/train/' + circuitName + '.aig'
    
    aigFile = state + '.aig' # current AIG file
    aigFilePath = outputDir + aigFile

    logFilePath = logDir+logFile
    #Generate the AIG (initial AIG taking actionCmd) and stored in nextState
    actionCmd = ''
    for action in actions:
        actionCmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = "/mnt/e/Code/ML/oss-cad-suite/bin/yosys-abc -c \"read " + circuitPath + ";" + actionCmd + "; read_lib " + libFile + ";  write " + aigFilePath + "; print_stats\" > " + logFilePath
    os.system(abcRunCmd)
    return aigFilePath

# initial aigFile + action -> new_aigFile, return new_aigFliePath
def get_aig_by_action(state="alu4_0", action=1, logFile="output.log", aigDir="./searchOutput/"):
    if state[-1]=='_':
        aigFile = state[:-1]+'.aig'
    else: 
        aigFile = state + '.aig'
    libFile = './lib/7nm/7nm.lib'
    logDir = "/mnt/e/Code/ML/log_output/"
    logFilePath = logDir+logFile
    aigFilePath = aigDir+aigFile
    
    new_aigFile = state+str(action)+'.aig'
    new_aigFilePath = aigDir+new_aigFile
    actionCmd = synthesisOpToPosDic[action]
    abcRunCmd = "/mnt/e/Code/ML/oss-cad-suite/bin/yosys-abc -c \"read " + aigFilePath + ";" + actionCmd + "; read_lib " + libFile + ";  write " + new_aigFilePath + "; print_stats\" > " + logFilePath
    os.system(abcRunCmd)
    return new_aigFilePath

# aigFile -> GraphData -> GNN input data
def get_graphdata(aigFilePath):
    data = get_data_from_aig(aigFilePath)
    # 准备节点特征
    node_type = data['node_type'].float().unsqueeze(1)  # 将维度扩展为 (num_nodes, 1)
    num_inverted_predecessors = data['num_inverted_predecessors'].float().unsqueeze(1)  # 将维度扩展为 (num_nodes, 1)
    x = torch.cat([node_type, num_inverted_predecessors], dim=1)
    # 创建Data对象
    data = Data(x=x, edge_index=data['edge_index'])
    return data

# aigFile -> GraphData 
def get_data_from_aig(aigFilePath):
    _abc = abcPy.AbcInterface()
    _abc.start()
    _abc.read(aigFilePath)
    data = {}
    numNodes = _abc.numNodes()
    data['node_type'] = np.zeros(numNodes, dtype=int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)
    edge_src_index = []
    edge_target_index = []
    for nodeIdx in range(numNodes):
        aigNode = _abc.aigNode(nodeIdx)
        # print(aigNode)
        nodeType = aigNode.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0
        if nodeType == 0 or nodeType == 2:
            data['node_type'][nodeIdx] = 0
        elif nodeType == 1:
            data['node_type'][nodeIdx] = 1
        else:
            data['node_type'][nodeIdx] = 2
            if nodeType == 4:
                data['num_inverted_predecessors'][nodeIdx] = 1
            if nodeType == 5:
                data['num_inverted_predecessors'][nodeIdx] = 2
        if (aigNode.hasFanin0()):
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
        if (aigNode.hasFanin1()):
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
    data['node_type'] = torch.tensor(data['node_type'])
    data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
    data['nodes'] = numNodes
    # print(data['edge_index'].shape)
    # print(data['node_type'].shape)
    # print(data['num_inverted_predecessors'].shape)
    return data