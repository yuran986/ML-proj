import heapq
import re
import torch
import abc_py as abcPy
import numpy as np
import os
from tools import get_aig_by_action, get_graphdata
from gnn import GNN
from torch_geometric.loader import DataLoader
from PredictValueNetwork_ import PredictValueNetwork

class BeamSearch:
    def __init__(self, beam_width):
        self.beam_width = beam_width

    def search(self, start_seq, max_length, model, model_weights1, model_weights2=None):
        
        searched_aig_dir = "./aig_searched/" 
        beam = [(0, start_seq)]  # (score, sequence)
        
        for _ in range(max_length):
            new_beam = []
            for score, sequence in beam:
                for action in range(7):  # Each node can expand to 7 child nodes (1 to 7)
                    aigs = os.listdir(searched_aig_dir)
                    new_aigFile = sequence+str(action)+'.aig'
                    new_aigFilePath = searched_aig_dir+new_aigFile
                    if new_aigFilePath in aigs:
                        aigFilePath = new_aigFilePath
                    else:
                        aigFilePath = get_aig_by_action(
                            state=sequence,
                            action=action,
                            logFile=f'beamsearch_{self.beam_width}.log',
                            aigDir=searched_aig_dir
                        )
                    new_sequence = sequence + f'{action}'
                    new_score = evaluate(aigFilePath, model,model_weights1, model_weights2)
                    new_beam.append((new_score, new_sequence))
            
            # Keep the top beam_width sequences
            beam = heapq.nlargest(self.beam_width, new_beam, key=lambda x: x[0])
            # print(beam)
        # Return the sequence with the highest score
        return max(beam, key=lambda x: x[0])


def evaluate(aigFielPath, model,model1, model2=None):
    if model2 == None:
        return evaluation(aigFielPath,model,model1)
    score = evaluation(aigFielPath, model, model1) + evaluation(aigFielPath, model, model2)
    return score


def evaluation(aigFilePath,model, model_weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(model_weights))
    model.eval()  # 设置模型为评估模式
    data = get_graphdata(aigFilePath=aigFilePath)
    data_list = [data]
    loader = DataLoader(data_list, batch_size=1, shuffle=True)
    # 将数据传入模型并获取输出
    with torch.no_grad():  # 评估模式不需要计算梯度
        for batch in loader:
            batch.to(device)
            output = model(batch)
    return output.item()


def get_evals(state):
     
    AIGPath = "./aig_searched/"  + state+'.aig'
    libFile = './lib/7nm/7nm.lib'
    logFile = "./log_output/" + state +'.aig'
    #Evaluate AIG
    abcRunCmd = "/mnt/e/Code/ML/oss-cad-suite/bin/yosys-abc -c \"read " + AIGPath + "; read_lib " + libFile + "; map ; topo;stime \" > " + logFile
    os.system(abcRunCmd)
    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        eval = float(areaInformation[-9]) * float(areaInformation[-4])
    print(f"Real eval for {state} is : {eval}")

    initalState = state.split('_')[0]
    # initalAIGPath = "./raw_data/InitialAIG/test" + initalState +'.aig'
    AIGPath_res = "./aig_searched/"  + initalState +'_resyn2.aig'
    #Evaluate and store the generated AIG
    RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
    abcRunCmd = "/mnt/e/Code/ML/oss-cad-suite/bin/yosys-abc -c \"read " + AIGPath + ";" + RESYN2_CMD + "read_lib " + libFile + ";  write " + AIGPath_res + "; map; topo; stime\" > " + logFile
    os.system(abcRunCmd)

    with open(logFile) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
        baseline = float(areaInformation[-9]) * float(areaInformation[-4])
    eval = 1 - eval / baseline
    print(f"Percentage increase for {state} is : {eval}")
    return eval

def save_results(data_list, file_path):
    """
    将列表中的数据追加到文本文件，每个元素占一行。
    
    参数：
    data_list -- 需要写入文件的列表
    file_path -- 输出文本文件的路径
    """
    try:
        model1 = "PredictValueNetwork"
        model2 = "Simple GNN"
        with open(file_path, 'a') as file:
            file.write(f"beam search with beam_width={beam_width} by decision_score, model={model1}: \n")
            for aig, best_sequence, eval in data_list:
                file.write(f"{aig}, {best_sequence}, {str(eval)}\n")
            file.write(f'----End for Search----\n')
        print(f"数据已成功追加到文件 {file_path}")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

# Usage of BeamSearch
beam_width = 3  # Adjust beam width as needed
max_length = 10  # Adjust max length as needed

test_path = './raw_data/InitialAIG/test'
# model_weights1="./models_network/model_weights/0.001_best.pth"
# model_weights2="./models_network2/model_weights/0.001_best.pth"
model_weights1= "./models_network2_shift/model_weights/0.001_best.pth"
model_weights2=None
model = PredictValueNetwork()

evals = []
test_files = os.listdir(test_path)


for aig in test_files:
    print('root: ', aig)

    start_sequence = aig.split('.')[0] + '_'

    beam_search = BeamSearch(beam_width)
    best_sequence = beam_search.search(start_seq=start_sequence, max_length=max_length,model=model, model_weights1=model_weights1,model_weights2=model_weights2)

    print("Best sequence:", best_sequence[1])
    print("Best score:", best_sequence[0])
    
    eval = get_evals(best_sequence[1])
    evals.append((aig, best_sequence[1], eval))

save_results(evals, "./search_results.txt")
print(f"best evals for test aig: {evals}")


