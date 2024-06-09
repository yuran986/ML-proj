import pickle
import os
import re
import numpy as np

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
    outputDir = "./all_aig/"
    logDir = "./log_output/"
    libFile = '7nm.lib'

    circuitName, actions = state.split('_')
    circuitPath = './InitialAIG/train/' + circuitName + '.aig'

    aigFile = state + '.aig'  # current AIG file
    aigFilePath = outputDir + aigFile

    logFile = 'load_data.log'
    logFilePath = logDir + logFile

    actionCmd = ''
    for action in actions:
        actionCmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = "oss-cad-suite\environment.bat && yosys-abc -c \"read " + circuitPath + ";" + actionCmd + "; read_lib " + libFile + ";  write " + aigFilePath + "; print_stats\" > " + logFilePath

    os.system(abcRunCmd)


def evaluate(state):
    outputDir = "./all_aig_test/"
    logDir = "./log_output/"
    libFile = '7nm.lib'

    circuitName, actions = state.split('_')
    circuitPath = './InitialAIG/test/' + circuitName + '.aig'

    aigFile = state + '.aig'  # current AIG file
    aigFilePath = outputDir + aigFile

    logFile = 'test_data.log'
    logFilePath = logDir + logFile

    actionCmd = ''
    for action in actions:
        actionCmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = "oss-cad-suite\environment.bat && yosys-abc -c \"read " + circuitPath + ";" + actionCmd + "; read_lib " + libFile + ";  write " + aigFilePath + "; print_stats\" > " + logFilePath
    os.system(abcRunCmd)


    abcRunCmd = "oss-cad-suite\environment.bat && yosys-abc -c \"read " + circuitPath + "; read_lib " + libFile + "; map ; topo;stime \" > " + logFilePath
    os.system(abcRunCmd)
    with open(logFilePath) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
    eval = float(areaInformation[-6]) * float(areaInformation[-3])

    RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
    abcRunCmd = "oss-cad-suite\environment.bat && yosys-abc -c \"read " + circuitPath + ";" + RESYN2_CMD + "read_lib " + \
                libFile + "; write " + aigFilePath + "; map; topo; stime\" > " + logFilePath
    os.system(abcRunCmd)

    with open(logFilePath) as f:
        areaInformation = re.findall('[a-zA-Z0-9.]+', f.readlines()[-1])
    baseline = float(areaInformation[-6]) * float(areaInformation[-3])
    eval = 1 - eval / baseline

    return eval

def test_generate():
    data_dir = "./InitialAIG/test"
    states = []
    for filename in os.listdir(data_dir):
        init_state, _ = filename.split('.')
        n_samples = 100
        for i in range(n_samples):
            temp = []
            # temp.append(i)
            current_state = init_state + "_"
            for j in range(10):
                current_state = current_state + str(int(np.random.randint(0, 6)))
                temp.append(current_state)
            states_set = set(states)
            for k in range(len(temp)):
                states_set.add(temp[k])
            states = list(states_set)

    state_list = list(states)
    print(len(state_list))
    state_list.sort()
    # print(state_list)
    evaluations = dict()

    for state in states:
        print(state)
        eval = evaluate(state)
        evaluations.update({state: eval})

    with open('test_evaluation.pkl', 'wb') as f:
        pickle.dump(evaluations, f)



if __name__ == "__main__":
    test_generate()
