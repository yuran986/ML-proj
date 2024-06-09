import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from dataset_ import load_dataset, CustomDataset
from tools import save_checkpoint, load_checkpoint
import time
import pickle as pk
import os
import sys
import csv
import pandas as pd
from early_stopping import EarlyStopping
# from PredictValueNetwork_mlp import PredictValueNetwork
from PredictValueNetwork_ import PredictValueNetwork
from torch.optim.lr_scheduler import StepLR

'''
train_encoder_mlp.py: train PredictValueNetwork for task1
'''

def train_val(model, optimizer, criterion, train_loader, valid_loader, num_epochs, lr, model_savingpath, saving_path):
    if not os.path.exists(model_savingpath+f'/checkpoints'): os.mkdir(model_savingpath+f'/checkpoints')
    if not os.path.exists(model_savingpath+f'/model_weights'): os.mkdir(model_savingpath+f'/model_weights')
    early_stopping = EarlyStopping(patience=50, min_delta=0)
    best_valid_loss = float('inf')
    best_train_loss = float('inf')
    best_epoch = 1
    for epoch in range(1, num_epochs + 1):
        # 验证
        # model.eval()
        # valid_loss = 0
        # with torch.no_grad():
        #     for data in valid_loader:
        #         data = data.to(device)
        #         output = model(data)
        #         loss = criterion(output, data.y)
        #         valid_loss += loss.item()
        # average_valid_loss = valid_loss / len(valid_loader)
        # print(f"Epoch {epoch}/{num_epochs}, Valid Loss: {average_valid_loss:.10f}")
        # sys.exit()
        
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {average_loss:.10f}")

        average_valid_loss = '/'
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, average_loss, filename=model_savingpath+f'/checkpoints/checkpoint_{lr}_{epoch}.pth')

            # 验证
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for data in valid_loader:
                    data = data.to(device)
                    output = model(data)
                    loss = criterion(output, data.y)
                    valid_loss += loss.item()
            average_valid_loss = valid_loss / len(valid_loader)
            print(f"Epoch {epoch}/{num_epochs}, Valid Loss: {average_valid_loss:.10f}")

            if average_valid_loss < best_valid_loss:
                best_valid_loss = average_valid_loss
                best_train_loss = average_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_savingpath+f'/model_weights/{lr}_best.pth')

        # 将loss数据保存到CSV文件
        loss_res = {'lr': lr, 'epoch': epoch, 'train_loss': average_loss, 'val_loss': average_valid_loss}
        df = pd.DataFrame([loss_res])
        loss_path = model_savingpath+'/losses.csv'
        if not os.path.exists(loss_path):
            df.to_csv(loss_path, index=False, mode = 'a')
        else:
            df.to_csv(loss_path, index=False, mode = 'a', header = None)
        
        # 调用早停法，监控训练集上的损失
        early_stopping(average_loss)

        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出训练循环
    
    result = {
        'model': 'network',
        'lr': lr,
        'best_train_loss': best_train_loss,
        'best_valid_loss': best_valid_loss,
        'best_epoch': best_epoch
    }
    print(result)
    df = pd.DataFrame([result])
    if not os.path.exists(saving_path):
        df.to_csv(saving_path, index=False, mode = 'a')
    else:
        df.to_csv(saving_path, index=False, mode = 'a', header = None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t0 = time.time()
data_list = load_dataset(data_dir = "./sampled_dataset", label_dir = './project_data1')
t1 = time.time()
print(f'End loading {len(data_list)} files, cost {t1-t0} s. ')

model_savingpath = './models_network'
if not os.path.exists(model_savingpath): os.mkdir(model_savingpath)

saving_path = 'result.csv'
lrs = [0.001]

for lr in lrs:
    print(f'Start training ...... lr: {lr}')

    train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(train_list)
    test_dataset = CustomDataset(test_list)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = PredictValueNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    num_epochs = 100

    # train_only(model, optimizer, criterion, train_loader, num_epochs, lr, model_savingpath, saving_path)
    train_val(model=model, optimizer=optimizer, criterion=criterion, train_loader=train_loader, valid_loader=valid_loader, num_epochs=num_epochs, lr=lr, model_savingpath=model_savingpath, saving_path=saving_path)