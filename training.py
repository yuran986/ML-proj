import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from dataset import load_dataset, split_list, save_to_file, load_from_file, CustomDataset
from tools import save_checkpoint, load_checkpoint
import time
import pickle as pk
import os
import csv
import pandas as pd
from early_stopping import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_val(model, optimizer, criterion, train_loader, valid_loader, num_epochs, lr, model_savingpath, saving_path, target):
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
        loss_path = model_savingpath+f'/losses_{lr}.csv'
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
        'target': target,
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

def train_only(model, optimizer, criterion, train_loader, num_epochs, lr, model_savingpath, saving_path):
    early_stopping = EarlyStopping(patience=50, min_delta=0)
    training_losses = []
    best_loss = float('inf')
    best_epoch = 1
    for epoch in range(1, num_epochs + 1):
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
        training_losses.append(average_loss)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {average_loss:.10f}")

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, average_loss, filename=model_savingpath+f'/checkpoints/checkpoint_{lr}_{epoch}.pth')

            if average_loss < best_loss:
                best_loss = average_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_savingpath+f'/model_weights/{lr}_best.pth')

        # 调用早停法，监控验证集上的损失
        early_stopping(average_loss)

        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出训练循环

    # 存储losses
    training_losses = torch.tensor(training_losses)
    torch.save(training_losses, model_savingpath+f'training_losses_{lr}_{num_epochs}.pth')

    # 绘制losses图
    epochs = [i for i in range(1, len(training_losses)+1)]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_losses, label='Training Loss')
    # plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(model_savingpath+f'training_losses_{lr}.png')

    result = {
        'model': 'eval',
        'lr': lr,
        'fold': '/',
        'best_loss': best_loss,
        'best_epoch': best_epoch
    }
    print(result)
    df = pd.DataFrame([result])
    if not os.path.exists(saving_path):
        df.to_csv(saving_path, index=False, mode = 'a')
    else:
        df.to_csv(saving_path, index=False, mode = 'a', header = None)  

# def resume_training(model, optimizer, criterion, loader, start_epoch, num_epochs, lr):
#     for epoch in range(start_epoch+1, num_epochs+1):
#         model.train()
#         total_loss = 0
#         for data in loader:
#             data = data.to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, data.y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         average_loss = total_loss / len(loader)
#         print(f"Epoch {epoch}/{num_epochs}, Average Loss: {average_loss:.4f}")
#         save_checkpoint(model, optimizer, epoch, loss, filename=f'./checkpoints/4_checkpoint_epoch_{lr}_{epoch}.pth')