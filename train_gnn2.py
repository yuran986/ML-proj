import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from dataset import load_dataset, CustomDataset
import time
from training import train_val
from gnn import GNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t0 = time.time()

data_list = load_dataset(data_dir = "./sampled_dataset", label_dir = './project_data2')
model_savingpath = './models2'

saving_path = 'result.csv'
lrs = [0.01]

for lr in lrs:
    print(f'Start training ...... lr: {lr}')

    train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(train_list)
    test_dataset = CustomDataset(test_list)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = GNN(in_channels=2, hidden_channels=16, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    criterion = nn.MSELoss()
    num_epochs = 500
    target = 'future rewards'

    train_val(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion, train_loader=train_loader, valid_loader=valid_loader, num_epochs=num_epochs, lr=lr, model_savingpath=model_savingpath, saving_path=saving_path, target=target)


# 五折交叉验证
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# saving_path = 'result2.csv'
# lrs = [0.01]
# for lr in lrs:
#     print(f'Start training with learning rate: {lr}')

#     for fold, (train_idx, valid_idx) in enumerate(kf.split(data_list)):
#         print(f'Fold {fold}')

#         train_data = [data_list[i] for i in train_idx]
#         valid_data = [data_list[i] for i in valid_idx]

#         train_dataset = CustomDataset(train_data)
#         valid_dataset = CustomDataset(valid_data)

#         train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#         valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

#         model = GNN(in_channels=2, hidden_channels=16, out_channels=1).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#         criterion = nn.MSELoss()
#         num_epochs = 200

#         train_val(model, optimizer, criterion, train_loader, valid_loader, num_epochs, lr, fold, model_savingpath, saving_path)
#         # train_only(model, optimizer, criterion, train_loader, valid_loader, num_epochs, lr, model_savingpath, saving_path)
