import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# 定义一个简单的 GNN 模型
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = torch.nn.Linear(out_channels, 1)  # Fully connected layer for final prediction

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        # Global pooling
        x = global_mean_pool(x, batch)  # shape: (batch_size, out_channels)
        # Final prediction
        x = self.fc(x)  # shape: (batch_size, 1)
        x = x.squeeze()
        return x