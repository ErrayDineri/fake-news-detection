import os.path as osp
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_max_pool
from torch_geometric.transforms import ToUndirected

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='gossipcop', choices=['politifact', 'gossipcop'])
parser.add_argument('--feature', type=str, default='bert', choices=['profile', 'spacy', 'bert', 'content'])
parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GAT', 'SAGE'])
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

# --- Data ---
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'UPFD')
train_dataset = UPFD(path, args.dataset, args.feature, 'train', transform=ToUndirected())
val_dataset = UPFD(path, args.dataset, args.feature, 'val', transform=ToUndirected())
test_dataset = UPFD(path, args.dataset, args.feature, 'test', transform=ToUndirected())

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# --- Model ---
class GNNFakeNews(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, out_channels):
        super().__init__()
        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNFakeNews(args.model, train_dataset.num_features, 128, train_dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

# --- Train ---
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

# --- Evaluate ---
@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += data.num_graphs
    return correct / total

# --- Training Loop ---
def main():
    best_val = 0
    for epoch in range(1, args.epochs + 1):
        loss = train()
        val_acc = evaluate(val_loader)
        test_acc = evaluate(test_loader)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), 'gnn_fakenews.pt')
        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

    print("Training finished. Best model saved as `gnn_fakenews.pt`.")

if __name__ == '__main__':
    main()
