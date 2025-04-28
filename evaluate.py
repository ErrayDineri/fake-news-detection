import os, torch
from torch_geometric.datasets import UPFD
from torch_geometric.transforms import ToUndirected
from detector.gcn_predict import GNNFakeNews

# 1) Load the model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'gnn_fakenews.pt')
model = GNNFakeNews('GCN', in_channels=768, hidden_channels=128, out_channels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# 2) Load the test split
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'detector', 'data', 'UPFD')
dataset = UPFD(DATA_ROOT, name='gossipcop', feature='bert',
               split='test', transform=ToUndirected())

# 3) Run through every example and accumulate accuracy
fake_corr = fake_tot = real_corr = real_tot = 0
for data in dataset:
    data = data.to(DEVICE)
    with torch.no_grad():
        out  = model(data.x, data.edge_index, data.batch)
        pred = int(out.argmax(dim=1).item())
        true = int(data.y.item())

    if true == 0:
        fake_tot  += 1
        fake_corr += (pred == true)
    else:
        real_tot  += 1
        real_corr += (pred == true)

print(f"Fake  accuracy: {fake_corr}/{fake_tot} = {fake_corr/fake_tot:.2%}")
print(f"Real  accuracy: {real_corr}/{real_tot} = {real_corr/real_tot:.2%}")
print(f"Overall    : {(fake_corr+real_corr)}/{fake_tot+real_tot} = {(fake_corr+real_corr)/(fake_tot+real_tot):.2%}")
