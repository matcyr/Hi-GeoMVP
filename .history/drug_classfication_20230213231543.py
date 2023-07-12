# %%
import torch_geometric.nn as pyg_nn
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from prepare_data.DRP_loader import *

# %%
class Drug_Graph_Classifier(nn.Module):
    def __init__(self,model_config):
        super(Drug_Graph_Classifier, self).__init__()
        self.embed_dim = model_config.get('embed_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.layer_num = model_config.get('layer_num')
        self.readout = model_config.get('readout')
        self.atom_int_embed_nn = torch.nn.Embedding(get_atom_int_feature_dims()[0], self.embed_dim)
        torch.nn.init.xavier_uniform_(self.atom_int_embed_nn.weight.data)
        self.bond_int_embed_nn = torch.nn.Embedding(get_bond_feature_int_dims()[0] + 3, self.embed_dim)
        torch.nn.init.xavier_uniform_(self.bond_int_embed_nn.weight.data)
        self.atom_conv = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        for i in range(self.layer_num):
            self.atom_conv.append(pyg_nn.GINEConv(nn = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim)),edge_dim=self.embed_dim))
            self.batch_norm.append(nn.BatchNorm1d(self.embed_dim))
        if self.readout == 'max':
            self.read_out = pyg_nn.global_max_pool
        elif self.readout == 'mean':
            self.read_out = pyg_nn.global_mean_pool
        self.classifier = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 23))
        self.regr = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 1))
    def forward(self,drug_atom):
        x, edge_index, edge_attr, batch = drug_atom.x, drug_atom.edge_index, drug_atom.edge_attr, drug_atom.batch
        x = self.atom_int_embed_nn(x[:, 0].to(dtype=torch.int64))
        edge_attr = self.bond_int_embed_nn(edge_attr[:, 0].to(dtype=torch.int64))
        hidden = [x]
        for i in range(self.layer_num):
            x = self.atom_conv[i](x = x, edge_attr = edge_attr, edge_index = edge_index)
            x = self.batch_norm[i](x)
            hidden.append(x)
        x = hidden[-1]
        graph_repr = self.read_out(x, batch)
        graph_pred_class = self.classifier(graph_repr)
        graph_pred_regr = self.regr(graph_repr)
        return graph_pred_class, graph_pred_regr
        


# %%
model_config = {'embed_dim':32,'dropout_rate':0.2,'layer_num':3,'readout':'max'}
def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)
criterion_class = cross_entropy_one_hot
def criterion_regre(input, target):
    target = target.view(-1, 1)
    return nn.MSELoss()(input, target)
def total_loss(class_drug,repr_drug, target, threshold, lamda = torch.tensor(1.0)):
    return lamda * criterion_class(class_drug,target) + criterion_regre(repr_drug,threshold)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Drug_Graph_Classifier(model_config)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = total_loss

# %%
train_drug_id,test_drug_id = split_drug_set()
train_drug_set = drug_classfication_dataset(drug_idx = train_drug_id)
test_drug_set  = drug_classfication_dataset(drug_idx = test_drug_id)
train_loader = drug_class_loader(train_drug_set, batch_size = 200)
test_loader = drug_class_loader(test_drug_set, batch_size = 200)
use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
# %%
def train(model, train_loader, optimizer, loss_op, device):
    model.train()
    total_loss = []
    for data in train_loader:
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            drug_atom, drug_bond, target,threshold  = data
            drug_atom, target,threshold = drug_atom.to(device), target.to(device),threshold.float().to(device)
            optimizer.zero_grad()
            drug_class, drug_regr = model(drug_atom)
            loss = loss_op(drug_class, drug_regr, target, threshold)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss.append(loss.item())
        optimizer.step()
    train_loss = np.average(total_loss)
    return np.sqrt(train_loss)

@torch.no_grad()
def test(model,loader,device):
    model.eval()
    y_true, preds = [], []
    correct = 0
    for data in tqdm(loader):
        drug_atom, drug_bond, target,threshold  = data
        drug_atom, target,threshold = drug_atom.to(device), target.to(device),threshold.to(device)
        drug_class, drug_regr = model(drug_atom)
        pred = drug_class.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == target.max(dim=1)[1]).sum())
        y_true.append(threshold.view(-1, 1).float())
        preds.append(drug_regr.float().cpu())
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(preds, dim=0).numpy()
    rmse = mean_squared_error(y_true,y_pred, squared=False)
    r = pearsonr(y_true.flatten(), y_pred.flatten())[0]
    correct_portion = correct / len(loader.dataset)
    return rmse,r,correct_portion

for epoch in range(1, 100):
    train(model, train_loader, optimizer, criterion, device)
    train_rmse,train_r,train_acc = test(model,train_loader,device)
    test_rmse,test_r,test_acc = test(model,test_loader,device)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


