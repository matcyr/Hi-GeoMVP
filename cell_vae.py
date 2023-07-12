# %%
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

# %%
from prepare_data.create_cell_feat import *

# %%
import pandas as pd
import random
GE = pd.read_csv('/home/yurui/Atten_Geom_DRP/Data/GDSC_data/Table_S1_GDSC_Gene_expression.csv')
##normalize
GE = (GE - GE.min().min()) / (GE.max().max() - GE.min().min())

# %%
## Read cell type
import json
cancer_type = json.load(open('/home/yurui/Atten_Geom_DRP/Data/GDSC_data/type_gdsccell.json'))

# %%
cell_cancer_type = pd.DataFrame(index= GE.index, columns=cancer_type.keys())
for type in cancer_type.keys():
    for cell in cell_cancer_type.index:
        if cell in cancer_type[type]:
            cell_cancer_type.loc[cell, type] = 1
        else:
            cell_cancer_type.loc[cell, type] = 0
cell_cancer_type = cell_cancer_type.astype(np.int32)            


# %%
num_cell = GE.shape[0]
train_portion = 0.2
cell_name = GE.index
test_cell_id = random.sample(range(num_cell), int(train_portion*num_cell))
test_cell = cell_name[test_cell_id]
train_cell = [cell for cell in cell_name if cell not in test_cell]

# %%
GE_test = GE.loc[test_cell].astype(np.float32)
GE_train = GE.loc[train_cell].astype(np.float32)

# %%
class Gene_expression_dataset(Dataset):
    def __init__(self, train = 'train') -> None:
        super(Gene_expression_dataset, self).__init__()
        if train == 'train':
            self.dataset = torch.tensor(GE_train.values)
            self.y = torch.tensor(cell_cancer_type.loc[train_cell].values).long()
        else: 
            self.dataset = torch.tensor(GE_test.values)
            self.y = torch.tensor(cell_cancer_type.loc[test_cell].values).long()
    def __len__(self):
        return self.dataset.shape[0]
    def __getitem__(self, index):
        return (self.dataset[index], self.y[index])

# %%
train_set = Gene_expression_dataset(train = 'train')
test_set = Gene_expression_dataset(train = 'test')

# %%
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size= 64)

# %%
# Setting dimensions
latent_space_dim = 128
input_dim_expr = GE.shape[1]
level_2_dim_expr = 4096
level_3_dim_expr = 1024
level_4_dim = 512
classifier_1_dim = 128
classifier_2_dim = 64
class_num = len(cancer_type.keys())
classifier_out_dim = class_num

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # ENCODER fc layers
        # level 1
        # Expr input
        self.e_fc1_expr = self.fc_layer(input_dim_expr, level_2_dim_expr)

        # Level 2
        self.e_fc2_expr = self.fc_layer(level_2_dim_expr, level_3_dim_expr)
        # self.e_fc2_expr = self.fc_layer(level_2_dim_expr, level_3_dim_expr, dropout=True)

        # Level 3
        self.e_fc3 = self.fc_layer(level_3_dim_expr, level_4_dim)
        # self.e_fc3 = self.fc_layer(level_3_dim_expr, level_4_dim, dropout=True)

        # Level 4
        self.e_fc4_mean = self.fc_layer(level_4_dim, latent_space_dim, activation=0)
        self.e_fc4_log_var = self.fc_layer(level_4_dim, latent_space_dim, activation=0)

        # DECODER fc layers
        # Level 4
        self.d_fc4 = self.fc_layer(latent_space_dim, level_4_dim)

        # Level 3
        self.d_fc3 = self.fc_layer(level_4_dim, level_3_dim_expr)
        # self.d_fc3 = self.fc_layer(level_4_dim, level_3_dim_expr, dropout=True)

        # Level 2
        self.d_fc2_expr = self.fc_layer(level_3_dim_expr, level_2_dim_expr)
        # self.d_fc2_expr = self.fc_layer(level_3_dim_expr, level_2_dim_expr, dropout=True)

        # level 1
        # Expr output
        self.d_fc1_expr = self.fc_layer(level_2_dim_expr, input_dim_expr, activation=2)
        # CLASSIFIER fc layers
        self.c_fc1 = self.fc_layer(latent_space_dim, classifier_1_dim)
        self.c_fc2 = self.fc_layer(classifier_1_dim, classifier_2_dim)
        # self.c_fc2 = self.fc_layer(classifier_1_dim, classifier_2_dim, dropout=True)
        self.c_fc3 = self.fc_layer(classifier_2_dim, classifier_out_dim, activation=0)
    # Activation - 0: no activation, 1: ReLU, 2: Sigmoid
    def fc_layer(self, in_dim, out_dim, activation=1, dropout=False, dropout_p=0.5):
        if activation == 0:
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim))
        elif activation == 2:
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Sigmoid())
        else:
            if dropout:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p))
            else:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU())
        return layer

    def encode(self, x):
        expr_level2_layer = self.e_fc1_expr(x)

        level_3_layer = self.e_fc2_expr(expr_level2_layer)

        level_4_layer = self.e_fc3(level_3_layer)

        latent_mean = self.e_fc4_mean(level_4_layer)
        latent_log_var = self.e_fc4_log_var(level_4_layer)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mean + eps * sigma

    def decode(self, z):
        level_4_layer = self.d_fc4(z)

        level_3_layer = self.d_fc3(level_4_layer)

        expr_level2_layer = self.d_fc2_expr(level_3_layer)

        recon_x = self.d_fc1_expr(expr_level2_layer)

        return recon_x

    def classifier(self, mean):
        level_1_layer = self.c_fc1(mean)
        level_2_layer = self.c_fc2(level_1_layer)
        output_layer = self.c_fc3(level_2_layer)
        return output_layer

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        classifier_x = mean
        recon_x = self.decode(z)
        pred_y = self.classifier(classifier_x)
        return z, recon_x, mean, log_var, pred_y


# %%
device = torch.device("cuda:3")

# %%
def expr_recon_loss(recon_x, x):
    loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return loss

def kl_loss(mean, log_var):
    loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return loss

def classifier_loss_one_hot(pred_y, y):
    _, labels = y.max(dim=1)
    return nn.CrossEntropyLoss()(pred_y, labels)

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

# %%
def total_loss(recon_x, x, mean, log_var, pred_y, y, m):
    rec_loss = expr_recon_loss(recon_x,x)
    kl_l = kl_loss(mean, log_var)
    class_l = cross_entropy_one_hot(pred_y,y)
    return rec_loss + kl_l + m*class_l 

# %%
model = VAE().to(device)

# %%
# model.e_fc1_expr[0].weight

# %%
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)


# %%
def train(model, train_loader, optimizer, loss_op, scaler, device):
    model.train()
    total_loss = []
    for data in train_loader:
        GE, target  = data
        GE, target = GE.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast():
            z, recon_x, mean, log_var, output = model(GE)
            loss = loss_op(recon_x, GE, mean, log_var, output, target, 1.0)
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        total_loss.append(loss.item())
        optimizer.step()
    train_loss = np.average(total_loss)
    return np.sqrt(train_loss)

@torch.no_grad()
def test_class(model,loader,device):
    model.eval()
    correct = 0
    y_true, preds = [], []
    for data in tqdm(loader):
        GE, target  = data
        GE, target = GE.to(device), target.to(device)
        z, recon_x, mean, log_var, pred_type = model(GE)
        pred = pred_type.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == target.max(dim=1)[1]).sum())
    correct_portion = correct / len(loader.dataset)
    return correct_portion


# %%
criterion = total_loss
scaler = GradScaler()
for epoch in range(1, 301):
    train(model, train_loader, optimizer, criterion, scaler, device)
    if epoch % 50 == 0:
        train_acc = test_class(model,train_loader,device)
        test_acc = test_class(model,test_loader,device)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


