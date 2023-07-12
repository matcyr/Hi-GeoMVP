import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FAConv,GCNConv,global_mean_pool,global_max_pool

class Cell_encoder_gene(torch.nn.Module):
    def __init__(self,model_config) -> None:
        super().__init__()
        self.hidden_dim = model_config.get('hidden_dim')
        self.conv1 = FAConv(channels=self.hidden_dim)
        self.conv2 = FAConv(channels=self.hidden_dim)
        self.conv3 = FAConv(channels=self.hidden_dim)
        self.lin1  = nn.Linear(1,self.hidden_dim)
        self.graph_pool = global_mean_pool
    def forward(self, x, edge_index,batch):
        # x, edge_index , batch = cell.cell_x, cell.cell_edge_index , cell.cell_x_batch
        x = self.lin1(x)
        x1 = self.conv1(x,x, edge_index)
        x2 = self.conv2(x1,x, edge_index)
        x3 = self.conv3(x2,x, edge_index)
        out = self.graph_pool(x3, batch)
        return out
class Cell_encoder_mut(torch.nn.Module):
    def __init__(self,model_config) -> None:
        super().__init__()
        self.hidden_dim = model_config.get('hidden_dim')
        self.conv1 = FAConv(channels=self.hidden_dim)
        self.conv2 = FAConv(channels=self.hidden_dim)
        self.conv3 = FAConv(channels=self.hidden_dim)
        self.embed1  = nn.Embedding(2,self.hidden_dim)
        self.graph_pool = global_mean_pool
    def forward(self, x, edge_index,batch):
        # x, edge_index , batch = cell.cell_x, cell.cell_edge_index , cell.cell_x_batch
        x = self.embed1(x)
        x1 = self.conv1(x,x, edge_index)
        x2 = self.conv2(x1,x, edge_index)
        x3 = self.conv3(x2,x, edge_index)
        out = self.graph_pool(x3, batch)
        return out
    
    
class Cell_encoder_cnv(torch.nn.Module):
    def __init__(self,model_config) -> None:
        super().__init__()
        self.hidden_dim = model_config.get('hidden_dim')
        self.conv1 = FAConv(channels=self.hidden_dim)
        self.conv2 = FAConv(channels=self.hidden_dim)
        self.conv3 = FAConv(channels=self.hidden_dim)
        self.embed1  = nn.Embedding(2,self.hidden_dim)
        self.graph_pool = global_mean_pool
    def forward(self, x, edge_index,batch):
        # x, edge_index , batch = cell.cell_x, cell.cell_edge_index , cell.cell_x_batch
        x = self.embed1(x)
        x1 = self.conv1(x,x, edge_index)
        x2 = self.conv2(x1,x, edge_index)
        x3 = self.conv3(x2,x, edge_index)
        out = self.graph_pool(x3, batch)
        return out    
    
class Sim_GNN(nn.Module): ## Implement for gene_expression.
    def __init__(self, model_config):
        super(Sim_GNN, self).__init__()
        self.num_layer = 3  ## Modify later.
        self.hidden_features = model_config.get('hidden_dim')
        self.lin1  = nn.Linear(1,self.hidden_features)
        self.GNN_layers = nn.ModuleList()
        self.GNN_sim_layers = nn.ModuleList()
        self.Weight_layers = nn.ModuleList()
        self.GNN_layers.append(GCNConv(self.hidden_features, self.hidden_features))
        self.GNN_sim_layers.append(GCNConv(self.hidden_features, self.hidden_features))
        self.Weight_layers.append(nn.Linear(self.hidden_features, 1)) ## Learnable weight
        for i in range(self.num_layer-1):
            self.GNN_layers.append(GCNConv(self.hidden_features, self.hidden_features))
            self.GNN_sim_layers.append(GCNConv(self.hidden_features, self.hidden_features))
            self.Weight_layers.append(nn.Linear(self.hidden_features, 1)) ## Learnable weight
        # self.class_mut = nn.Linear(hidden_features, 2) ## Mutation classifier
        # self.class_cnv = nn.Linear(hidden_features, 2) ## CNV classifier
        self.pooling =  global_max_pool ## Graph pooling

    def forward(self, x, edge_index, edge_index_sim, batch):
        hidden = self.lin1(x)
        for i in range(self.num_layer):
            x_gcn = F.relu(self.GNN_layers[i](hidden, edge_index))
            x_sim_gcn = F.relu(self.GNN_sim_layers[i](hidden, edge_index_sim))
            s = torch.sigmoid(self.Weight_layers[i](hidden))
            hidden = s * x_gcn + (1-s) * x_sim_gcn
        # mut_logits = self.fc2(hidden)
        # cnv_logits = self.fc3(hidden)
        graph_repr = self.pooling(hidden,batch)
        # if y is not None:
        #     node_loss = F.cross_entropy(node_logits[y > 0], y[y > 0].long())
        #     return node_loss, graph_repr
        return graph_repr





# class Cell_CNN(nn.Module):
# ## Implement for convolution layer for cell line.
#     def __init__(self, model_config, x_feat_in_dim):
#         super(Cell_CNN, self).__init__()
#         n_filters=32
#         self.embed_dim = model_config.get('embed_dim')
#         self.hidden_dim = model_config.get('hidden_dim')
#         self.embed_layer = nn.Embedding(2,self.hidden_dim)
#         self.conv1 = nn.Conv1d(1, n_filters, kernel_size=8)
#         self.bn1 = nn.BatchNorm1d(n_filters)
#         self.pool_layer_1 = nn.MaxPool1d(3)
#         self.embed_layer_1 = nn.Sequential(self.conv1, self.bn1, nn.ReLU(),self.pool_layer_1)
#         self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size=8)
#         self.pool_layer_2 = nn.MaxPool1d(3)
#         self.bn2 = nn.BatchNorm1d(n_filters*2)
#         self.embed_layer_2 = nn.Sequential(self.conv2, self.bn2, nn.ReLU(),self.pool_layer_2)
#         self.conv3 = nn.Conv1d(n_filters*2, n_filters*4, kernel_size=8)
#         self.bn3 = nn.BatchNorm1d(n_filters*4)
#         self.pool_layer_3 = nn.MaxPool1d(3)
#         self.embed_layer_3 = nn.Sequential(self.conv3, self.bn3, nn.ReLU(),self.pool_layer_3)
#         self.ln_1 = nn.Linear(x_feat_in_dim,512)
#         self.bn = nn.BatchNorm1d(512)
#         self.act_1 = nn.ReLU()
#         self.ln_2 = nn.Linear(512,self.embed_dim)
#         self.bn_2 = nn.BatchNorm1d(self.embed_dim)        
#         self.embed_cell = nn.Sequential(self.ln_1,self.bn,self.act_1,self.ln_2,self.bn_2)
#     def forward(self, x_feat):
#         conv_x_feat = self.embed_layer_1(x_feat)
#         conv_x_feat = self.embed_layer_2(conv_x_feat)
#         conv_x_feat = self.embed_layer_3(conv_x_feat)   
#         out = conv_x_feat.view(-1,conv_x_feat.shape[1]*conv_x_feat.shape[2])
#         conv_out = self.embed_cell(out)
#         return conv_out




class cell_cnn(nn.Module):
    def __init__(self, model_config) -> None:
        super().__init__()
        n_filters = 8 
        self.hidden_dim = model_config.get('hidden_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.cell_type = model_config.get('cell_type') ## mut, cnv, ge
        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)
        if self.cell_type == mut:
            self.fc1_xt = nn.Linear(2560, self.hidden_dim)
        elif self.cell_type == cnv:
            self.fc1_xt = nn.Linear(2560, self.hidden_dim) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
    def forward(self,mut):
        conv_xt = self.conv_xt_1(mut)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)
        return xt





   
class Cell_multi_omics_Encoder(nn.Module):
    def __init__(self, model_config,use_cnn = False):
        super(Cell_multi_omics_Encoder, self).__init__()
        self.use_cnn = use_cnn
        self.embed_dim = model_config.get('embed_dim')
        self.hidden_dim = model_config.get('hidden_dim')
        if self.use_cnn:
            self.mut_cnn = Cell_CNN(model_config, 2560)
            self.cnv_cnn = Cell_CNN(model_config, 2816)
        self.ge_gnn = Sim_GNN(model_config)
        self.mut_gnn = Cell_encoder_mut(model_config)
        self.cnv_gnn = Cell_encoder_cnv(model_config)
    def forward(self,ge, ge_sim, cnv, mut):
        batch_size = int(mut.x.shape[0]/636)
        if self.use_cnn:
            mut_repr = self.mut_cnn(mut.x.view(batch_size,1,636)) + self.mut_gnn(mut.x.view(-1).long(),mut.edge_index,mut.batch)
            cnv_repr = self.cnv_cnn(cnv.x.view(batch_size,1,694)) + self.cnv_gnn(cnv.x.view(-1).long(),cnv.edge_index,cnv.batch)
        else:
            mut_repr = self.mut_gnn(mut.x.view(-1).long(),mut.edge_index,mut.batch)
            cnv_repr = self.cnv_gnn(cnv.x.view(-1).long(),cnv.edge_index,cnv.batch)  
        ge_repr = self.ge_gnn(ge.x,ge.edge_index,ge_sim.edge_index,ge.batch)
        return mut_repr,cnv_repr,ge_repr   