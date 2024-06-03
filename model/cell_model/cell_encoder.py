import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FAConv,GCNConv,global_mean_pool,global_max_pool, GATConv, max_pool, GraphNorm, GraphMultisetTransformer


class cell_cnn(nn.Module):
    def __init__(self, model_config, out_dim) -> None:
        super().__init__()
        n_filters = 32 
        self.hidden_dim = model_config.get('hidden_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8, bias = False)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8, bias = False)
        self.bn2 = nn.BatchNorm1d(n_filters*2)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8, bias = False)
        self.bn3 = nn.BatchNorm1d(n_filters*4)
        self.pool_xt_3 = nn.MaxPool1d(3)
        self.fc1_xt = nn.Linear(out_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
    def forward(self,mut):
        conv_xt = self.bn1(self.conv_xt_1(mut.float()))
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.bn2(self.conv_xt_2(conv_xt))
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.bn3(self.conv_xt_3(conv_xt))
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)
        return xt

    
class GE_vae(nn.Module):
    def __init__(self, model_config):
        super(GE_vae, self).__init__()
        level_2_dim_expr = 4096
        level_3_dim_expr = 1024
        level_4_dim = 512
        self.hidden_dim = model_config.get('hidden_dim')
        self.e_fc1_expr = self.fc_layer(8046, level_2_dim_expr)
        self.e_fc2_expr = self.fc_layer(level_2_dim_expr, level_3_dim_expr, dropout=True)
        self.e_fc3 = self.fc_layer(level_3_dim_expr, level_4_dim, dropout=True)
        self.e_fc4_mean = self.fc_layer(level_4_dim, self.hidden_dim, activation=0)
        self.e_fc4_log_var = self.fc_layer(level_4_dim, self.hidden_dim, activation=0)

        # DECODER fc layers
        self.d_fc4 = self.fc_layer(self.hidden_dim, level_4_dim)
        self.d_fc3 = self.fc_layer(level_4_dim, level_3_dim_expr, dropout=True)
        self.d_fc2_expr = self.fc_layer(level_3_dim_expr, level_2_dim_expr, dropout=True)
        self.d_fc1_expr = self.fc_layer(level_2_dim_expr, 8046, activation=2)
    # Activation - 0: no activation, 1: ReLU, 2: Sigmoid
    def fc_layer(self, in_dim, out_dim, activation=1, dropout=True, dropout_p=0.5):
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
    
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class GNN_cell_view(torch.nn.Module):
    def __init__(self, layer_cell, dim_cell, cluster_predefine, omics_type, dropout_ratio = 0.4):
        super().__init__()
        dropout_ratio = 0.4
        self.dropout_rate = dropout_ratio
        self.num_feature = 2     
        self.omics_type = omics_type
        self.layer_cell = layer_cell
        self.dim_cell = dim_cell
        self.embed1  = nn.Embedding(self.num_feature , self.dim_cell)
        self.ln1 = nn.Linear(1,self.dim_cell)
        self.cluster_predefine = cluster_predefine
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        self.layer_norm_cell = torch.nn.ModuleList()
        self.graph_norm_cell = torch.nn.ModuleList()        
        self.cell_embed_layer = nn.Sequential(
                    nn.Linear(dim_cell * self.final_node, 1024),
                    nn.ReLU(),
                    nn.Dropout(p= dropout_ratio),
                    nn.Linear(1024, 256),
                    nn.ReLU(),
                    nn.Dropout(p= dropout_ratio),
        )

        for i in range(self.layer_cell):
            conv = GATConv(self.dim_cell, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False) 
            self.layer_norm_cell.append(nn.LayerNorm(self.dim_cell))
            self.graph_norm_cell.append(GraphNorm(self.dim_cell))
            self.convs_cell.append(conv)
            self.bns_cell.append(bn)
    def forward(self, cell):
        num_graphs = cell.num_graphs
        if self.omics_type == 'ge':
            cell.x = self.ln1(cell.x)
        else: cell.x = self.embed1(cell.x)
        hidden = [cell.x]
        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            num_node = int(cell.x.size(0) / cell.num_graphs)
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(num_graphs)])
            cell = max_pool(cluster, cell, transform=None)
            cell.x = self.layer_norm_cell[i](cell.x)
            cell.x = self.graph_norm_cell[i](cell.x)
        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell)
        cell_embed = self.cell_embed_layer(node_representation)

        return cell_embed

      

class GNN_cell_view_predifine(torch.nn.Module):
    def __init__(self, layer_cell, dim_cell, cluster_predefine, omics_type, dropout_ratio = 0.4):
        super().__init__()
        self.num_feature = 2     
        self.omics_type = omics_type
        self.layer_cell = layer_cell
        self.dim_cell = dim_cell
        self.embed1  = nn.Embedding(self.num_feature , self.dim_cell)
        self.ln1 = nn.Linear(1,self.dim_cell)
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        self.layer_norm_cell = torch.nn.ModuleList()
        self.graph_norm_cell = torch.nn.ModuleList()        
        self.cell_embed_layer = nn.Sequential(
                    nn.Linear(dim_cell * self.layer_cell, 1024),
                    nn.ReLU(),
                    nn.Dropout(p= dropout_ratio),
                    nn.Linear(1024, 256),
                    nn.ReLU(),
                    nn.Dropout(p= dropout_ratio),
        )
        self.pool = GraphMultisetTransformer(dim_cell * self.layer_cell, k=30, heads=4)
        for i in range(self.layer_cell):
            conv = GATConv(self.dim_cell, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False) 
            self.layer_norm_cell.append(nn.LayerNorm(self.dim_cell))
            self.graph_norm_cell.append(GraphNorm(self.dim_cell))
            self.convs_cell.append(conv)
            self.bns_cell.append(bn)

    def forward(self, cell):
        num_graphs = cell.num_graphs
        if self.omics_type == 'ge':
            cell.x = self.ln1(cell.x)
        else: cell.x = self.embed1(cell.x)
        hidden = []
        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            cell.x = self.layer_norm_cell[i](cell.x)
            cell.x = self.graph_norm_cell[i](cell.x)
            hidden.append(cell.x)
        x = torch.cat(hidden, dim=-1)
        node_representation = self.pool(x, cell.batch)
        cell_embed = self.cell_embed_layer(node_representation)

        return cell_embed