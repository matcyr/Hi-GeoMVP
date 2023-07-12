import numpy as np
import sys
sys.path.append ( '..' )
from prepare_data.create_drug_feat import get_atom_int_feature_dims,get_bond_feature_int_dims
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv,GraphSizeNorm,global_mean_pool,global_add_pool,global_max_pool

class atom_embedding_net(nn.Module):
    def __init__(self,model_config) -> None:
        super(atom_embedding_net,self).__init__()
        self.embed_dim = model_config.get('embed_dim')
        self.atom_embedding = nn.ModuleList()
        self.num_atom_feature = len(get_atom_int_feature_dims())
        for i in range(self.num_atom_feature):
            self.atom_embedding.append(nn.Embedding(get_atom_int_feature_dims()[i],self.embed_dim))
            torch.nn.init.xavier_uniform_(self.atom_embedding[i].weight.data)
    def forward(self,x):
        out = 0
        for i in range(self.num_atom_feature):
            out += self.atom_embedding[i](x[:,i].to(dtype=torch.int64))
        return out
    
class bond_embedding_net(nn.Module):
    def __init__(self,model_config) -> None:
        super(bond_embedding_net,self).__init__()
        self.embed_dim = model_config.get('embed_dim')
        self.bond_embedding = nn.ModuleList()
        self.num_bond_feature = len(get_bond_feature_int_dims())
        for i in range(self.num_bond_feature):
            self.bond_embedding.append(nn.Embedding(get_bond_feature_int_dims()[i] + 3,self.embed_dim))
            torch.nn.init.xavier_uniform_(self.bond_embedding[i].weight.data)
    def forward(self,x):
        out = 0
        for i in range(self.num_bond_feature):
            out += self.bond_embedding[i](x[:,i].to(dtype=torch.int64))
        return out


class RBF(nn.Module):
    """
    Radial Basis Function
    """
    def __init__(self, centers, gamma, dtype=torch.float32):
        super(RBF, self).__init__()
        self.centers = torch.reshape(torch.tensor(centers, dtype=dtype), [1, -1])
        self.gamma = gamma
    
    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = torch.reshape(x, [-1, 1])
        device = x.get_device()
        if device == -1:
            return torch.exp(-self.gamma * torch.square(x - self.centers))
        else:
            return torch.exp(-self.gamma * torch.square(x - self.centers.to(device)))
class FloatEmbedding(nn.Module):
    """
    Atom Float Encoder
    """
    def __init__(self, embed_dim, rbf_params=None):
        super(FloatEmbedding, self).__init__()        
        self.rbf_params = rbf_params
        centers, gamma = self.rbf_params
        self.rbf = RBF(centers, gamma)
        self.linear = nn.Linear(len(centers), embed_dim)


    def forward(self, feats):
        """
        Args: 
            feats(dict of tensor): node float features.
        """
        out_embed = self.rbf(feats)
        out_embed = self.linear(out_embed)
        return out_embed
class GIN_conv(nn.Module):  
    def __init__(self, embed_dim, dropout_rate, last_act = False):
        super(GIN_conv, self).__init__()

        self.embed_dim = embed_dim
        self.last_act = last_act
        self.mlp = nn.Sequential(
                        nn.Linear(self.embed_dim, self.embed_dim * 2),
                        nn.ReLU(),
                        nn.Linear(self.embed_dim * 2, self.embed_dim))
        self.gnn = GINEConv(nn = self.mlp,edge_dim=self.embed_dim)
        self.norm_layer = nn.LayerNorm(self.embed_dim)
        self.graph_norm = GraphSizeNorm()
        if last_act:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, node_hidden, edge_hidden,edge_index,batch):
        """tbd"""      
        out = self.gnn(x=node_hidden,edge_attr = edge_hidden,edge_index = edge_index)
        ### Layer normalization
        out = self.norm_layer(out)
        ### Graph size normalization
        # out = self.graph_norm(out,batch = batch)
        if self.last_act:
            out = self.act(out)
        out = self.dropout(out)
        ### Residual connection
        out = out + node_hidden
        return out

class GeoGNNModel(nn.Module):
    """
    The GeoGNN Model used in GEM.
    Args:
        model_config(dict): a dict of model configurations.
    """
    def __init__(self, model_config={}):
        super(GeoGNNModel, self).__init__()

        self.embed_dim = model_config.get('embed_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.layer_num = model_config.get('layer_num')
        self.readout = model_config.get('readout')

        self.atom_rbf_para = (np.arange(0, 2, 0.1), 10.0)   # (centers, gamma)
        self.bond_rbf_para = (np.arange(0, 2, 0.1), 10.0)
        self.angle_rbf_para = (np.arange(0, np.pi, 0.1), 10.0)
        self.atom_rbf_emb_nn = FloatEmbedding(self.embed_dim,self.atom_rbf_para)
        self.bond_rbf_emb_nn = FloatEmbedding(self.embed_dim,self.bond_rbf_para)
        self.angle_rbf_emb_nn = FloatEmbedding(self.embed_dim,self.angle_rbf_para)    

        self.atom_int_embed_nn = nn.ModuleList([torch.nn.Embedding(dim, self.embed_dim) for dim in get_atom_int_feature_dims()])
        for item in self.atom_int_embed_nn:
            torch.nn.init.xavier_uniform_(item.weight.data)
        self.bond_int_embed_nn = nn.ModuleList([torch.nn.Embedding(dim+3, self.embed_dim) for dim in get_bond_feature_int_dims()])
        for item in self.bond_int_embed_nn:
            torch.nn.init.xavier_uniform_(item.weight.data)
        
        self.bond_embedding_list = nn.ModuleDict()
        self.bond_float_rbf_list = nn.ModuleList()
        self.bond_angle_float_rbf_list = nn.ModuleList()
        self.atom_bond_block_list = nn.ModuleList()
        self.bond_angle_block_list = nn.ModuleList()
        for layer_id in range(self.layer_num):
            layer = str(layer_id)
            bond_emb = nn.ModuleList([torch.nn.Embedding(dim+3, self.embed_dim) for dim in get_bond_feature_int_dims()])
            for item in bond_emb:
                torch.nn.init.xavier_uniform_(item.weight.data)
            self.bond_embedding_list[layer] = bond_emb ### bond int feature embedding dictionary
            self.bond_float_rbf_list.append(
                    FloatEmbedding(self.embed_dim,self.bond_rbf_para))
            self.bond_angle_float_rbf_list.append(
                    FloatEmbedding(self.embed_dim,self.angle_rbf_para))
            self.atom_bond_block_list.append(
                    GIN_conv(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
            self.bond_angle_block_list.append(
                    GIN_conv(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
        if self.readout == 'mean':
            self.graph_pool = global_mean_pool
        else:
            self.graph_pool = global_add_pool

        print('[GeoGNNModel] embed_dim:%s' % self.embed_dim)
        print('[GeoGNNModel] dropout_rate:%s' % self.dropout_rate)
        print('[GeoGNNModel] layer_num:%s' % self.layer_num)
        print('[GeoGNNModel] readout:%s' % self.readout)

    @property
    def node_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def forward(self, mol):
        """
        Build the network.
        """       
        #### Embedding for initial feature
        ### Embedding for atom feature. 9 int + 1 float
        node_hidden = 0
        for i in range(len(self.atom_int_embed_nn)):
            node_hidden += self.atom_int_embed_nn[i](mol.x_atom[:,i].to(dtype=torch.int64))
        node_hidden += self.atom_rbf_emb_nn(mol.x_atom[:,-1])    
        ### Embedding for bond feature. 3 int + 1 float
        bond_hidden = 0 
        for i in range(len(self.bond_int_embed_nn)):
            bond_hidden += self.bond_int_embed_nn[i](mol.edge_attr_atom[:,i].to(dtype=torch.int64))
        edge_hidden = bond_hidden + self.bond_rbf_emb_nn(mol.edge_attr_atom[:,-1])        

        node_hidden_list = [node_hidden]
        edge_hidden_list = [edge_hidden]
        
        for layer_id in range(self.layer_num):
            layer = str(layer_id)
            node_hidden = self.atom_bond_block_list[layer_id](node_hidden = node_hidden_list[layer_id],
                                                              edge_hidden = edge_hidden_list[layer_id],
                                                              edge_index = mol.edge_index_atom, 
                                                              batch = mol.x_atom_batch)   
            cur_edge_hidden = 0        
            bond_int_embed_nn = self.bond_embedding_list[layer]         
            for i in range(len(bond_int_embed_nn)):
                cur_edge_hidden += bond_int_embed_nn[i](mol.x_bond[:,i].to(dtype=torch.int64))        
            cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](mol.x_bond[:,-1])
            cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](mol.edge_attr_bond)
            edge_hidden = self.bond_angle_block_list[layer_id](node_hidden = cur_edge_hidden,
                                                              edge_hidden = cur_angle_hidden,
                                                              edge_index = mol.edge_index_bond, 
                                                              batch = mol.x_bond_batch)
            node_hidden_list.append(node_hidden)
            edge_hidden_list.append(edge_hidden)
        
        node_repr = node_hidden_list[-1]
        edge_repr = edge_hidden_list[-1]
        graph_repr = self.graph_pool(node_repr,batch = mol.x_atom_batch)
        return node_repr, edge_repr, graph_repr
    
class drug_classfication(nn.Module):
    def __init__(self, model_config,graph_encoder):
        super(drug_classfication, self).__init__()   
        self.graph_encoder = graph_encoder       
        self.embed_dim = model_config.get('embed_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.layer_num = model_config.get('layer_num')
        self.readout = model_config.get('readout')
        self.norm = nn.LayerNorm(graph_encoder.graph_dim)
        self.lin1 = nn.Linear(self.embed_dim,self.embed_dim)
        self.lin2 = nn.Linear(self.embed_dim,2)
    def forward(self,graph):
        node_repr,edge_repr,graph_repr = self.graph_encoder(graph)
        graph_repr = self.norm(graph_repr)
        h = self.lin1(graph_repr)
        h = h.relu()
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = self.lin2(h)
        # h = nn.Sigmoid()(h)
        return h 


class drug_conv_1d(torch.nn.Module):
    def __init__(self, in_channels= 30, out_channels=[40, 80, 60]):
        super(drug_conv_1d, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.conv_smiles_1 = nn.Conv1d(
            in_channels=self.in_channels, out_channels=self.out_channels[0], kernel_size=7, stride=1, padding='same')
        self.smiles_bn1 = nn.BatchNorm1d(self.out_channels[0])
        self.smiles_pool1 = nn.MaxPool1d(3)
        self.smiles_act1 = nn.ReLU()
        self.smiles_conv_block_1 = nn.Sequential(
            self.conv_smiles_1, self.smiles_bn1, self.smiles_act1, self.smiles_pool1)
        self.conv_smiles_2 = nn.Conv1d(
            in_channels=out_channels[0], out_channels=self.out_channels[1], kernel_size=7, stride=1, padding='same')
        self.smiles_bn2 = nn.BatchNorm1d(out_channels[1])
        self.smiles_pool2 = nn.MaxPool1d(3)
        self.smiles_act2 = nn.ReLU()
        self.smiles_conv_block_2 = nn.Sequential(
            self.conv_smiles_2, self.smiles_bn2, self.smiles_act2, self.smiles_pool2)
        self.conv_smiles_3 = nn.Conv1d(
            in_channels=out_channels[1], out_channels=self.out_channels[2], kernel_size=7, stride=1, padding='same')
        self.smiles_bn3 = nn.BatchNorm1d(out_channels[2])
        self.smiles_pool3 = nn.MaxPool1d(3)
        self.smiles_act3 = nn.ReLU()
        self.smiles_conv_block_3 = nn.Sequential(
            self.conv_smiles_3, self.smiles_bn3, self.smiles_act3, self.smiles_pool3)

    def forward(self, smiles):
        out = self.smiles_conv_block_1(smiles)
        out = self.smiles_conv_block_2(out)
        out = self.smiles_conv_block_3(out)
        out = out.view(-1, out.shape[1]*out.shape[2])
        return out




# class drug_fp_conv_1d(torch.nn.Module):
#     def __init__(self, n_filters=4):
#         super(drug_fp_conv_1d, self).__init__()
#         self.conv_fp_1 = nn.Conv1d(
#             in_channels=1, out_channels=n_filters, kernel_size=8, bias = False)
#         self.fp_bn1 = nn.BatchNorm1d(n_filters)
#         self.fp_pool1 = nn.MaxPool1d(3)
#         self.fp_act1 = nn.ReLU()
#         self.fp_conv_block_1 = nn.Sequential(
#             self.conv_fp_1, self.fp_bn1, self.fp_act1, self.fp_pool1)
#         self.conv_fp_2 = nn.Conv1d(
#             in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8, bias = False)
#         self.fp_bn2 = nn.BatchNorm1d(n_filters * 2)
#         self.fp_pool2 = nn.MaxPool1d(3)
#         self.fp_act2 = nn.ReLU()
#         self.fp_conv_block_2 = nn.Sequential(
#             self.conv_fp_2, self.fp_bn2, self.fp_act2, self.fp_pool2)
#         self.conv_fp_3 = nn.Conv1d(
#             in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8, bias = False)
#         self.fp_bn3 = nn.BatchNorm1d(n_filters * 4)
#         self.fp_pool3 = nn.MaxPool1d(3)
#         self.fp_act3 = nn.ReLU()
#         self.fp_conv_block_3 = nn.Sequential(
#             self.conv_fp_3, self.fp_bn3, self.fp_act3, self.fp_pool3)

#     def forward(self, smiles):
#         out = self.fp_conv_block_1(smiles)
#         out = self.fp_conv_block_2(out)
#         out = self.fp_conv_block_3(out)
#         out = out.view(-1, out.shape[1]*out.shape[2])
#         return out


class drug_1d_embedding(nn.Module):
    def __init__(self, smiles_in_dim, fp_in_dim, embed_dim) -> None:
        super(drug_1d_embedding, self).__init__()
        self.ln_smiles_1 = nn.Linear(smiles_in_dim, 512)
        self.bn_smiles = nn.BatchNorm1d(512)
        self.act_smiles1 = nn.ReLU()
        self.ln_smiles2 = nn.Linear(512, embed_dim)
        self.bn_smiles2 = nn.BatchNorm1d(embed_dim)
        self.embed_smiles = nn.Sequential(
            self.ln_smiles_1, self.bn_smiles, self.act_smiles1, self.ln_smiles2, self.bn_smiles2)
        self.ln_fp_1 = nn.Linear(fp_in_dim, 512)
        self.bn_fp = nn.BatchNorm1d(512)
        self.act_fp1 = nn.ReLU()
        self.ln_fp2 = nn.Linear(512, embed_dim)
        self.bn_fp2 = nn.BatchNorm1d(embed_dim)
        self.embed_fp = nn.Sequential(
            self.ln_fp_1, self.bn_fp, self.act_fp1, self.ln_fp2, self.bn_fp2)

    def forward(self, smiles, fp):
        smiles_out = self.embed_smiles(smiles)
        fp_out = self.embed_fp(fp)
        out = torch.cat((smiles_out, fp_out), 1)
        return out

class drug_2d_encoder(nn.Module):
    def __init__(self,model_config = {}) -> None:
        super(drug_2d_encoder,self).__init__()
        self.embed_dim = model_config.get('embed_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.layer_num = model_config.get('layer_num')
        self.readout = model_config.get('readout')
        self.atom_int_embed_nn = torch.nn.Embedding(get_atom_int_feature_dims()[0], self.embed_dim)
        torch.nn.init.xavier_uniform_(self.atom_int_embed_nn.weight.data)
        self.bond_int_embed_nn = torch.nn.Embedding(get_bond_feature_int_dims()[0] + 3, self.embed_dim)
        torch.nn.init.xavier_uniform_(self.bond_int_embed_nn.weight.data)
        self.atom_bond_block_list = nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.layer_num):
            self.batch_norms.append(torch.nn.BatchNorm1d(self.embed_dim))
        for layer_id in range(self.layer_num):
            self.atom_bond_block_list.append(
                    GIN_conv(self.embed_dim, self.dropout_rate))
        if self.readout == 'mean':
            self.graph_pool = global_mean_pool
        elif self.readout == 'add':
            self.graph_pool = global_add_pool
        else :self.graph_pool = global_max_pool
    def forward(self,drug_atom):
        ## embed only for atom type
        x,edge_index,edge_attr,batch = drug_atom.x, drug_atom.edge_index, drug_atom.edge_attr, drug_atom.batch
        h = self.atom_int_embed_nn(x[:,0].to(dtype=torch.int64))
        edge_attr = self.bond_int_embed_nn(edge_attr[:,0].to(dtype=torch.int64))
        for layer_id in range(self.layer_num):
            h = self.atom_bond_block_list[layer_id](h, edge_attr,edge_index,batch = batch)
            h = self.batch_norms[layer_id](h)
            if layer_id == self.layer_num - 1:
                h = F.dropout(h, self.dropout_rate, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout_rate, training = self.training)
        return self.graph_pool(h,batch)