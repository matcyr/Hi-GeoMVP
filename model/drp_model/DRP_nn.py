import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')
from model.cell_model.cell_encoder import *
from model.drug_model.drug_encoder import *
from torch_geometric.nn import GINConv
import math

# def scaled_dot_product(q, k, v, mask=None):
#     d_k = q.size()[-1]
#     attn_logits = torch.matmul(q, k.transpose(-2, -1))
#     attn_logits = attn_logits / math.sqrt(d_k)
#     if mask is not None:
#         attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
#     attention = F.softmax(attn_logits, dim=-1)
#     values = torch.matmul(attention, v)
#     return values, attention

# class AttentionBlock(nn.Module):
#     def __init__(self,model_config) -> None:
#         super(AttentionBlock,self).__init__()
#         self.embed_dim = model_config.get('embed_dim')
#         self.hidden_dim = model_config.get('hidden_dim')
#         self.dropout_rate = model_config.get('dropout_rate')
#         self.use_cnn = model_config.get('use_cnn')
#         self.use_smiles = model_config.get('use_smiles')
#         self.use_fp = model_config.get('use_fp')
#         self.atten_dim = model_config.get('atten_dim')
#         self.drug_dim = self.embed_dim + self.use_fp * self.embed_dim + self.use_smiles * self.embed_dim
#         self.cell_dim = model_config.get('hidden_dim')
#         self.Q_drug_proj = nn.Linear(self.drug_dim, self.atten_dim)
#         self.K_omics_proj = nn.Linear(self.cell_dim, self.atten_dim)
#         self.V_ic50_proj = nn.Linear(self.cell_dim+self.drug_dim, self.atten_dim)
#     def forward(self,x_omics,x_drug):
#         Q_drug = self.Q_drug_proj(x_drug)
#         K_omics = self.K_omics_proj(x_omics)
#         V_ic50 = self.V_ic50_proj(torch.cat([x_drug,x_omics],-1))    
#         pred, attention = scaled_dot_product(Q_drug, K_omics, V_ic50)
#         return pred, attention
    
# class Multi_Omics_Regression(nn.Module):
#     def __init__(self, model_config):
#         super(Multi_Omics_Regression, self).__init__()
#         self.hidden_dim = model_config.get('hidden_dim')
#         self.embed_dim = model_config.get('embed_dim')
#         self.dropout_rate = model_config.get('dropout_rate')
#         self.atten_dim = model_config.get('atten_dim')
#         self.use_cnn = model_config.get('use_cnn')
#         self.use_smiles = model_config.get('use_smiles')
#         self.use_fp = model_config.get('use_fp')
#         self.gene_attn = AttentionBlock(model_config)
#         self.mut_attn = AttentionBlock(model_config)
#         self.cnv_attn = AttentionBlock(model_config)
#         self.regression = nn.Sequential(
#             nn.Linear(3*self.atten_dim, self.atten_dim),
#             nn.ELU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.atten_dim, self.atten_dim),
#             nn.ELU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.atten_dim, 1)
#         )
#         self.drug_encoder = drug_encoder(model_config)
#         self.cell_encoder = Cell_multi_omics_Encoder(model_config)
#     def forward(self, drug_atom, drug_bond, ge, ge_sim, cnv, mut):
#         drug_feat = self.drug_encoder(drug_atom,drug_bond)
#         cnv_feat,mut_feat,gene_feat = self.cell_encoder(ge, ge_sim, cnv, mut)
#         attended_gene_feat,attention_weights_gene = self.gene_attn(gene_feat, drug_feat)
#         attended_mut_feat, attention_weights_mut = self.mut_attn(mut_feat, drug_feat)
#         attended_cnv_feat, attention_weights_cnv = self.cnv_attn(cnv_feat, drug_feat)
#         attended_feat = torch.cat([attended_gene_feat, attended_mut_feat, attended_cnv_feat], dim=-1)
#         attend_list = [attention_weights_gene, attention_weights_mut, attention_weights_cnv]
#         x = self.regression(attended_feat)
#         return x

# class Single_Omics_Regression(nn.Module):
#     def __init__(self, model_config):
#         super(Single_Omics_Regression, self).__init__()
#         self.hidden_dim = model_config.get('hidden_dim')
#         self.embed_dim = model_config.get('embed_dim')
#         self.dropout_rate = model_config.get('dropout_rate')
#         self.atten_dim = model_config.get('atten_dim')
#         self.use_cnn = model_config.get('use_cnn')
#         self.use_smiles = model_config.get('use_smiles')
#         self.use_fp = model_config.get('use_fp')
#         self.gene_attn = AttentionBlock(model_config)
#         self.regression = nn.Sequential(
#             nn.Linear(self.atten_dim, self.atten_dim),
#             nn.ELU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.atten_dim, self.atten_dim),
#             nn.ELU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.atten_dim, 1)
#         )
#         self.drug_encoder = Drug_3d_Encoder(model_config)
#         self.cell_encoder = Cell_encoder_gene(model_config)
#     def forward(self, drug_atom, drug_bond, ge):
#         drug_feat = self.drug_encoder(drug_atom,drug_bond)
#         gene_feat = self.cell_encoder(ge.x, ge.edge_index, ge.batch)
#         attended_gene_feat,attention_weights_gene = self.gene_attn(gene_feat, drug_feat)
#         x = self.regression(attended_gene_feat)
#         return x

class DRP(nn.Module):
    def __init__(self, model_config) -> None:
        super().__init__()
        self.embed_dim = model_config.get('embed_dim')
        self._hidden_dim = model_config.get('hidden_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.drug_encoder = Drug_3d_Encoder(model_config)
        self.cell_encoder = Cell_encoder_gene(model_config)
        self.fc1_drug = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc1_cell = nn.Linear(self._hidden_dim, self.embed_dim)
        self.regression = nn.Sequential(
            nn.Linear(2*self.embed_dim, 1024),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, drug_atom,drug_bond,ge):
        drug_embed = self.drug_encoder(drug_atom,drug_bond)
        drug_embed = nn.ReLU()(self.fc1_drug(drug_embed))
        cell_embed = self.cell_encoder(ge.x,ge.edge_index,ge.batch)
        cell_embed = nn.ReLU()(self.fc1_cell(cell_embed))
        x = torch.cat([drug_embed, cell_embed], -1)
        x = self.regression(x)
        x = nn.Sigmoid()(x)
        return x
    
class MultiOmicsDRP(nn.Module):
    def __init__(self, model_config) -> None:
        super().__init__()
        self.embed_dim = model_config.get('embed_dim')
        self.hidden_dim = model_config.get('hidden_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.drug_encoder = Drug_3d_Encoder(model_config)
        self.cell_encoder_ge = Cell_encoder_gene(model_config)
        self.cell_encoder_mut = Cell_encoder_mut(model_config)
        self.cell_encoder_cnv = Cell_encoder_cnv(model_config)
        self.regression = nn.Sequential(
            nn.Linear(self.hidden_dim*3 + self.embed_dim, self.embed_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, drug_atom, drug_bond, ge, cell_cnv, cell_mut):
        drug_embed = self.drug_encoder(drug_atom,drug_bond)
        ge_embed = self.cell_encoder_ge(ge.x,ge.edge_index,ge.batch)
        mut_embed = self.cell_encoder_mut(cell_mut.x,cell_mut.edge_index,cell_mut.batch)
        cnv_embed = self.cell_encoder_cnv(cell_cnv.x,cell_cnv.edge_index,cell_cnv.batch) 
        cell_embed = torch.cat([ge_embed,mut_embed,cnv_embed],dim = -1)
        x = torch.cat([drug_embed, cell_embed], -1)
        x = self.regression(x)
        return x
    
    
    
    
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, attention_dim):
        super(SelfAttention,self).__init__()
        self.embed_dim = embed_dim
        self.num_head = 2
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.head_dim = self.attention_dim // self.num_head
        assert self.head_dim * self.num_head == self.attention_dim, "attention_dim must be divisible by num_head"
        self.values = nn.Linear(1, self.attention_dim)
        self.keys = nn.Linear(1, self.attention_dim)
        self.queries = nn.Linear(1, self.attention_dim)
        self.fc_out = nn.Linear(self.attention_dim, self.attention_dim)
    def forward(self, values, keys, queries): ## Cell, Cell, Drug
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        values = values.unsqueeze(dim=-1)
        keys = keys.unsqueeze(dim=-1)
        queries = queries.unsqueeze(dim = -1)
        values = self.values(values)
        keys = self.keys(keys)
        queries_out = self.queries(queries)
        ## Split embedding into self.num_head pieces
        values = values.reshape(N, self.hidden_dim, self.num_head, self.head_dim)
        keys = keys.reshape(N , self.hidden_dim, self.num_head, self.head_dim)
        queries = queries_out.reshape(N, self.embed_dim, self.num_head, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)
        attention = torch.softmax(energy / (self.embed_dim ** (1/2)), dim = 3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_head * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.
        out = self.fc_out(out) 
        return out,queries_out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, attention_dim):
        super(TransformerBlock, self).__init__()
        self.attention_dim = attention_dim
        self.attention = SelfAttention(embed_dim, hidden_dim,attention_dim)
        self.embed_size = embed_dim
        self.norm1 = nn.LayerNorm(self.attention_dim)
        self.norm2 = nn.LayerNorm(self.attention_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.attention_dim, 2 * self.attention_dim),
            nn.ReLU(),
            nn.Linear(2 * self.attention_dim, self.attention_dim),
        )

        self.dropout = nn.Dropout(0.4)

    def forward(self, value, key, query):
        attention,queries = self.attention(value, key, query)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + queries))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class Multi_Omics_TransRegression(nn.Module):
    def __init__(self, model_config):
        super(Multi_Omics_TransRegression, self).__init__()
        self.hidden_dim = model_config.get('hidden_dim')
        self.embed_dim = model_config.get('embed_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.use_cnn = model_config.get('use_cnn')
        self.use_smiles = model_config.get('use_smiles')
        self.use_fp = model_config.get('use_fp')
        self.attention_dim = model_config.get('attention_dim')
        self.drug_encoder = drug_encoder(model_config)
        self.cell_encoder_ge = Cell_encoder_gene(model_config)
        self.cell_encoder_mut = Cell_encoder_mut(model_config)
        self.cell_encoder_cnv = Cell_encoder_cnv(model_config)
        self.graph_dim = self.drug_encoder.graph_dim
        self.gene_attn = TransformerBlock(self.graph_dim, self.hidden_dim, self.attention_dim)
        self.mut_attn = TransformerBlock(self.graph_dim, self.hidden_dim, self.attention_dim)
        self.cnv_attn = TransformerBlock(self.graph_dim, self.hidden_dim, self.attention_dim)
        self.regression = nn.Sequential(
            nn.Linear(3*self.graph_dim * self.attention_dim, 1024),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(1024,1) 
            # nn.ELU(),
            # nn.Dropout(p=self.dropout_rate),
            # nn.Linear(self.graph_dim, self.embed_dim),
            # nn.ELU(),
            # nn.Dropout(p=self.dropout_rate),
            # nn.Linear(self.embed_dim, 1)
        )

    def forward(self, drug_atom, drug_bond, ge, cell_cnv, cell_mut):
        drug_embed = self.drug_encoder(drug_atom,drug_bond)
        ge_embed = self.cell_encoder_ge(ge.x,ge.edge_index,ge.batch)
        mut_embed = self.cell_encoder_mut(cell_mut.x,cell_mut.edge_index,cell_mut.batch)
        cnv_embed = self.cell_encoder_cnv(cell_cnv.x,cell_cnv.edge_index,cell_cnv.batch) 
        attended_gene_feat = self.gene_attn(ge_embed, ge_embed, drug_embed)
        attended_mut_feat = self.mut_attn(mut_embed, mut_embed, drug_embed)
        attended_cnv_feat = self.cnv_attn(cnv_embed, cnv_embed, drug_embed)
        attended_gene_feat = attended_gene_feat.view(attended_gene_feat.size(0),-1)
        attended_mut_feat = attended_mut_feat.view(attended_mut_feat.size(0),-1)
        attended_cnv_feat = attended_cnv_feat.view(attended_cnv_feat.size(0),-1)
        attended_feat = torch.cat([attended_gene_feat, attended_mut_feat, attended_cnv_feat], dim=-1)
        x = self.regression(attended_feat)
        return x
 
 
 
    
def expr_recon_loss(recon_x, x):
    loss = ((recon_x - x)**2).sum()
    return loss
def kl_loss(mean, log_var):
    loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return loss   
    
class Drug_3d_VAE_resp(nn.Module):
    def __init__(self, model_config):
        super(Drug_3d_VAE_resp, self).__init__()
        self.hidden_dim = model_config.get('hidden_dim')
        self.embed_dim = model_config.get('embed_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.attention_dim = model_config.get('attention_dim')
        self.layer_num = model_config['layer_num']
        self.cell_encoder = GE_vae(model_config)
        self.drug_encoder = Drug_3d_Encoder(model_config)
        self.drug_emb = nn.Sequential(
            nn.Linear(self.embed_dim * (self.layer_num + 1), 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.cell_emb = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
        )

        self.regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(512, 1)
        )
    def forward(self, drug_atom, drug_bond, ge):
        z, recon_x, mean, log_var = self.cell_encoder(ge)
        cell_embeds = self.cell_emb(mean)
        drug_embeds = self.drug_encoder(drug_atom, drug_bond)
        drug_embeds = self.drug_emb(drug_embeds)
        x = torch.cat([drug_embeds, cell_embeds], -1)
        x = self.regression(x)
        loss_vae = expr_recon_loss(recon_x, x) + kl_loss(mean, log_var)   
        return x, loss_vae
    
    
    
class DRP_multi_view(nn.Module):
    def __init__(self, mut_cluster, cnv_cluster, ge_cluster,  model_config):
        super().__init__()
        self.dim_drug = model_config.get('embed_dim')
        self.use_cnn = model_config.get('use_cnn')
        self.layer_cell = model_config.get('layer_num')
        self.layer_drug = model_config.get('layer_num') + 1
        self.dim_cell = model_config.get('hidden_dim')
        self.dropout_ratio = model_config.get('dropout_rate')
        self.view_dim = model_config.get('view_dim')
        self.use_regulizer = model_config.get('use_regulizer')
        # self.dim_hvcdn = pow(self.view_dim,3)
        self.use_regulizer_drug = model_config.get('use_regulizer_drug')
        self.use_regulizer_pathway = model_config.get('use_drug_path_way')
        self.use_predined_gene_cluster = model_config.get('use_predined_gene_cluster')
        # drug graph branch
        self.GNN_drug = drug_hier_encoder(model_config)
        # self.drug_emb = nn.Sequential(
        #     nn.Linear(self.dim_drug * self.layer_drug, 256),
        #     nn.ReLU(),
        #     nn.Dropout(p=self.dropout_ratio),
        # )

        # cell graph branch
        if self.use_predined_gene_cluster == 'False':
            self.mut_model = GNN_cell_view(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=mut_cluster, omics_type = 'mut',dropout_ratio = self.dropout_ratio)
            self.cnv_model = GNN_cell_view(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=cnv_cluster, omics_type = 'cnv',dropout_ratio = self.dropout_ratio)
            self.ge_model = GNN_cell_view(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=ge_cluster, omics_type = 'ge',dropout_ratio = self.dropout_ratio)
        else:
            self.mut_model = GNN_cell_view_predifine(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=mut_cluster, omics_type = 'mut',dropout_ratio = self.dropout_ratio)
            self.cnv_model = GNN_cell_view_predifine(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=cnv_cluster, omics_type = 'cnv',dropout_ratio = self.dropout_ratio)
            self.ge_model = GNN_cell_view_predifine(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=ge_cluster, omics_type = 'ge',dropout_ratio = self.dropout_ratio)
        ## cell non-graph feature. 
        self.ge_encoder = GE_vae(model_config)
        if self.use_cnn == 'True':
            self.mut_encoder = cell_cnn(model_config, 2560)
            self.cnv_encoder = cell_cnn(model_config, 2816)
        
        
        self.ge_vae_emb = nn.Sequential(
            nn.Linear(self.dim_cell, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )
        if self.use_cnn == 'True':
            self.mut_cnn_emb = nn.Sequential(
                nn.Linear(self.dim_cell, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
            )
            self.cnv_cnn_emb = nn.Sequential(
                nn.Linear(self.dim_cell, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
            )
        if self.use_regulizer == 'True':
            if self.use_cnn == 'True':
                self.cell_regulizer = nn.Linear(256 * 6, 26) ## 26 cancer types
            else:
                self.cell_regulizer = nn.Linear(1024,26)
        if self.use_regulizer_drug == 'True':
            self.drug_regulizer = nn.Sequential(nn.Linear(2*self.dim_drug , 1024), 
                                            nn.ReLU(),
                                            nn.Linear(1024,1))
        if self.use_regulizer_pathway == 'True':
            self.drug_path_way_class = nn.Sequential(nn.Linear(2*self.dim_drug , 1024), 
                                            nn.ReLU(),
                                            nn.Linear(1024,23))
        self.regression_ge = nn.Sequential(
            nn.Linear(2*self.dim_drug + 256, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, self.view_dim),
            nn.ELU()
        )
        self.regression_cnv = nn.Sequential(
            nn.Linear(2*self.dim_drug + 256, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, self.view_dim),
            nn.ELU()
        )
        self.regression_mut = nn.Sequential(
            nn.Linear(2*self.dim_drug + 256, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, self.view_dim),
            nn.ELU()
        )
        self.regression_raw_ge = nn.Sequential(
            nn.Linear(2*self.dim_drug + 256, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, self.view_dim),
            nn.ELU()
        )
        if self.use_cnn == 'True':
            self.regression_raw_mut = nn.Sequential(
                nn.Linear(2*self.dim_drug + 256, 512),
                nn.ELU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(512, self.view_dim),
                nn.ELU()
            )
            self.regression_raw_cnv = nn.Sequential(
                nn.Linear(2*self.dim_drug + 256, 512),
                nn.ELU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(512, self.view_dim),
                nn.ELU()
            )
            self.pred_layer = nn.Sequential(
                nn.Linear(6*self.view_dim + 6*256 + 2*self.dim_drug, 512),
                nn.ELU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(512, 1)
            )
        else:
            self.pred_layer = nn.Sequential(
                nn.Linear(4*self.view_dim + 4*256 + 2*self.dim_drug, 512),
                nn.ELU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(512, 1)
            )
        # # fusion layers
        # self.fusion_layer = nn.Sequential(nn.Linear(2 * self.dim_drug + 3* 256, 1024),
        #                                   nn.BatchNorm1d(1024),
        #                                   nn.Linear(1024, 128),
        #                                   nn.BatchNorm1d(128),
        #                                   nn.Linear(128, 1))
    def forward(self, drug_atom, drug_bond, ge, mut, cnv, raw_gene):
        drug_class = None
        cell_class = None
        drug_pathway = None
        batch_size = drug_atom.batch.max()+1
        raw_mut = mut.x.view(batch_size,1,636)
        raw_cnv = cnv.x.view(batch_size,1,694)
        # forward drug
        x_drug = self.GNN_drug(drug_atom, drug_bond) ###[fp,repr] 2*embed_size
        # x_drug = self.drug_emb(x_drug)

        # forward cell
        x_ge = self.ge_model(ge)
        x_mut = self.mut_model (mut)
        x_cnv = self.cnv_model(cnv)
        z, recon_x, mean, log_var = self.ge_encoder(raw_gene)
        x_ge_vae = self.ge_vae_emb(mean)
        if self.use_cnn == 'True':
            x_cnn_mut = self.mut_encoder(raw_mut)
            x_cnn_mut = self.mut_cnn_emb(x_cnn_mut)
            x_cnn_cnv = self.cnv_encoder(raw_cnv)
            x_cnn_cnv = self.cnv_cnn_emb(x_cnn_cnv)
            cell_embed = torch.cat([x_ge, x_mut, x_cnv, x_ge_vae, x_cnn_mut, x_cnn_cnv], dim = -1)
        else: 
            cell_embed = torch.cat([x_ge, x_mut, x_cnv, x_ge_vae], dim = -1)
        # combine drug feature and cell line feature
        x_dg = torch.cat([x_drug, x_ge], -1)
        x_dm = torch.cat([x_drug, x_mut], -1)
        x_dc = torch.cat([x_drug, x_cnv], -1)
        x_dgr = torch.cat([x_drug, x_ge_vae], -1)
        x_dg, x_dm, x_dc, x_dgr = self.regression_ge(x_dg) ,self.regression_mut(x_dm), self.regression_cnv(x_dc), self.regression_raw_ge(x_dgr)
        if self.use_cnn == 'True':
            x_dmr, x_dcr = self.regression_ge(torch.cat([x_drug, x_cnn_mut], -1)), self.regression_ge(torch.cat([x_drug, x_cnn_cnv], -1))
            x = torch.cat([x_dg, x_dm, x_dc, x_dgr,x_dmr,x_dcr, x_drug, cell_embed], -1)
        else:
            x = torch.cat([x_dg, x_dm, x_dc, x_dgr, x_drug, cell_embed], -1)  ##Residual connection.        
        x = self.pred_layer(x)
        if self.use_regulizer == 'True':
            cell_class = self.cell_regulizer(cell_embed)
        # x = torch.cat([x_drug, x_ge, x_mut, x_cnv, x_ge_vae], dim = -1)
        # x = self.fusion_layer(x)
        if self.use_regulizer_drug == 'True':
            drug_class = self.drug_regulizer(x_drug)
        if self.use_regulizer_pathway =='True':
            drug_pathway = self.drug_path_way_class(x_drug)
        return {'pred': x, 'cell_regulizer':cell_class, 'drug_regulizer': drug_class, 'drug_pathway':drug_pathway}
    
    
    
class DRP_multi_view_ablation(nn.Module):
    def __init__(self, mut_cluster, cnv_cluster, ge_cluster,  model_config):
        super().__init__()
        self.dim_drug = model_config.get('embed_dim')
        self.use_cnn = model_config.get('use_cnn')
        self.layer_cell = model_config.get('layer_num')
        self.layer_drug = model_config.get('layer_num') + 1
        self.dim_cell = model_config.get('hidden_dim')
        self.dropout_ratio = model_config.get('dropout_rate')
        self.view_dim = model_config.get('view_dim')
        self.use_regulizer = model_config.get('use_regulizer')
        # self.dim_hvcdn = pow(self.view_dim,3)
        self.use_regulizer_drug = model_config.get('use_regulizer_drug')
        self.use_regulizer_pathway = model_config.get('use_drug_path_way')
        self.use_predined_gene_cluster = model_config.get('use_predined_gene_cluster')
        # drug graph branch
        self.GNN_drug = drug_hier_encoder(model_config)
        # self.drug_emb = nn.Sequential(
        #     nn.Linear(self.dim_drug * self.layer_drug, 256),
        #     nn.ReLU(),
        #     nn.Dropout(p=self.dropout_ratio),
        # )

        # cell graph branch
        if self.use_predined_gene_cluster == 'False':
            self.mut_model = GNN_cell_view(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=mut_cluster, omics_type = 'mut',dropout_ratio = self.dropout_ratio)
            self.cnv_model = GNN_cell_view(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=cnv_cluster, omics_type = 'cnv',dropout_ratio = self.dropout_ratio)
            self.ge_model = GNN_cell_view(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=ge_cluster, omics_type = 'ge',dropout_ratio = self.dropout_ratio)
        else:
            self.mut_model = GNN_cell_view_predifine(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=mut_cluster, omics_type = 'mut',dropout_ratio = self.dropout_ratio)
            self.cnv_model = GNN_cell_view_predifine(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=cnv_cluster, omics_type = 'cnv',dropout_ratio = self.dropout_ratio)
            self.ge_model = GNN_cell_view_predifine(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=ge_cluster, omics_type = 'ge',dropout_ratio = self.dropout_ratio)
        ## cell non-graph feature. 
        if self.use_cnn == 'True':
            self.mut_encoder = cell_cnn(model_config, 2560)
            self.cnv_encoder = cell_cnn(model_config, 2816)
        if self.use_cnn == 'True':
            self.mut_cnn_emb = nn.Sequential(
                nn.Linear(self.dim_cell, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
            )
            self.cnv_cnn_emb = nn.Sequential(
                nn.Linear(self.dim_cell, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
            )
        if self.use_regulizer == 'True':
            if self.use_cnn == 'True':
                self.cell_regulizer = nn.Linear(256 * 6, 26) ## 26 cancer types
            else:
                self.cell_regulizer = nn.Linear(1024,26)
        if self.use_regulizer_drug == 'True':
            self.drug_regulizer = nn.Sequential(nn.Linear(2*self.dim_drug , 1024), 
                                            nn.ReLU(),
                                            nn.Linear(1024,1))
        if self.use_regulizer_pathway == 'True':
            self.drug_path_way_class = nn.Sequential(nn.Linear(2*self.dim_drug , 1024), 
                                            nn.ReLU(),
                                            nn.Linear(1024,23))
        self.regression_ge = nn.Sequential(
            nn.Linear(2*self.dim_drug + 256, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, self.view_dim),
            nn.ELU()
        )
        self.regression_cnv = nn.Sequential(
            nn.Linear(2*self.dim_drug + 256, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, self.view_dim),
            nn.ELU()
        )
        self.regression_mut = nn.Sequential(
            nn.Linear(2*self.dim_drug + 256, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, self.view_dim),
            nn.ELU()
        )
        if self.use_cnn == 'True':
            self.regression_raw_mut = nn.Sequential(
                nn.Linear(2*self.dim_drug + 256, 512),
                nn.ELU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(512, self.view_dim),
                nn.ELU()
            )
            self.regression_raw_cnv = nn.Sequential(
                nn.Linear(2*self.dim_drug + 256, 512),
                nn.ELU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(512, self.view_dim),
                nn.ELU()
            )
            self.pred_layer = nn.Sequential(
                nn.Linear(5*self.view_dim + 5*256 + 2*self.dim_drug, 512),
                nn.ELU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(512, 1)
            )
        else:
            self.pred_layer = nn.Sequential(
                nn.Linear(3*self.view_dim + 3*256 + 2*self.dim_drug, 512),
                nn.ELU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(512, 1)
            )
        # # fusion layers
        # self.fusion_layer = nn.Sequential(nn.Linear(2 * self.dim_drug + 3* 256, 1024),
        #                                   nn.BatchNorm1d(1024),
        #                                   nn.Linear(1024, 128),
        #                                   nn.BatchNorm1d(128),
        #                                   nn.Linear(128, 1))
    def forward(self, drug_atom, drug_bond, ge, mut, cnv):
        drug_class = None
        cell_class = None
        drug_pathway = None
        batch_size = drug_atom.batch.max()+1
        raw_mut = mut.x.view(batch_size,1,636)
        raw_cnv = cnv.x.view(batch_size,1,694)
        # forward drug
        x_drug = self.GNN_drug(drug_atom, drug_bond) ###[fp,repr] 2*embed_size
        # x_drug = self.drug_emb(x_drug)

        # forward cell
        x_ge = self.ge_model(ge)
        x_mut = self.mut_model (mut)
        x_cnv = self.cnv_model(cnv)
        if self.use_cnn == 'True':
            x_cnn_mut = self.mut_encoder(raw_mut)
            x_cnn_mut = self.mut_cnn_emb(x_cnn_mut)
            x_cnn_cnv = self.cnv_encoder(raw_cnv)
            x_cnn_cnv = self.cnv_cnn_emb(x_cnn_cnv)
            cell_embed = torch.cat([x_ge, x_mut, x_cnv, x_cnn_mut, x_cnn_cnv], dim = -1)
        else: 
            cell_embed = torch.cat([x_ge, x_mut, x_cnv], dim = -1)
        # combine drug feature and cell line feature
        x_dg = torch.cat([x_drug, x_ge], -1)
        x_dm = torch.cat([x_drug, x_mut], -1)
        x_dc = torch.cat([x_drug, x_cnv], -1)
        x_dg, x_dm, x_dc = self.regression_ge(x_dg) ,self.regression_mut(x_dm), self.regression_cnv(x_dc)
        if self.use_cnn == 'True':
            x_dmr, x_dcr = self.regression_ge(torch.cat([x_drug, x_cnn_mut], -1)), self.regression_ge(torch.cat([x_drug, x_cnn_cnv], -1))
            x = torch.cat([x_dg, x_dm, x_dc,x_dmr,x_dcr, x_drug, cell_embed], -1)
        else:
            x = torch.cat([x_dg, x_dm, x_dc, x_drug, cell_embed], -1)  ##Residual connection.        
        x = self.pred_layer(x)
        if self.use_regulizer == 'True':
            cell_class = self.cell_regulizer(cell_embed)
        # x = torch.cat([x_drug, x_ge, x_mut, x_cnv, x_ge_vae], dim = -1)
        # x = self.fusion_layer(x)
        if self.use_regulizer_drug == 'True':
            drug_class = self.drug_regulizer(x_drug)
        if self.use_regulizer_pathway =='True':
            drug_pathway = self.drug_path_way_class(x_drug)
        return {'pred': x, 'cell_regulizer':cell_class, 'drug_regulizer': drug_class, 'drug_pathway':drug_pathway}
    
    
    
 
 
 
 
 
 
class GNN_drug_ablation(torch.nn.Module):
    def __init__(self, layer_drug, dim_drug):
        super().__init__()
        self.layer_drug = layer_drug
        self.dim_drug = dim_drug
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()
        self.atom_embedding = torch.nn.Embedding(get_atom_int_feature_dims()[0], self.dim_drug)
        for i in range(self.layer_drug):
            block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                    nn.Linear(self.dim_drug, self.dim_drug))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

    def forward(self, drug):
        x, edge_index, batch = drug.x[:,0].to(dtype=torch.int64), drug.edge_index, drug.batch
        x_drug_list = []
        x  = self.atom_embedding(x)
        for i in range(self.layer_drug):
            x = F.relu(self.convs_drug[i](x, edge_index))
            x = self.bns_drug[i](x)
            x_drug_list.append(x)

        node_representation = self.JK(x_drug_list)
        x_drug = global_max_pool(node_representation, batch)
        return x_drug
 
    
class DRP_multi_view_ablation_drug(nn.Module):
    def __init__(self, mut_cluster, cnv_cluster, ge_cluster,  model_config):
        super().__init__()
        self.dim_drug = model_config.get('embed_dim')
        self.use_cnn = model_config.get('use_cnn')
        self.layer_cell = model_config.get('layer_num')
        self.layer_drug = model_config.get('layer_num') + 1
        self.dim_cell = model_config.get('hidden_dim')
        self.dropout_ratio = model_config.get('dropout_rate')
        self.view_dim = model_config.get('view_dim')
        self.use_regulizer = model_config.get('use_regulizer')
        # self.dim_hvcdn = pow(self.view_dim,3)
        self.use_regulizer_drug = model_config.get('use_regulizer_drug')
        self.use_regulizer_pathway = model_config.get('use_drug_path_way')
        self.use_predined_gene_cluster = model_config.get('use_predined_gene_cluster')
        # drug graph branch
        self.GNN_drug = GNN_drug_ablation(layer_drug= self.layer_drug, dim_drug= self.dim_drug)
        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, self.dim_drug),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )
        # self.drug_emb = nn.Sequential(
        #     nn.Linear(self.dim_drug * self.layer_drug, 256),
        #     nn.ReLU(),
        #     nn.Dropout(p=self.dropout_ratio),
        # )

        # cell graph branch
        if self.use_predined_gene_cluster == 'False':
            self.mut_model = GNN_cell_view(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=mut_cluster, omics_type = 'mut',dropout_ratio = self.dropout_ratio)
            self.cnv_model = GNN_cell_view(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=cnv_cluster, omics_type = 'cnv',dropout_ratio = self.dropout_ratio)
            self.ge_model = GNN_cell_view(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=ge_cluster, omics_type = 'ge',dropout_ratio = self.dropout_ratio)
        else:
            self.mut_model = GNN_cell_view_predifine(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=mut_cluster, omics_type = 'mut',dropout_ratio = self.dropout_ratio)
            self.cnv_model = GNN_cell_view_predifine(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=cnv_cluster, omics_type = 'cnv',dropout_ratio = self.dropout_ratio)
            self.ge_model = GNN_cell_view_predifine(layer_cell=self.layer_cell, dim_cell=self.dim_cell, cluster_predefine=ge_cluster, omics_type = 'ge',dropout_ratio = self.dropout_ratio)
        ## cell non-graph feature. 
        if self.use_cnn == 'True':
            self.mut_encoder = cell_cnn(model_config, 2560)
            self.cnv_encoder = cell_cnn(model_config, 2816)
        if self.use_cnn == 'True':
            self.mut_cnn_emb = nn.Sequential(
                nn.Linear(self.dim_cell, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
            )
            self.cnv_cnn_emb = nn.Sequential(
                nn.Linear(self.dim_cell, 1024),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
            )
        if self.use_regulizer == 'True':
            if self.use_cnn == 'True':
                self.cell_regulizer = nn.Linear(256 * 6, 26) ## 26 cancer types
            else:
                self.cell_regulizer = nn.Linear(1024,26)
        if self.use_regulizer_drug == 'True':
            self.drug_regulizer = nn.Sequential(nn.Linear(self.dim_drug , 1024), 
                                            nn.ReLU(),
                                            nn.Linear(1024,1))
        if self.use_regulizer_pathway == 'True':
            self.drug_path_way_class = nn.Sequential(nn.Linear(self.dim_drug , 1024), 
                                            nn.ReLU(),
                                            nn.Linear(1024,23))
        self.regression_ge = nn.Sequential(
            nn.Linear(self.dim_drug + 256, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, self.view_dim),
            nn.ELU()
        )
        self.regression_cnv = nn.Sequential(
            nn.Linear(self.dim_drug + 256, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, self.view_dim),
            nn.ELU()
        )
        self.regression_mut = nn.Sequential(
            nn.Linear(self.dim_drug + 256, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, self.view_dim),
            nn.ELU()
        )
        if self.use_cnn == 'True':
            self.regression_raw_mut = nn.Sequential(
                nn.Linear(self.dim_drug + 256, 512),
                nn.ELU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(512, self.view_dim),
                nn.ELU()
            )
            self.regression_raw_cnv = nn.Sequential(
                nn.Linear(self.dim_drug + 256, 512),
                nn.ELU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(512, self.view_dim),
                nn.ELU()
            )
            self.pred_layer = nn.Sequential(
                nn.Linear(5*self.view_dim + 5*256 + self.dim_drug, 512),
                nn.ELU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(512, 1)
            )
        else:
            self.pred_layer = nn.Sequential(
                nn.Linear(3*self.view_dim + 3*256 + self.dim_drug, 512),
                nn.ELU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(512, 1)
            )
        # # fusion layers
        # self.fusion_layer = nn.Sequential(nn.Linear(2 * self.dim_drug + 3* 256, 1024),
        #                                   nn.BatchNorm1d(1024),
        #                                   nn.Linear(1024, 128),
        #                                   nn.BatchNorm1d(128),
        #                                   nn.Linear(128, 1))
    def forward(self, drug_atom, drug_bond, ge, mut, cnv):
        drug_class = None
        cell_class = None
        drug_pathway = None
        batch_size = drug_atom.batch.max()+1
        raw_mut = mut.x.view(batch_size,1,636)
        raw_cnv = cnv.x.view(batch_size,1,694)
        # forward drug
        x_drug = self.GNN_drug(drug_atom) ###[fp,repr] 2*embed_size
        x_drug = self.drug_emb(x_drug)

        # forward cell
        x_ge = self.ge_model(ge)
        x_mut = self.mut_model (mut)
        x_cnv = self.cnv_model(cnv)
        if self.use_cnn == 'True':
            x_cnn_mut = self.mut_encoder(raw_mut)
            x_cnn_mut = self.mut_cnn_emb(x_cnn_mut)
            x_cnn_cnv = self.cnv_encoder(raw_cnv)
            x_cnn_cnv = self.cnv_cnn_emb(x_cnn_cnv)
            cell_embed = torch.cat([x_ge, x_mut, x_cnv, x_cnn_mut, x_cnn_cnv], dim = -1)
        else: 
            cell_embed = torch.cat([x_ge, x_mut, x_cnv], dim = -1)
        # combine drug feature and cell line feature
        x_dg = torch.cat([x_drug, x_ge], -1)
        x_dm = torch.cat([x_drug, x_mut], -1)
        x_dc = torch.cat([x_drug, x_cnv], -1)
        x_dg, x_dm, x_dc = self.regression_ge(x_dg) ,self.regression_mut(x_dm), self.regression_cnv(x_dc)
        if self.use_cnn == 'True':
            x_dmr, x_dcr = self.regression_ge(torch.cat([x_drug, x_cnn_mut], -1)), self.regression_ge(torch.cat([x_drug, x_cnn_cnv], -1))
            x = torch.cat([x_dg, x_dm, x_dc,x_dmr,x_dcr, x_drug, cell_embed], -1)
        else:
            x = torch.cat([x_dg, x_dm, x_dc, x_drug, cell_embed], -1)  ##Residual connection.        
        x = self.pred_layer(x)
        if self.use_regulizer == 'True':
            cell_class = self.cell_regulizer(cell_embed)
        # x = torch.cat([x_drug, x_ge, x_mut, x_cnv, x_ge_vae], dim = -1)
        # x = self.fusion_layer(x)
        if self.use_regulizer_drug == 'True':
            drug_class = self.drug_regulizer(x_drug)
        if self.use_regulizer_pathway =='True':
            drug_pathway = self.drug_path_way_class(x_drug)
        return {'pred': x, 'cell_regulizer':cell_class, 'drug_regulizer': drug_class, 'drug_pathway':drug_pathway}