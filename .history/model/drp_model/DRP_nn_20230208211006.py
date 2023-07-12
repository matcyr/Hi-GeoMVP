import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')
from model.cell_model.cell_encoder import *
from model.drug_model.drug_encoder import *
import math

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class AttentionBlock(nn.Module):
    def __init__(self,model_config) -> None:
        super(AttentionBlock,self).__init__()
        self.embed_dim = model_config.get('embed_dim')
        self.hidden_dim = model_config.get('hidden_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.use_cnn = model_config.get('use_cnn')
        self.use_smiles = model_config.get('use_smiles')
        self.use_fp = model_config.get('use_fp')
        self.atten_dim = model_config.get('atten_dim')
        self.drug_dim = self.embed_dim + self.use_fp * self.embed_dim + self.use_smiles * self.embed_dim
        self.cell_dim = model_config.get('hidden_dim')
        self.Q_drug_proj = nn.Linear(self.drug_dim, self.atten_dim)
        self.K_omics_proj = nn.Linear(self.cell_dim, self.atten_dim)
        self.V_ic50_proj = nn.Linear(self.cell_dim+self.drug_dim, self.atten_dim)
    def forward(self,x_omics,x_drug):
        Q_drug = self.Q_drug_proj(x_drug)
        K_omics = self.K_omics_proj(x_omics)
        V_ic50 = self.V_ic50_proj(torch.cat([x_drug,x_omics],-1))    
        pred, attention = scaled_dot_product(Q_drug, K_omics, V_ic50)
        return pred, attention
    
class Multi_Omics_Regression(nn.Module):
    def __init__(self, model_config):
        super(Multi_Omics_Regression, self).__init__()
        self.hidden_dim = model_config.get('hidden_dim')
        self.embed_dim = model_config.get('embed_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.atten_dim = model_config.get('atten_dim')
        self.use_cnn = model_config.get('use_cnn')
        self.use_smiles = model_config.get('use_smiles')
        self.use_fp = model_config.get('use_fp')
        self.gene_attn = AttentionBlock(model_config)
        self.mut_attn = AttentionBlock(model_config)
        self.cnv_attn = AttentionBlock(model_config)
        self.regression = nn.Sequential(
            nn.Linear(3*self.atten_dim, self.atten_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.atten_dim, self.atten_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.atten_dim, 1)
        )
        self.drug_encoder = drug_encoder(model_config)
        self.cell_encoder = Cell_multi_omics_Encoder(model_config)
    def forward(self, drug_atom, drug_bond, ge, ge_sim, cnv, mut):
        drug_feat = self.drug_encoder(drug_atom,drug_bond)
        cnv_feat,mut_feat,gene_feat = self.cell_encoder(ge, ge_sim, cnv, mut)
        attended_gene_feat,attention_weights_gene = self.gene_attn(gene_feat, drug_feat)
        attended_mut_feat, attention_weights_mut = self.mut_attn(mut_feat, drug_feat)
        attended_cnv_feat, attention_weights_cnv = self.cnv_attn(cnv_feat, drug_feat)
        attended_feat = torch.cat([attended_gene_feat, attended_mut_feat, attended_cnv_feat], dim=-1)
        attend_list = [attention_weights_gene, attention_weights_mut, attention_weights_cnv]
        x = self.regression(attended_feat)
        return x

class Single_Omics_Regression(nn.Module):
    def __init__(self, model_config):
        super(Single_Omics_Regression, self).__init__()
        self.hidden_dim = model_config.get('hidden_dim')
        self.embed_dim = model_config.get('embed_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.atten_dim = model_config.get('atten_dim')
        self.use_cnn = model_config.get('use_cnn')
        self.use_smiles = model_config.get('use_smiles')
        self.use_fp = model_config.get('use_fp')
        self.gene_attn = AttentionBlock(model_config)
        self.regression = nn.Sequential(
            nn.Linear(self.atten_dim, self.atten_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.atten_dim, self.atten_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.atten_dim, 1)
        )
        self.drug_encoder = Drug_3d_Encoder(model_config)
        self.cell_encoder = Cell_encoder_gene(model_config)
    def forward(self, drug_atom, drug_bond, ge):
        drug_feat = self.drug_encoder(drug_atom,drug_bond)
        gene_feat = self.cell_encoder(ge.x, ge.edge_index, ge.batch)
        attended_gene_feat,attention_weights_gene = self.gene_attn(gene_feat, drug_feat)
        x = self.regression(attended_gene_feat)
        return x

class DRP(nn.Module):
    def __init__(self, model_config) -> None:
        super().__init__()
        self.embed_dim = model_config.get('embed_dim')
        self._hidden_dim = model_config.get('hidden_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.drug_encoder = Drug_3d_Encoder(model_config)
        self.cell_encoder = Cell_encoder_gene(model_config)
        self.regression = nn.Sequential(
            nn.Linear(self._hidden_dim + self.embed_dim, self.embed_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, drug_atom,drug_bond,ge):
        drug_embed = self.drug_encoder(drug_atom,drug_bond)
        cell_embed = self.cell_encoder(ge.x,ge.edge_index,ge.batch)
        x = torch.cat([drug_embed, cell_embed], -1)
        x = self.regression(x)
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
    def __init__(self, embed_dim, heads):
        super(SelfAttention,self).__init__()
        self.embed_dim = embed_dim
        self.num_head = heads
        self.head_dim = self.embed_dim_dim // self.num_head
        assert self.head_dim * self.num_head == self.embed_dim, "embed_dim must be divisible by num_head"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim)