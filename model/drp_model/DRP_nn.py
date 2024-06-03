import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')
from model.cell_model.cell_encoder import *
from model.drug_model.drug_encoder import *

 
class DRP_multi_view(nn.Module):
    """
        Multi-view deep learning model designed for drug response prediction.
        
        The model integrates multiple biological data. It leverages these diverse data views to enhance the prediction accuracy of drug
        response. The architecture of the model is flexible and can be configured with different parameters as specified in
        the model_config dictionary.
        
        Attributes:
            dim_drug (int): Dimension of the drug embeddings.
            use_cnn (bool): Flag indicating whether to use convolutional neural networks (CNN) for seq/genomics features of cell line.
            layer_cell (int): Number of layers in the cell feature extraction module.
            layer_drug (int): Number of layers in the drug feature extraction module.
            dim_cell (int): Dimension of the hidden layers in the cell feature extraction module.
            dropout_ratio (float): Dropout rate applied to the layers to prevent overfitting.
            view_dim (int): Dimensionality of the integrated multi-view representation.
            use_regulizer (bool): Flag indicating whether to apply regularization on the cancer_type.
            use_regulizer_drug (bool): Flag indicating whether to apply regularization specifically on drug representations.
            use_regulizer_pathway (bool): Flag indicating whether to use predefined drug pathways for regularization.
            use_predined_gene_cluster (bool): Flag indicating whether to use predefined gene clusters for regularization.
        
        Args:
            mut_cluster (torch.Tensor): Mutation cluster data.
            cnv_cluster (torch.Tensor): Copy number variation cluster data.
            ge_cluster (torch.Tensor): Gene expression cluster data.
            model_config (dict): Configuration dictionary specifying various hyperparameters for the model.
        """
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
        self.use_regulizer_drug = model_config.get('use_regulizer_drug')
        self.use_regulizer_pathway = model_config.get('use_drug_path_way')
        self.use_predined_gene_cluster = model_config.get('use_predined_gene_cluster')
        # drug graph branch
        self.GNN_drug = drug_hier_encoder(model_config)
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
    