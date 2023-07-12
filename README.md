# Atten_Geom_DRP
Atten_Geom_DRP is a deep leaning model using graph neural network to represent the drug and cell line featrues.
# How to use:
## 1. Generate dataset:
    Run create_cell_feat.py to generate the used cell_line feature.
    Run create_drug_feat.py to generate the used drug feature. 
    Run create_drp_dict.py to generate the drug_cell_ic50 feature and train_test split for mix,cell_blind and drug_blind sets.
    All the used data are saved in .npy format and under the ./Data/DRP_dataset
## 2. Train the model:
    python train_multi_omcis_drp.py --train_type {train_type} --use_norm_ic50' {use_norm_ic50} --devie {device}
