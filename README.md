# Hi-GeoMVP
Hi-GeoMVP is a deep-leaning model using graph neural networks to represent the drug and cell line features for drug response prediction.
# How to use:
## 1. Generate dataset:
    Run create_cell_feat.py to generate the used cell_line feature.
    Run create_drug_feat.py to generate the used drug feature. 
    Run create_drp_dict.py to generate the drug_cell_ic50 feature and train_test split for mix,cell_blind and drug_blind sets.
    All the used data are saved in .npy format and under the ./Data/DRP_dataset
    An example to create the dataset and dataloader in Pytorch is provided in data_preprocess.py
## 2. Train the model:
    python train_GeoMVP_5fold.py --train_type {train_type} 
    --devie {device}--use_norm_ic50' {use_norm_ic50}  
    --use_drug_path_way {use_drug_path_way} 
    --use_regulizer {use_regulizer}
    --use_regulizer_drug {use_regulizer_drug} 
    --use_raw_gen {use_raw_gen}

