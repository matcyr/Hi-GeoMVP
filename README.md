# Hi-GeoMVP: a hierarchical geometry-enhanced deep learning model for drug response prediction
Hi-GeoMVP is a deep-leaning model using graph neural networks to represent the drug and cell line features for drug response prediction.

![plot](https://github.com/matcyr/Hi-GeoMVP/blob/main/model_structure/Model_arch.pdf)

# How to use:
## 1. Generate dataset:
    Run create_cell_feat.py to generate the used cell_line feature.
    Run create_drug_feat.py to generate the used drug feature. 
    Run create_drp_dict.py to generate the drug_cell_ic50 feature and train_test split for mix,cell_blind and drug_blind sets.
    All the used data are saved in .npy format and under the ./Data/DRP_dataset
    An example to create the dataset and dataloader in Pytorch is provided in data_preprocess.py
## 2. Train the model:
We provide the script to train the model with different settings and hyperparameters:  
train_type: 'cb', 'db', or 'mix' for cell blind, drug blind, and mix setting.  
use_norm_ic50: 'True' or 'False', for usage of normalized IC50 value.  
use_regulizer: 'True' or 'False', usage of MTL cancer type.  
use_regulizer_drug: 'True' or 'False', usage of MTL drug threshold.  
use_drug_path_way: 'True' or 'False', usage of MTL drug targeting pathway.  
use_raw_gene: 'True' or 'False', usage of whole gene expression data.
regular_weight: float number, the weight of MTL cancer type in the loss function.  
regular_weight_drug: float number, the weight of MTL drug threshold in the loss function.  
regular_weight_drug_path_way: float number, the weight of drug targeting pathway in the loss function.  

You can run the training in the command line with:  

    python train_GeoMVP_5fold.py --train_type {train_type} 
        --device {device}--use_norm_ic50' {use_norm_ic50}  
        --use_drug_path_way {use_drug_path_way} 
        --use_regulizer {use_regulizer}
        --use_regulizer_drug {use_regulizer_drug} 
        --use_raw_gen {use_raw_gen}


