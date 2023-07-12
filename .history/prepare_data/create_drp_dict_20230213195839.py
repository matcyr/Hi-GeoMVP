import math
import os
import random
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("prepare_data")
root = 
def create_drp_dict(path_ic50,path_drug_id_name):
    IC50table = pd.read_csv(path_ic50)
    IC50table = IC50table.rename(columns={'0': 'cell_id'})
    IC50table = IC50table.set_index('cell_id')
    drug_id_name = pd.read_csv(path_drug_id_name, sep='\t', header=None)
    drug_id_name.rename(columns={0: 'id', 1: 'name'}, inplace=True)
    def re_drug(x): return drug_id_name[drug_id_name['name'] == x]['id'].item()
    IC50table.rename(columns={x: re_drug(x)
                              for x in IC50table.columns}, inplace=True)
    cell_drug_interaction = [[cell, drug]
                             for cell in IC50table.index for drug in IC50table.columns]
    # Remove all the nan value
    drug_response_dict = [[int(cell), int(drug), IC50table[drug][cell], 1/(1+math.exp(IC50table[drug][cell])**(-0.1))]
                          for [cell, drug] in cell_drug_interaction if not np.isnan(IC50table[drug][cell])]
    drug_name = IC50table.columns
    cell_name = IC50table.index
    def save_dr_dict():
        save_path = root + '/Data/DRP_dataset'
        np.save(os.path.join(save_path, 'drug_response.npy'), drug_response_dict)
        np.save(os.path.join(save_path, 'drug_name.npy'), drug_name)
        np.save(os.path.join(save_path, 'cell_name.npy'), cell_name)
        print("finish saving drug response data!")
    save_dr_dict()
    return drug_response_dict, drug_name, cell_name
def read_dr_dict():
    drug_response_dict = np.load(root + '/Data/DRP_dataset/drug_response.npy',allow_pickle=True)
    drug_name = np.load(root + '/Data/DRP_dataset/drug_name.npy',allow_pickle=True)
    cell_name = np.load(root + '/Data/DRP_dataset/cell_name.npy',allow_pickle=True)
    return drug_response_dict, drug_name, cell_name
def create_drp_set(type='mix', drug_name = None, cell_name = None, drug_response_dict = None, seed = 0): ## type: mix, cb, db
    random.seed(seed)
    train_portion = 0.2
    num_cell = cell_name.shape[0]
    num_drug = drug_name.shape[0]
    num_total = len(drug_response_dict)
    test_cell_id = random.sample(range(num_cell), int(train_portion*num_cell))
    test_cell = cell_name[test_cell_id]
    test_drug_id = random.sample(range(num_drug), int(train_portion*num_drug))
    test_drug = drug_name[test_drug_id]
    test_id = random.sample(range(num_total), int(train_portion*num_total))
    train_id = list(set([i for i in range(num_total)])-set(test_id))
    # Create mixed set:
    if type == 'mix':
        train_idx = train_id
        test_idx = test_id
    # Create cell blind set:
    elif type == 'cb':
        train_idx = [idx for idx, [cell, drug,
                                             ic50,norm_ic50] in enumerate(drug_response_dict) if cell not in test_cell]
        test_idx = [idx for idx,[cell,
                                            drug, ic50,norm_ic50] in enumerate(drug_response_dict) if cell in test_cell]  
    # Create drug blind set:
    elif type == 'db':
        train_idx = [idx for idx, [cell, drug,
                                             ic50,norm_ic50] in enumerate(drug_response_dict) if drug not in test_drug]
        test_idx = [idx for idx, [cell,
                                            drug, ic50,norm_ic50] in enumerate(drug_response_dict) if drug in test_drug]
    return train_idx, test_idx


if __name__ == '__main__':
    if os.path.exists(root + f"/Data/DRP_dataset") is False:
        os.makedirs(root + f"/Data/DRP_dataset")
    path_ic50 = root + '/Data/GDSC_data/IC50reduced.csv'
    path_drug_id_name = root + '/Data/GDSC_data/drugid_name.txt'
    drug_response_dict, drug_name, cell_name = create_drp_dict(path_ic50,path_drug_id_name)
    drug_response_dict, drug_name, cell_name = read_dr_dict()
    train_idx, text_idx = create_drp_set(type='mix', drug_name = drug_name, cell_name = cell_name, drug_response_dict = drug_response_dict, seed = 0)
