import math
import os
import random
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("prepare_data")
root = os.getcwd()
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
# def n_fold_split(type='mix', drug_name = None, cell_name = None, drug_response_dict = None, seed = 0, n_folds = 5): ## type: mix, cb, db
#    random.seed(seed)
#    if type == 'mix':
#         num_total = len(drug_response_dict)
#         indices = np.arange(num_total)
#         np.random.shuffle(indices)
#         fold_size = num_total // n_folds
#         train_dict = { } 
#         val_dict = {}
#         test_dict = {}
#         for i in range(5):
#             validation_start = i * fold_size
#             test_start = (i + 1) * fold_size
#             train_start = (i + 2) * fold_size
#             validation_indices = indices[validation_start:test_start]
#             test_indices = indices[test_start:train_start] if i < n_folds - 1 else np.concatenate((indices[test_start:],indices[:fold_size]))
#             train_indices = np.concatenate((indices[:validation_start], indices[train_start:])) if i < n_folds - 1 else indices[fold_size:validation_start]
#             train_dict[i] = list(train_indices)
#             val_dict[i] = list(validation_indices)
#             test_dict[i] = list(test_indices)    
#    elif type == 'cb':
#         num_cells = len(cell_name)
#         indices = np.arange(num_cells)
#         np.random.shuffle(indices)
#         cell_folds = np.array_split(indices, n_folds)
#         train_dict = { } 
#         val_dict = {}
#         test_dict = {}
#         for i in range(5):
#             val_cell_indices = cell_folds[i]
#             test_cell_indices = cell_folds[(i + 1) % n_folds]
#             train_cell_indices = np.concatenate([cell_folds[j] for j in range(n_folds) if j != i and j != (i + 1) % n_folds])
#             train_idx = [idx for idx, [cell, drug,
#                                                     ic50,norm_ic50] in enumerate(drug_response_dict) if cell in cell_name[train_cell_indices]]
#             test_idx = [idx for idx,[cell,
#                                                 drug, ic50,norm_ic50] in enumerate(drug_response_dict) if cell in cell_name[test_cell_indices]]  
#             val_idx = [idx for idx, [cell, drug,
#                                                     ic50,norm_ic50] in enumerate(drug_response_dict) if cell in cell_name[val_cell_indices]]
#             train_dict[i] = train_idx
#             test_dict[i] = test_idx
#             val_dict[i] = val_idx
#    elif type == 'db':
#         num_drugs = len(drug_name)
#         indices = np.arange(num_drugs)
#         np.random.shuffle(indices)
#         drug_folds = np.array_split(indices, n_folds)
#         train_dict = { } 
#         val_dict = {}
#         test_dict = {}
#         for i in range(5):
#             val_drug_indices = drug_folds[i]
#             test_drug_indices = drug_folds[(i + 1) % n_folds]
#             train_drug_indices = np.concatenate([drug_folds[j] for j in range(n_folds) if j != i and j != (i + 1) % n_folds])
#             train_idx = [idx for idx, [cell, drug,
#                                                     ic50,norm_ic50] in enumerate(drug_response_dict) if drug in drug_name[train_drug_indices]]
#             test_idx = [idx for idx,[cell,
#                                                 drug, ic50,norm_ic50] in enumerate(drug_response_dict) if drug in drug_name[test_drug_indices]]  
#             val_idx = [idx for idx, [cell, drug,
#                                                     ic50,norm_ic50] in enumerate(drug_response_dict) if drug in drug_name[val_drug_indices]]
#             train_dict[i] = train_idx
#             test_dict[i] = test_idx
#             val_dict[i] = val_idx
#    return train_dict, val_dict, test_dict

def n_fold_split(type='mix', drug_name = None, cell_name = None, drug_response_dict = None, seed = 0, n_folds = 10, test_portion = 0.1): ## type: mix, cb, db
   random.seed(seed)
   if type == 'mix':
        random.seed(seed)
        np.random.seed(seed)
        num_total = len(drug_response_dict)
        indices = np.arange(num_total)
        np.random.shuffle(indices)
        test_size = int(num_total * 0.1)
        train_val_size = num_total - test_size
        test_idx = indices[:test_size]
        train_val_indices = indices[test_size:]
        fold_size = train_val_size // n_folds
        train_dict = { } 
        val_dict = {}
        for i in range(5):
            validation_indices = train_val_indices[i * fold_size:(i+1) * fold_size]
            train_indices = np.concatenate((train_val_indices[:i * fold_size], train_val_indices[(i + 1) * fold_size:]))
            train_dict[i] = list(train_indices)
            val_dict[i] = list(validation_indices)
   elif type == 'cb':
        random.seed(seed)
        np.random.seed(seed)
        num_cells = len(cell_name)
        cell_indices = np.arange(num_cells)
        np.random.shuffle(cell_indices)
        test_size = int(num_cells * 0.1)
        train_val_size = num_cells - test_size
        test_cell_indices = cell_indices[:test_size]
        cell_folds = np.array_split(cell_indices[test_size:], n_folds)
        test_idx = [idx for idx,[cell,
                                            drug, ic50,norm_ic50] in enumerate(drug_response_dict) if cell in cell_name[test_cell_indices]] 
        train_dict = { } 
        val_dict = {}
        for i in range(5):
            val_cell_indices = cell_folds[i]
            train_cell_indices = np.concatenate([cell_folds[j] for j in range(n_folds) if j != i])
            train_idx = [idx for idx, [cell, drug,
                                                    ic50,norm_ic50] in enumerate(drug_response_dict) if cell in cell_name[train_cell_indices]]
            val_idx = [idx for idx, [cell, drug,
                                                    ic50,norm_ic50] in enumerate(drug_response_dict) if cell in cell_name[val_cell_indices]]
            train_dict[i] = train_idx
            val_dict[i] = val_idx
   elif type == 'db':
        random.seed(seed)
        np.random.seed(seed)
        num_drugs = len(drug_name)
        drug_indices = np.arange(num_drugs)
        np.random.shuffle(drug_indices)
        test_size = int(num_drugs * 0.1)
        train_val_size = num_drugs - test_size
        test_drug_indices = drug_indices[:test_size]
        drug_folds = np.array_split(drug_indices[test_size:], n_folds)
        test_idx = [idx for idx,[cell,
                                            drug, ic50,norm_ic50] in enumerate(drug_response_dict) if drug in drug_name[test_drug_indices]]  
        train_dict = { } 
        val_dict = {}
        for i in range(5):
            val_drug_indices = drug_folds[i]
            test_drug_indices = drug_folds[(i + 1) % n_folds]
            train_drug_indices = np.concatenate([drug_folds[j] for j in range(n_folds) if j != i ])
            train_idx = [idx for idx, [cell, drug,
                                                    ic50,norm_ic50] in enumerate(drug_response_dict) if drug in drug_name[train_drug_indices]]
            val_idx = [idx for idx, [cell, drug,
                                                    ic50,norm_ic50] in enumerate(drug_response_dict) if drug in drug_name[val_drug_indices]]
            train_dict[i] = train_idx
            val_dict[i] = val_idx
   return train_dict, val_dict, test_idx


if __name__ == '__main__':
    if os.path.exists(root + f"/Data/DRP_dataset") is False:
        os.makedirs(root + f"/Data/DRP_dataset")
    path_ic50 = root + '/Data/GDSC_data/IC50reduced.csv'
    path_drug_id_name = root + '/Data/GDSC_data/drugid_name.txt'
    drug_response_dict, drug_name, cell_name = create_drp_dict(path_ic50,path_drug_id_name)
    drug_response_dict, drug_name, cell_name = read_dr_dict()
    train_idx, text_idx = create_drp_set(type='mix', drug_name = drug_name, cell_name = cell_name, drug_response_dict = drug_response_dict, seed = 0)
