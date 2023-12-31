import sys
sys.path.append('prepare_data')
import random
from create_cell_feat import *
from create_drug_feat import *
from create_drp_dict import *
from binarization_drp import *
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
GE_data = load_raw_GE()
drug_response_dict, drug_name, cell_name = read_dr_dict()
ge_HN_feat, ge_sim_dict, cnv_dict, mut_dict = load_cell_feat()
drug_atom_dict,drug_bond_dict = load_drug_feat()
drug_response_binary, drug_threshold = load_binary_ic50()
# train_idx,test_idx = create_drp_set(type= 'mix', drug_name = drug_name, cell_name = cell_name, drug_response_dict = drug_response_dict, seed = 0)
def split_drug_set(seed = 0):
    random.seed(seed)
    train_portion = 0.2
    num_drug = drug_name.shape[0]
    test_drug_id = random.sample(range(num_drug), int(train_portion*num_drug))
    train_drug_id = list(set([i for i in range(num_drug)])-set(test_drug_id))
    return train_drug_id,test_drug_id

# class drug_classfication_dataset(Dataset):
#     def __init__(self,drug_idx):
#         super(drug_classfication_dataset, self).__init__()
#         self.drug_atom, self.drug_bond_dict, self.target_drug, self.drug_threshold, self.name = drug_atom_dict, drug_bond_dict, drug_target_pathway, drug_threshold, drug_name[drug_idx]
#         self.length = len(self.name)
#     def __len__(self):
#         return self.length
#     def __getitem__(self, index):
#         drug = self.name[index]
#         return (self.drug_atom[drug], self.drug_bond_dict[drug], self.target_drug[drug], self.drug_threshold[drug])
# def _collate_drug_class(samples):
#     drug_atom, drug_bond, target, threshold = map(list, zip(*samples))
#     batched_drug_atom = Batch.from_data_list(drug_atom)
#     batched_drug_bond = Batch.from_data_list(drug_bond)
#     batched_target = torch.cat(target, dim = 1).t()
#     return batched_drug_atom, batched_drug_bond, batched_target, torch.tensor(threshold) ## batched_target : [batch_size, num_target]
# def drug_class_loader(data_set,batch_size,shuffle = True, num_workers = 4):
#     data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_drug_class, num_workers=num_workers,pin_memory=True)
#     return data_loader


class multi_DRP_dataset(Dataset):
    def __init__(self, drp_idx ,use_norm_ic50 = 'True', use_raw_gene = 'False'):
        super(multi_DRP_dataset, self).__init__()
        self.drug_atom, self.drug_bond_dict, self.ge_dict, self.cnv_dict, self.mut_dict, self.DR = drug_atom_dict, drug_bond_dict, ge_HN_feat, cnv_dict, mut_dict, drug_response_dict[drp_idx]
        self.raw_gene = GE_data
        self.use_raw_gene = use_raw_gene
        self.use_norm_ic50 = use_norm_ic50
        self.length = len(self.DR)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        if self.use_norm_ic50 == 'True':
            cell, drug, _, ic = self.DR[index]
        else: cell, drug, ic, _ = self.DR[index]
        if self.use_raw_gene == 'False':
            return (self.drug_atom[drug], self.drug_bond_dict[drug], self.ge_dict[cell], self.cnv_dict[cell], self.mut_dict[cell], ic)
        else:
            return (self.drug_atom[drug], self.drug_bond_dict[drug], self.ge_dict[cell], self.cnv_dict[cell], self.mut_dict[cell], self.raw_gene[cell], ic)
def _collate_multi_omics(samples):
    drug_atom, drug_bond, ge, cnv, mut, labels = map(list, zip(*samples))
    batched_drug_atom = Batch.from_data_list(drug_atom)
    batched_drug_bond = Batch.from_data_list(drug_bond)
    batch_ge = Batch.from_data_list(ge)
    batch_cnv = Batch.from_data_list(cnv)
    batch_mut = Batch.from_data_list(mut)
    return batched_drug_atom, batched_drug_bond, batch_ge, batch_cnv, batch_mut, torch.tensor(labels)
def _collate_multi_omics_with_raw_gene(samples):
    drug_atom, drug_bond, ge, cnv, mut, raw_gene, labels = map(list, zip(*samples))
    batched_drug_atom = Batch.from_data_list(drug_atom)
    batched_drug_bond = Batch.from_data_list(drug_bond)
    batch_ge = Batch.from_data_list(ge)
    batch_cnv = Batch.from_data_list(cnv)
    batch_mut = Batch.from_data_list(mut)
    return batched_drug_atom, batched_drug_bond, batch_ge, batch_cnv, batch_mut, torch.stack(raw_gene).float(), torch.tensor(labels)
def multi_drp_loader(data_set,batch_size,shuffle = True, num_workers = 4, use_raw_gene = 'False'):
    if use_raw_gene == 'False':
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_multi_omics, num_workers=num_workers,pin_memory=True)
    else: data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_multi_omics_with_raw_gene, num_workers=num_workers,pin_memory=True)
    return data_loader

class single_DRP_dataset(Dataset):
    def __init__(self, drp_idx, omics = 'ge', use_norm_ic50 = 'True'):
        super(single_DRP_dataset, self).__init__()

        if omics == 'ge':
            self.drug_atom, self.drug_bond_dict, self.ge_dict, self.DR = drug_atom_dict, drug_bond_dict, ge_HN_feat, drug_response_dict[drp_idx]
        elif omics == 'cnv':
            self.drug_atom, self.drug_bond_dict, self.ge_dict, self.DR = drug_atom_dict, drug_bond_dict, cnv_dict, drug_response_dict[drp_idx]
        elif omics == 'mut':
            self.drug_atom, self.drug_bond_dict, self.ge_dict, self.DR = drug_atom_dict, drug_bond_dict, mut_dict, drug_response_dict[drp_idx]
        self.use_norm_ic50 = use_norm_ic50
        self.length = len(self.DR)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        if self.use_norm_ic50 == 'True':
            cell, drug, _, ic = self.DR[index]
        else: cell, drug, ic, _ = self.DR[index]
        return (self.drug_atom[drug], self.drug_bond_dict[drug], self.ge_dict[cell],  ic)
def _collate_single_DRP(samples):
    drug_atom, drug_bond, ge, labels = map(list, zip(*samples))
    batched_drug_atom = Batch.from_data_list(drug_atom)
    batched_drug_bond = Batch.from_data_list(drug_bond)
    batch_ge = Batch.from_data_list(ge)
    return batched_drug_atom, batched_drug_bond, batch_ge, torch.tensor(labels)
def single_drp_loader(data_set,batch_size,shuffle = True, num_workers = 4):
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_single_DRP, num_workers=num_workers,pin_memory=True)
    return data_loader
if __name__ == '__main__':
    train_drug_id,test_drug_id = split_drug_set()