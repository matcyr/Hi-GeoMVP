import sys
sys.path.append('prepare_data')
from create_cell_feat import *
from create_drug_feat import *
from create_drp_dict import *
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
drug_response_dict, drug_name, cell_name = read_dr_dict()
ge_HN_feat, ge_sim_dict, cnv_dict, mut_dict = load_cell_feat()
drug_atom_dict,drug_bond_dict, target_drug, pathway = load_drug_feat()
train_idx,test_idx = create_drp_set(type= 'mix', drug_name = drug_name, cell_name = cell_name, drug_response_dict = drug_response_dict, seed = 0)
class multi_DRP_dataset(Dataset):
    def __init__(self, drp_idx ,use_norm_ic50 = 'True'):
        super(multi_DRP_dataset, self).__init__()
        self.drug_atom, self.drug_bond_dict, self.ge_dict, self.cnv_dict, self.mut_dict, self.DR = drug_atom_dict, drug_bond_dict, ge_HN_feat, cnv_dict, mut_dict, drug_response_dict[drp_idx]
        self.use_norm_ic50 = use_norm_ic50
        self.length = len(self.DR)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        if self.use_norm_ic50 == 'True':
            cell, drug, _, ic = self.DR[index]
        else: cell, drug, ic, _ = self.DR[index]
        return (self.drug_atom[drug], self.drug_bond_dict[drug], self.ge_dict[cell], self.cnv_dict[cell], self.mut_dict[cell], ic)
def _collate_multi_omics(samples):
    drug_atom, drug_bond, ge, cnv, mut, labels = map(list, zip(*samples))
    batched_drug_atom = Batch.from_data_list(drug_atom)
    batched_drug_bond = Batch.from_data_list(drug_bond)
    batch_ge = Batch.from_data_list(ge)
    batch_cnv = Batch.from_data_list(cnv)
    batch_mut = Batch.from_data_list(mut)
    return batched_drug_atom, batched_drug_bond, batch_ge, batch_cnv, batch_mut, torch.tensor(labels)
def multi_drp_loader(data_set,batch_size,shuffle = True, num_workers = 4):
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_multi_omics, num_workers=num_workers,pin_memory=True)
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