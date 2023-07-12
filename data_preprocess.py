from create_cell_feat import *
from create_drug_feat import *
from create_drp_dict import *
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

drug_response_dict, drug_name, cell_name = read_dr_dict()
ge_HN_feat, ge_sim_dict, cnv_dict, mut_dict = load_cell_feat()
drug_atom_dict,drug_bond_dict = load_drug_feat()
class DRP_dataset(Dataset):
    def __init__(self, drug_atom_dict,drug_bond_dict, ge_dict, ge_sim_dict, cnv_dict, mut_dict, drug_response_dict):
        super(DRP_dataset, self).__init__()
        self.drug_atom, self.drug_bond, self.ge_dict, self.ge_sim_dict, self.cnv_dict, self.mut_dict, self.DR = drug_atom_dict, drug_bond_dict, ge_dict, ge_sim_dict, cnv_dict, mut_dict, drug_response_dict
        self.length = len(self.DR)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        cell, drug, ic, ic_norm = drug_response_dict[index]
        return (self.drug_atom[drug], self.drug_bond[drug], self.ge_dict[cell], self.ge_sim_dict[cell], self.cnv_dict[cell], self.mut_dict[cell], ic, ic_norm)
def _collate(samples):
    drug_atom, drug_bond, ge, ge_sim, cnv, mut, labels, labels_norm = map(list, zip(*samples))
    batched_drug_atom = Batch.from_data_list(drug_atom)
    batched_drug_bond = Batch.from_data_list(drug_bond)
    batch_ge = Batch.from_data_list(ge)
    batch_ge_sim = Batch.from_data_list(ge_sim)
    batch_cnv = Batch.from_data_list(cnv)
    batch_mut = Batch.from_data_list(mut)
    return batched_drug_atom, batched_drug_bond, batch_ge, batch_ge_sim, batch_cnv,batch_mut, torch.tensor(labels), torch.tensor(labels_norm)
# def train_test_split(type = 'mix'): ## type: mix, cb, db
#     drug_response_dict, drug_name, cell_name = read_dr_dict()
#     ge_HN_feat, ge_sim_dict, cnv_dict, mut_dict = load_cell_feat()
#     drug_atom_dict,drug_bond_dict = load_drug_feat()
#     train_idx, test_idx = create_drp_set(type=type, drug_name = drug_name, cell_name = cell_name, drug_response_dict = drug_response_dict, seed = 0)
#     train_set = DRP_dataset(drug_atom_dict=drug_atom_dict, drug_bond_dict=drug_bond_dict, ge_dict=ge_HN_feat, ge_sim_dict= ge_sim_dict, cnv_dict= cnv_dict, mut_dict = mut_dict, drug_response_dict=drug_response_dict[train_idx])
#     test_set = DRP_dataset(drug_atom_dict=drug_atom_dict, drug_bond_dict=drug_bond_dict, ge_dict=ge_HN_feat, ge_sim_dict= ge_sim_dict, cnv_dict= cnv_dict, mut_dict = mut_dict, drug_response_dict=drug_response_dict[test_idx])
#     return train_set, test_set
def drp_loader(data_set,batch_size,shuffle = True, num_workers = 4):
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate, num_workers=num_workers)
    return data_loader
if __name__ == '__main__':
    drug_response_dict, drug_name, cell_name = read_dr_dict()
    ge_HN_feat, ge_sim_dict, cnv_dict, mut_dict = load_cell_feat()
    drug_atom_dict,drug_bond_dict = load_drug_feat()
    ## Splity train_test dataset
    train_idx, test_idx = create_drp_set(type='mix', drug_name = drug_name, cell_name = cell_name, drug_response_dict = drug_response_dict, seed = 0)
    train_set = DRP_dataset(drug_atom_dict=drug_atom_dict, drug_bond_dict=drug_bond_dict, ge_dict=ge_HN_feat, ge_sim_dict= ge_sim_dict, cnv_dict= cnv_dict, mut_dict = mut_dict, drug_response_dict=drug_response_dict[train_idx])
    test_set = DRP_dataset(drug_atom_dict=drug_atom_dict, drug_bond_dict=drug_bond_dict, ge_dict=ge_HN_feat, ge_sim_dict= ge_sim_dict, cnv_dict= cnv_dict, mut_dict = mut_dict, drug_response_dict=drug_response_dict[test_idx])
    print(len(test_set))
    train_set[0]
    print(train_set[0][0].x.shape)
    train_loader = drp_loader(train_set, batch_size= 1024, shuffle = True, num_workers= 4)
    test_loader = drp_loader(test_set, batch_size= 1024, shuffle = False, num_workers= 4)