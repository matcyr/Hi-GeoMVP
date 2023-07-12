from prepare_data.create_drp_dict import create_drp_set, read_dr_dict
from prepare_data.DRP_loader import single_DRP_dataset, single_drp_loader

if __name__ == '__main__':
    drug_response_dict, drug_name, cell_name = read_dr_dict()
    train_idx, test_idx = create_drp_set(type= 'cb', drug_name = drug_name, cell_name = cell_name, drug_response_dict = drug_response_dict, seed = 0)
    print(min(train_idx),min(test_idx))
    train_set = single_DRP_dataset(drp_idx = train_idx,use_norm_ic50= False)
    test_set = single_DRP_dataset(drp_idx = test_idx,use_norm_ic50= True)
    print(len(test_set))
    print(test_set[0])