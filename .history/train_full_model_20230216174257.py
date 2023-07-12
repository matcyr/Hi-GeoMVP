import argparse
from prepare_data.DRP_loader import *
from train_model.train_test_multi_omics_atten import *
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model.drp_model.DRP_nn import Multi_Omics_TransRegression
from model.drug_model.drug_encoder import drug_encoder
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size (default: 1024)')
    parser.add_argument('--embed_dim',type = int, default= 32 )
    parser.add_argument('--dropout_rate',type = float, default= 0.2 )
    parser.add_argument('--layer_num',type = int, default= 4 )
    parser.add_argument('--hidden_dim',type = int, default= 8 )
    parser.add_argument('--readout',type = str, default= 'max' )
    parser.add_argument('--train_type',type = str, default= 'mix' )
    parser.add_argument('--device',type = int, default= 0 )
    parser.add_argument('--num_workers',type = int, default= 4 )
    parser.add_argument('--epochs',type = int, default= 301 )
    parser.add_argument('--lrf',type = float, default= 0.1 )
    parser.add_argument('--use_norm_ic50',type = str, default= 'False')
    parser.add_argument('--use_attention',type = str, default= 'True')
    parser.add_argument('--use_fp',type = str, default= 'True')
    parser.add_argument('--use_cnn', type = str, default= 'True')
    return parser.parse_args()
if __name__ == '__main__':
    args = arg_parse()
    parameters = dict(
    lr=[0.001], ##0.001*args.use_norm_ic50 + 0.01*(1-args.use_norm_ic50)
    embed_dim= [128],
    batch_siz=[256],
    layer_num=[2],
    hidden_dim = [128],
    readout = ['mean','max','add'],
    use_fp = ['False']
    optimizer = [['adam', 'sgd']],
)
    drug_atom_dict,drug_bond_dict = load_drug_feat()
    train_idx, test_idx = create_drp_set(type= args.train_type, drug_name = drug_name, cell_name = cell_name, drug_response_dict = drug_response_dict, seed = 0)
    train_set = multi_DRP_dataset(drug_atom_dict=drug_atom_dict, drug_bond_dict =drug_bond_dict, ge_dict=ge_HN_feat, cnv_dict = cnv_dict, mut_dict = mut_dict, drug_response_dict=drug_response_dict[train_idx],use_norm_ic50= args.use_norm_ic50)
    test_set = multi_DRP_dataset(drug_atom_dict=drug_atom_dict, drug_bond_dict = drug_bond_dict, ge_dict=ge_HN_feat, cnv_dict = cnv_dict, mut_dict = mut_dict, drug_response_dict=drug_response_dict[test_idx],use_norm_ic50= args.use_norm_ic50)
    device  = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu") 
    # train_loader = multi_drp_loader(train_set,batch_size= parameters['batch_size'][0],shuffle = True, num_workers = args.num_workers)
    # test_loader = multi_drp_loader(test_set,batch_size= 1024,shuffle = True, num_workers = args.num_workers)
    # model_config = {'embed_dim': args.embed_dim, 'dropout_rate': 0.4,'hidden_dim' : args.hidden_dim,
    #                     'layer_num': args.layer_num, 'readout': args.readout, 'use_cnn' : True, 'use_fp' : args.use_fp, 'use_smiles' : 'False'}
    # drug_model = drug_encoder(model_config)
    # print(drug_model.graph_dim)
    train_model(args, Multi_Omics_TransRegression, train_set, test_set ,parameters, optimizer)