import argparse
from prepare_data.DRP_loader import *
from train_model.train_test_GeoMVP_5_fold import *
from torch_geometric.data import Batch
from torch_geometric.nn import  graclus, max_pool
import argparse
from prepare_data.create_cell_feat import *
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size (default: 512)')
    parser.add_argument('--embed_dim',type = int, default= 256 )
    parser.add_argument('--dropout_rate',type = float, default= 0.4 )
    parser.add_argument('--layer_num',type = int, default= 2 )
    parser.add_argument('--hidden_dim',type = int, default= 128 )
    parser.add_argument('--readout',type = str, default= 'mean' )
    parser.add_argument('--train_type',type = str, default= 'mix' )
    parser.add_argument('--device',type = int, default= 0)
    parser.add_argument('--seed',type = int, default= 1234)
    parser.add_argument('--num_workers',type = int, default= 4 )
    parser.add_argument('--epochs',type = int, default= 301 )
    parser.add_argument('--lrf',type = float, default= 0.1 )
    parser.add_argument('--lr',type = float, default= 0.0001 )
    parser.add_argument('--use_norm_ic50',type = str, default= 'False')
    parser.add_argument('--use_regulizer',type = str, default= 'True')
    parser.add_argument('--use_regulizer_drug',type = str, default= 'True')
    parser.add_argument('--use_drug_path_way',type = str, default= 'True')
    parser.add_argument('--use_cnn',type = str, default= 'False')
    parser.add_argument('--use_raw_gene',type = str, default= 'True')
    parser.add_argument('--use_predined_gene_cluster',type = str, default= 'False')
    parser.add_argument('--regular_weight', type = float, default= 0.5)
    parser.add_argument('--regular_weight_drug', type = float, default= 0.5)
    parser.add_argument('--regular_weight_drug_path_way', type = float, default= 0.5)
    parser.add_argument('--check_step', type = int, default= 2)
    parser.add_argument('--early_stop_count', type = int, default= 6)
    parser.add_argument('--view_dim', type= int, default=256)
    parser.add_argument('--part', type= int, default=0) ##cross validation part 0,1,2,3,4
    parser.add_argument('--scheduler_type',type = str, default= 'OP', help= 'OP(OnPlateau) or ML(Multistep)')
    parser.add_argument('--drug_ablation', type= str, default= 'False')
    return parser.parse_args()

ge_HN_feat, ge_sim_dict, cnv_dict, mut_dict = load_cell_feat()
GE_data = load_raw_GE()

ge = ge_HN_feat[906826]
mut = mut_dict[906826]
cnv = cnv_dict[906826]
cluster_predefine = {}
def get_cluster(g):
    cluster_predefine = {}
    g = Batch.from_data_list([g])
    for i in range(5):
        cluster = graclus(edge_index = g.edge_index, num_nodes = g.x.size(0))
        # print(len(cluster.unique()))
        g = max_pool(cluster, g, transform=None)
        cluster_predefine[i] = cluster
    return cluster_predefine
mut_cluster = get_cluster(mut)
cnv_cluster = get_cluster(cnv)
ge_cluster = get_cluster(ge)

def cluster_to_device(cluster_predefine, device):
    cluster_predefine = {i: j.to(device) for i, j in cluster_predefine.items()}
    return cluster_predefine

ge_pathway_cluster, mut_pathway_cluster, cnv_pathway_cluster = load_gene_cluster()

if __name__ == '__main__':
    args = arg_parse()
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu") 
    # device = torch.device('cpu')
    mut_cluster, cnv_cluster, ge_cluster = cluster_to_device(mut_cluster, device), cluster_to_device(cnv_cluster, device), cluster_to_device(ge_cluster, device)
    drug_response_dict, drug_name, cell_name = read_dr_dict()
    ge_HN_feat, ge_sim_dict, cnv_dict, mut_dict = load_cell_feat()
    drug_atom_dict,drug_bond_dict = load_drug_feat()
    ## 5 fold train_test dataset
    train_dict, val_dict, test_dict = n_fold_split(drug_name=drug_name, cell_name= cell_name, drug_response_dict= drug_response_dict, type= args.train_type, seed = args.seed)
    i = args.part
    train_idx, val_idx, test_idx = train_dict[i], val_dict[i], test_dict
    train_set = multi_DRP_dataset(drp_idx = train_idx,use_norm_ic50= args.use_norm_ic50, use_raw_gene= args.use_raw_gene)
    val_set = multi_DRP_dataset(drp_idx = val_idx,use_norm_ic50= args.use_norm_ic50, use_raw_gene= args.use_raw_gene)
    test_set = multi_DRP_dataset(drp_idx = test_idx ,use_norm_ic50= args.use_norm_ic50, use_raw_gene= args.use_raw_gene)
    if args.use_predined_gene_cluster == 'True':
        ge_pathway_cluster, mut_pathway_cluster, cnv_pathway_cluster = ge_pathway_cluster.to(device), mut_pathway_cluster.to(device), cnv_pathway_cluster.to(device)
        train_multi_view_model(args, train_set, val_set, test_set, mut_pathway_cluster, cnv_pathway_cluster, ge_pathway_cluster, i)
    else:
        train_multi_view_model(args, train_set, val_set, test_set, mut_cluster, cnv_cluster, ge_cluster, i)