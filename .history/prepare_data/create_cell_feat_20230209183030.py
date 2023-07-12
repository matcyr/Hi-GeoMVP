import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
import os
import torch_geometric
os.chdir(".")
root = os.getcwd()

def normalize(data):
    # get the minimum and maximum values for each gene
    min_values, _ = torch.min(data, dim=0)
    max_values, _ = torch.max(data, dim=0)
    # subtract the minimum value and divide by the range
    normalized_data = (data - min_values) / (max_values - min_values)
    return normalized_data


def cosine_similarity(nodes):
    nodes = nodes.float()
    dot_product = torch.mm(nodes, nodes.t())
    norms = torch.norm(nodes, dim=1, keepdim=True)+1e-8
    similarities = dot_product / (norms * norms.t())
    return similarities

# Build graph with cosine similarity


def build_graph(nodes, keep_edge_prob = 0.01):
    similarities = cosine_similarity(nodes)
    # Threshold similarities to create adjacency matrix
    threshold = np.percentile(similarities, 100*(1 - keep_edge_prob))
    adjacency_matrix = (similarities >= threshold).float()
    # Create edge indices
    row, col = adjacency_matrix.nonzero().t()
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def pcc_matrix(df):
    n_genes = df.shape[1]
    df = df.astype(float)
    pcc_matrix = np.corrcoef(df, rowvar=False)
    p_value_matrix = np.zeros((n_genes, n_genes))
    for i in range(n_genes):
        for j in range(i, n_genes):
            p_value = pearsonr(df.iloc[:, i], df.iloc[:, j])[1]
            p_value_matrix[i, j] = p_value
            p_value_matrix[j, i] = p_value
    similarity_threshold = 0.5
    p_value_threshold = 0.05
    adj_matrix = np.zeros((n_genes, n_genes))
    adj_matrix[np.abs(pcc_matrix) >= similarity_threshold] = 1
    adj_matrix[p_value_matrix > p_value_threshold] = 0
    adj_matrix = (adj_matrix + adj_matrix.T) > 0
    return adj_matrix


def build_pcc_graph(df):
    adjacency_matrix = pcc_matrix(df)
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float)
    # Create edge indices
    row, col = adjacency_matrix.nonzero().t()
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def create_cell_data(norm = False): ## Min-max to gene expression
    root = os.getcwd()
    mut_path = root + '/Data/GDSC_data/Table_S3_GDSC_Mutation.csv'
    copy_num_path = root + '/Data/GDSC_data/Table_S4_GDSC_Copy_number_variation.csv'
    mutation = pd.read_csv(mut_path)
    copy_number = pd.read_csv(copy_num_path)
    GSP_interaction = pd.read_table(
        root+'/Data/GDSC_data/HumanNet-GSP.tsv', header=None)
    GSP_interaction.columns = ['start', 'end']
    annoted_gene_table = pd.read_csv(
        root+'/Data/GDSC_data/Census_allMon Jan 17 12_22_33 2022.csv')
    gene_exp_table = pd.read_csv(
        root+'/Data/GDSC_data/Table_S1_GDSC_Gene_expression.csv')
    related_gene = list(set(gene_exp_table.columns).intersection(
        set(annoted_gene_table['Gene Symbol'].values)))
    annoted_gene_table = annoted_gene_table.set_index('Gene Symbol')
    Tier_gene_table = annoted_gene_table.loc[related_gene, [
        'Entrez GeneId', 'Tier']]
    GSP_interaction['in_GDSC'] = GSP_interaction.apply(lambda x: (
        x['start'] in Tier_gene_table['Entrez GeneId'].values and x['end'] in Tier_gene_table['Entrez GeneId'].values), axis=1)
    new_GSP_interaction = GSP_interaction[GSP_interaction.in_GDSC].reset_index(
        drop=True)
    Tier_gene_table.reset_index(inplace=True)

    def convert_geneid_enid(enid):
        # Map enid to gene-gene interaction graph
        tmp_gene = Tier_gene_table[Tier_gene_table['Entrez GeneId']
                                   == enid].index[0]
        return tmp_gene
    new_GSP_interaction['start_id'] = new_GSP_interaction.apply(
        lambda x: (convert_geneid_enid(x['start'])), axis=1)
    new_GSP_interaction['end_id'] = new_GSP_interaction.apply(
        lambda x: (convert_geneid_enid(x['end'])), axis=1)
    edge_list = [(row[1].start_id, row[1].end_id)
                 for row in new_GSP_interaction.iterrows()]
    # create Undirected graph
    other_direction = [(row[1].end_id, row[1].start_id)
                       for row in new_GSP_interaction.iterrows()]
    edge_list = edge_list+other_direction
    edge_list = list(set(edge_list))
    human_go_edge_list = torch.tensor(np.array(edge_list), dtype=torch.long)
    human_go_edge_list = human_go_edge_list.transpose(0, 1)
    genes = list(Tier_gene_table['Gene Symbol'])
    gene_exp_table_sub = gene_exp_table[genes]
    if norm:
        scaler = MinMaxScaler()
        gene_exp_table_sub = pd.DataFrame(scaler.fit_transform(
            gene_exp_table_sub), columns=gene_exp_table_sub.columns, index=gene_exp_table_sub.index)
    x_feat_rna_seq = {}
    for cell_id in gene_exp_table.index:
        x_feat = []
        for i in Tier_gene_table['Gene Symbol'].values:
            gene_feat = gene_exp_table_sub.loc[cell_id, i]
            feat = [gene_feat]
            x_feat.append(feat)
        x_feat = np.asarray(x_feat)
        x_feat_rna_seq[cell_id] = torch.tensor(x_feat, dtype=torch.float)
    gene_exp_sim_edge_list = build_pcc_graph(gene_exp_table_sub)
    exp_edge_list = human_go_edge_list
    if torch_geometric.utils.contains_self_loops(exp_edge_list):
        exp_edge_list = torch_geometric.utils.remove_self_loops(exp_edge_list)[0]
    if torch_geometric.utils.contains_self_loops(gene_exp_sim_edge_list):
        gene_exp_sim_edge_list = torch_geometric.utils.remove_self_loops(gene_exp_sim_edge_list)[0]
    print(f'Gene graph is undirected graph: {torch_geometric.utils.is_undirected(exp_edge_list)}')
    print(f'Gene graph has self loop: {torch_geometric.utils.contains_self_loops(exp_edge_list)}')
    print(f'Gene sim graph is undirected graph: {torch_geometric.utils.is_undirected(gene_exp_sim_edge_list)}')
    print(f'Gene sim graph has self loop: {torch_geometric.utils.contains_self_loops(gene_exp_sim_edge_list)}')


    x_feat_CNV = {}
    for cell_id in copy_number.index:
        x_feat = []
        # We give none to the gene which is not in the mutation dataframe.
        for gene in copy_number.columns:
            CNV = copy_number.loc[cell_id, gene]
            feat = CNV
            x_feat.append(feat)
        x_feat = np.asarray(x_feat)
        x_feat_CNV[cell_id] = torch.tensor(x_feat, dtype=torch.long)
    cnv_edge_index = build_graph(torch.tensor(copy_number.values.T))
    if torch_geometric.utils.contains_self_loops(cnv_edge_index):
        cnv_edge_index = torch_geometric.utils.remove_self_loops(cnv_edge_index)[0]
    print(f'CNV graph is undirected graph: {torch_geometric.utils.is_undirected(cnv_edge_index)}')
    print(f'CNV graph has self loop: {torch_geometric.utils.contains_self_loops(cnv_edge_index)}')

    x_feat_mut = {}
    for cell_id in mutation.index:
        x_feat = []
        # We give none to the gene which is not in the mutation dataframe.
        for gene in mutation.columns:
            mut = mutation.loc[cell_id, gene]
            feat = mut
            x_feat.append(feat)
        x_feat = np.asarray(x_feat)
        x_feat_mut[cell_id] = torch.tensor(x_feat, dtype=torch.long)
    mut_edge_list = build_graph(torch.tensor(mutation.values.T))
    if torch_geometric.utils.contains_self_loops(mut_edge_list):
        mut_edge_list = torch_geometric.utils.remove_self_loops(mut_edge_list)[0]
    print(f'MUT graph is undirected graph: {torch_geometric.utils.is_undirected(mut_edge_list)}')
    print(f'MUT graph has self loop: {torch_geometric.utils.contains_self_loops(mut_edge_list)}')
    # Gene expression HumanNet
    Gene_HumanNet_feature = {}
    for cell in x_feat_rna_seq.keys():
        Gene_HumanNet_feature[cell] = Data(
            x=x_feat_rna_seq[cell], edge_index=exp_edge_list, cell_id=cell)
    # Gene expression similarity
    Gene_Sim_Feature = {}
    for cell in x_feat_rna_seq.keys():
        Gene_Sim_Feature[cell] = Data(
            x=x_feat_rna_seq[cell], edge_index=gene_exp_sim_edge_list, cell_id=cell)
    # CNV
    CNV_feature = {}
    for cell in x_feat_CNV.keys():
        CNV_feature[cell] = Data(
            x=x_feat_CNV[cell], edge_index=cnv_edge_index, cell_id=cell)
    # Mutation
    Mutation_feature = {}
    for cell in x_feat_mut.keys():
        Mutation_feature[cell] = Data(
            x=x_feat_mut[cell], edge_index=mut_edge_list, cell_id=cell)

    def save_cell_feat():
        save_path = root + '/Data/DRP_dataset'
        np.save(os.path.join(save_path, 'cell_feature_ge_HN.npy'),
                Gene_HumanNet_feature)
        np.save(os.path.join(save_path, 'cell_feature_ge_sim.npy'),
                Gene_Sim_Feature)
        np.save(os.path.join(save_path, 'cell_feature_cnv.npy'), CNV_feature)
        np.save(os.path.join(save_path, 'cell_feature_mut.npy'), Mutation_feature)
        print("finish saving cell data!")
    save_cell_feat()
    return Gene_HumanNet_feature, Gene_Sim_Feature, CNV_feature, Mutation_feature


def load_cell_feat():
    ge_HN_feat = np.load(
        root + '/Data/DRP_dataset/cell_feature_ge_HN.npy', allow_pickle=True).item()
    ge_sim_dict = np.load(
        root + '/Data/DRP_dataset/cell_feature_ge_sim.npy', allow_pickle=True).item()
    cnv_dict = np.load(
        root + '/Data/DRP_dataset/cell_feature_cnv.npy', allow_pickle=True).item()
    mut_dict = np.load(
        './Data/DRP_dataset/cell_feature_mut.npy', allow_pickle=True).item()
    return ge_HN_feat, ge_sim_dict, cnv_dict, mut_dict


if __name__ == '__main__':
    if os.path.exists(f"./Data/DRP_dataset") is False:
        os.makedirs(f"./Data/DRP_dataset")
    ge_HN_feat, ge_sim_dict, cnv_dict, mut_dict = create_cell_data()
    ge_HN_feat, ge_sim_dict, cnv_dict, mut_dict = load_cell_feat()
    print('finish loading cell data!')
