import os
import torch
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from tqdm import tqdm
import os
os.chdir(".")
root = os.getcwd()

# Adapted from https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py
allowable_atom_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True]}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(
            allowable_atom_features['possible_atomic_num_list'], atom.GetAtomicNum()),
        safe_index(allowable_atom_features['possible_chirality_list'], str(
            atom.GetChiralTag())),
        safe_index(
            allowable_atom_features['possible_degree_list'], atom.GetTotalDegree()),
        safe_index(
            allowable_atom_features['possible_formal_charge_list'], atom.GetFormalCharge()),
        safe_index(
            allowable_atom_features['possible_numH_list'], atom.GetTotalNumHs()),
        safe_index(
            allowable_atom_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
        safe_index(allowable_atom_features['possible_hybridization_list'], str(
            atom.GetHybridization())),
        allowable_atom_features['possible_is_aromatic_list'].index(
            atom.GetIsAromatic()),
        allowable_atom_features['possible_is_in_ring_list'].index(
            atom.IsInRing()),
        atom.GetMass()
    ]
    return atom_feature


def atom_feature_vector_to_dict(atom_feature):
    [atomic_num_idx,
     chirality_idx,
     degree_idx,
     formal_charge_idx,
     num_h_idx,
     number_radical_e_idx,
     hybridization_idx,
     is_aromatic_idx,
     is_in_ring_idx, weight] = atom_feature

    feature_dict = {
        'atomic_num': allowable_atom_features['possible_atomic_num_list'][atomic_num_idx],
        'chirality': allowable_atom_features['possible_chirality_list'][chirality_idx],
        'degree': allowable_atom_features['possible_degree_list'][degree_idx],
        'formal_charge': allowable_atom_features['possible_formal_charge_list'][formal_charge_idx],
        'num_h': allowable_atom_features['possible_numH_list'][num_h_idx],
        'num_rad_e': allowable_atom_features['possible_number_radical_e_list'][number_radical_e_idx],
        'hybridization': allowable_atom_features['possible_hybridization_list'][hybridization_idx],
        'is_aromatic': allowable_atom_features['possible_is_aromatic_list'][is_aromatic_idx],
        'is_in_ring': allowable_atom_features['possible_is_in_ring_list'][is_in_ring_idx],
        'weight': weight
    }

    return feature_dict


def get_atom_int_feature_dims():
    return list(map(len, [
        allowable_atom_features['possible_atomic_num_list'],
        allowable_atom_features['possible_chirality_list'],
        allowable_atom_features['possible_degree_list'],
        allowable_atom_features['possible_formal_charge_list'],
        allowable_atom_features['possible_numH_list'],
        allowable_atom_features['possible_number_radical_e_list'],
        allowable_atom_features['possible_hybridization_list'],
        allowable_atom_features['possible_is_aromatic_list'],
        allowable_atom_features['possible_is_in_ring_list']
    ]))


allowable_bond_features = {
    'possible_bond_type_list': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(allowable_bond_features['possible_bond_type_list'], str(
            bond.GetBondType())),
        allowable_bond_features['possible_bond_stereo_list'].index(
            str(bond.GetStereo())),
        allowable_bond_features['possible_is_conjugated_list'].index(
            bond.GetIsConjugated()),
    ]
    return bond_feature


def get_bond_feature_int_dims():
    return list(map(len, [
        allowable_bond_features['possible_bond_type_list'],
        allowable_bond_features['possible_bond_stereo_list'],
        allowable_bond_features['possible_is_conjugated_list']
    ]))


def bond_feature_vector_to_dict(bond_feature):
    [bond_type_idx,
     bond_stereo_idx,
     is_conjugated_idx] = bond_feature

    feature_dict = {
        'bond_type': allowable_bond_features['possible_bond_type_list'][bond_type_idx],
        'bond_stereo': allowable_bond_features['possible_bond_stereo_list'][bond_stereo_idx],
        'is_conjugated': allowable_bond_features['possible_is_conjugated_list'][is_conjugated_idx]
    }

    return feature_dict


def self_loop_bond_feature():
    bond_feat = [len(allowable_bond_features[key]) +
                 2 for key in allowable_bond_features]  # N + 2 for self-loop
    # Length of the bond
    bond_feat += [0.0]
    return bond_feat

def mol_to_3d_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    def _get_atom_poses(mol):
        # Get atom coordiniate in 3d
        atom_poses = []
        try:
            new_mol = Chem.AddHs(mol)
            cids = AllChem.EmbedMultipleConfs(new_mol, numConfs=10)
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            index = np.argmin([x[1] for x in res])
            new_mol = Chem.RemoveHs(new_mol)
            conf = new_mol.GetConformer(id=int(index))
        except:
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            new_mol = mol
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())

            # print(atom.GetIdx())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return new_mol, np.array(atom_poses, 'float32')
# 1. Atom: One hot vectors.

    def _get_node_features(mol):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []

    ####
    #   - Atom type: A one-hot vector of this atom, "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other atoms".
    #   - Formal charge: Integer electronic charge.
    #   - Hybridization: A one-hot vector of "sp", "sp2", "sp3".
    #   - Hydrogen bonding: A one-hot vector of whether this atom is a hydrogen bond donor or acceptor.
    #   - Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
    #   - Degree: A one-hot vector of the degree (0-5) of this atom.
    #   - Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.
    #   - Ring : A one-hot vector of whether the atom belongs to a ring.
    #   - Mass : The mass of this atom.
        for atom in mol.GetAtoms():
            node_feats = atom_to_feature_vector(atom)
            # Append node features to matrix
            all_node_feats.append(node_feats)
        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _cal_bond_length(mol, atom_pos):
        bond_length = []
        for i, bond in enumerate(mol.GetBonds()):
            startid = bond.GetBeginAtomIdx()
            endid = bond.GetEndAtomIdx()
            b_l = np.linalg.norm(atom_pos[startid] - atom_pos[endid])
            bond_length.append(b_l)
        bond_length = np.array(bond_length, 'float32')
        return bond_length

    def _get_edge_features(mol, bond_length):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []
    ####
    #   - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic". #4
    #   - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.  #1
    #   - Conjugated: A one-hot vector of whether this bond is conjugated or not. #1
    #   - Stereo: A one-hot vector of the stereo configuration of a bond. #5
    #   - Bond length #1
        for bond in mol.GetBonds():
            edge_feats = bond_to_feature_vector(bond)
            # Feature float: bond length
            length = bond_length[bond.GetIdx()]
            # Append node features to matrix (twice, per direction)
            edge_feats.append(length)
            all_edge_feats += [edge_feats, edge_feats]
        N_atom = mol.GetNumAtoms()
        for i in range(N_atom):
            edge_feats = self_loop_bond_feature()
            all_edge_feats += [edge_feats]
        all_edge_feats = np.asarray(all_edge_feats, dtype=float)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(mol):
        edge_indices = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[start, end], [end, start]]
        N = mol.GetNumAtoms()
        for i in range(N):
            edge_indices += [(i, i)]
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices
    # Build bond-bond graph. Each bond is a node, each angle is edge.

    def _get_ba_adjacency_info(edge_indices, atom_pos, edge_attr_atom):
        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle
        edge_indices_pair = edge_indices.t()
        super_edge_indices = []
        bond_angle = []
        x_bond = []
        E = len(edge_indices_pair)
        edge_indices = np.arange(E)
        for bondidx, bond in enumerate(edge_indices_pair):
            x_bond += [edge_attr_atom[bondidx]]
            src_edge_indices = edge_indices[edge_indices_pair[:, 1] == bond[0]]
            for src_edge_i in src_edge_indices:
                if src_edge_i == bondidx:
                    continue
                src_edge = edge_indices_pair[src_edge_i]
                super_edge_indices += [[src_edge_i, bondidx]]
                vec1 = atom_pos[bond[0]]-atom_pos[bond[1]]
                vec2 = atom_pos[src_edge[0]]-atom_pos[src_edge[1]]
                angle = _get_angle(vec1, vec2)
                bond_angle += [angle]
        super_edge_indices = torch.tensor(super_edge_indices)
        super_edge_indices = super_edge_indices.t().to(torch.long).view(2, -1)
        bond_angle = np.asarray(bond_angle, 'float32')
        x_bond = torch.stack(x_bond)
        bond_angle = torch.tensor(bond_angle, dtype=torch.float)
        bond_angle = bond_angle.reshape(bond_angle.shape[0], 1)
        return super_edge_indices, bond_angle, x_bond
    new_mol, atom_poses = _get_atom_poses(mol)  # MMFF optimization
    x_atom = _get_node_features(new_mol)
    bond_length = _cal_bond_length(new_mol, atom_poses)
    edge_attr_atom = _get_edge_features(mol, bond_length)
    edge_index_atom = _get_adjacency_info(new_mol)
    edge_index_bond, edge_attr_bond, x_bond = _get_ba_adjacency_info(
        edge_index_atom, atom_poses, edge_attr_atom)
    drug_atom = Data(x = x_atom, edge_index = edge_index_atom, edge_attr = edge_attr_atom)
    drug_bond = Data(x = x_bond, edge_index = edge_index_bond, edge_attr = edge_attr_bond)
    drug_atom.smiles = smiles
    drug_bond.smiles = smiles
    return drug_atom,drug_bond


def create_drug_dict():
    raw_paths = './Data/GDSC_data/drugid_smiles.txt'
    dataset = pd.read_csv(raw_paths, delimiter="\t",
                        header=None).reset_index(drop=True)
    dataset.rename(columns={0: 'drugid', 1: 'SMILES'}, inplace=True)
    dataset = dataset.set_index('drugid')
    SMILES_CHARS = []
    for item in dataset['SMILES']:
        string = [x for x in item]
        SMILES_CHARS = SMILES_CHARS+string
    SMILES_CHARS = set(SMILES_CHARS)
    smi2index = dict((c, i) for i, c in enumerate(SMILES_CHARS))
    def smiles_encoder(smiles, maxlen=160):
        X = np.zeros((len(SMILES_CHARS), maxlen))
        for i, c in enumerate(smiles):
            X[smi2index[c], i] = 1
        return X
    maxlen = 179
    def fp_as_array(smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3)
        arr = np.zeros((1,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr  
    
    drug_atom_dict = {}
    drug_bond_dict = {}
    for row in tqdm(dataset.iterrows()):
        drug_id = row[0]
        smiles = row[1].item()
        drug_atom,drug_bond = mol_to_3d_from_smiles(smiles)
        fp = fp_as_array(smiles).reshape(1, -1)
        one_hot_smiles = torch.tensor(smiles_encoder(smiles, maxlen))
        drug_atom.drug_id = drug_id
        drug_bond.drug_id = drug_id
        drug_atom.fp = torch.tensor(fp)
        drug_atom.one_hot_smiles = one_hot_smiles
        drug_atom_dict[drug_id] = drug_atom
        drug_bond_dict[drug_id] = drug_bond
    print('Finish generating drugs')
### drug 174:RDKit ERROR: [22:42:02] UFFTYPER: Unrecognized charge state for atom: 54
    drug_id_name = pd.read_csv('./Data/GDSC_data/drugid_name.txt',sep='\t',header=None)
    drug_id_name_dict = {item[1]:item[0] for item in drug_id_name.values} ##'BMS-536924': 10390396
    drug_name_id_dict = {item[0]:item[1] for item in drug_id_name.values}
    target_drug = pd.read_csv('./Data/GDSC_data/Table_S12_GDSC_Drug_group.csv')
    target_drug = target_drug.set_index('TARGET_PATHWAY').DRUG_NAME.str.split(';', expand=True).stack().reset_index(level=1, drop=True).reset_index()
    target_drug = target_drug.pivot(index= 0 , columns='TARGET_PATHWAY', values=0).fillna(0)
    target_drug.rename(index = drug_id_name_dict, inplace = True)
    target_drug = target_drug[target_drug.index.isin(drug_id_name_dict.values())]
    drug_target_pathway = {}
    for column in target_drug.columns:
        target_drug[column] = pd.to_numeric(target_drug[column], errors='coerce').fillna(1).astype(int)
    for index in target_drug.index:
        drug_target_pathway[index] = torch.tensor(target_drug.loc[index].values).view(-1,1)
    
    pathway = list(target_drug.columns)
    def save_drug_feat():
        save_path = root + '/Data/DRP_dataset'
        np.save(os.path.join(save_path, 'drug_feat_atom.npy'), drug_atom_dict)
        np.save(os.path.join(save_path, 'drug_feat_bond.npy'), drug_bond_dict)
        np.save(os.path.join(save_path, 'drug_target_pathway.npy'), drug_target_pathway)
        np.save(os.path.join(save_path, 'pathway.npy'), pathway)
        print("finish saving drug data!")
    save_drug_feat()
    return drug_atom_dict, drug_bond_dict, target_drug, pathway
def load_drug_feat():
    drug_atom_feat = np.load(root + '/Data/DRP_dataset/drug_feat_atom.npy',allow_pickle=True).item()
    drug_bond_feat = np.load(root + '/Data/DRP_dataset/drug_feat_bond.npy',allow_pickle=True).item()
    drug_target_pathway = np.load('./Data/DRP_dataset/drug_target_pathway.npy',allow_pickle=True).item()
    pathway = np.load('./Data/DRP_dataset/pathway.npy')
    return drug_atom_feat, drug_bond_feat, drug_target_pathway, pathway
if __name__ =='__main__':
    root = os.getcwd()
    if os.path.exists(root + f"/Data/DRP_dataset") is False:
        os.makedirs(root + "./Data/DRP_dataset")
    drug_atom_dict,drug_bond_dict, drug_target_pathway, pathway = create_drug_dict()
    drug_atom_dict,drug_bond_dict, drug_target_pathway, pathway = load_drug_feat()
    print('finish loading drug data!')