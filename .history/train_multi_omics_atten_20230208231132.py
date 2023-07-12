import argparse
import torch
import torch.optim as opt
from torch_geometric.data import Batch
from scipy.stats import pearsonr
from create_cell_feat import *
from create_drug_feat import *
from create_drp_dict import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from itertools import product
from model.drp_model.DRP_nn import Multi_Omics_TransRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

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
    parser.add_argument('--use_attention',type = bool, default= False)
    return parser.parse_args()

class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, path,patience= 15, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train(model, train_loader, optimizer, loss_op, writer, epoch,device):
    model.train()
    total_loss = []
    optimizer.zero_grad()
    for data in tqdm(train_loader):
        drug_atom,drug_bond,cell_ge,cell_cnv,cell_mut, ic50 = data
        drug_atom = drug_atom.to(device)  
        drug_bond = drug_bond.to(device)
        cell_ge = cell_ge.to(device)
        # ge_sim = ge_sim.to(device)
        ic50 = ic50.to(device)  
        cell_cnv = cell_cnv.to(device)
        cell_mut = cell_mut.to(device)
        # with torch.cuda.amp.autocast():
        loss = loss_op(model(drug_atom,drug_bond,cell_ge,cell_cnv,cell_mut), ic50.view(-1, 1).float())
        total_loss.append(loss.item())
        # scalr.scale(loss).backward()
        # scalr.step(optimizer)
        # scalr.update()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    train_loss = np.average(total_loss)
    writer.add_scalar("Loss", train_loss, epoch)
    return train_loss

@torch.no_grad()
def test(model,loader,device):
    model.eval()
    y_true, preds = [], []
    for data in tqdm(loader):
        drug_atom,drug_bond,cell_ge,cell_cnv,cell_mut, ic50 = data
        drug_atom,drug_bond,cell_ge,cell_cnv,cell_mut, ic50 = data = drug_atom.to(device), drug_bond.to(device), cell_ge.to(device), cell_cnv.to(device), cell_mut.to(device), ic50.to(device)
        y_true.append(ic50.view(-1, 1).float())
        out = model(drug_atom,drug_bond,cell_ge,cell_cnv,cell_mut)
        preds.append(out.float().cpu())
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(preds, dim=0).numpy()
    rmse = mean_squared_error(y_true,y_pred, squared=False)
    r = pearsonr(y_true.flatten(), y_pred.flatten())[0]
    return rmse,r

class multi_DRP_dataset(Dataset):
    def __init__(self, drug_atom_dict, drug_bond_dict, ge_dict, cnv_dict, mut_dict, drug_response_dict, use_norm_ic50 = 'True'):
        super(multi_DRP_dataset, self).__init__()
        self.drug_atom, self.drug_bond_dict, self.ge_dict, self.cnv_dict, self.mut_dict, self.DR = drug_atom_dict, drug_bond_dict, ge_dict, cnv_dict, mut_dict, drug_response_dict
        self.use_norm_ic50 = use_norm_ic50
        self.length = len(self.DR)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        if self.use_norm_ic50 == 'True':
            cell, drug, _, ic = self.DR[index]
        else: cell, drug, ic, _ = self.DR[index]
        return (self.drug_atom[drug], self.drug_bond_dict[drug], self.ge_dict[cell], self.cnv_dict[cell], self.mut_dict[cell], ic)
def _collate(samples):
    drug_atom, drug_bond, ge, cnv, mut, labels = map(list, zip(*samples))
    batched_drug_atom = Batch.from_data_list(drug_atom)
    batched_drug_bond = Batch.from_data_list(drug_bond)
    batch_ge = Batch.from_data_list(ge)
    batch_cnv = Batch.from_data_list(cnv)
    batch_mut = Batch.from_data_list(mut)
    return batched_drug_atom, batched_drug_bond, batch_ge, batch_cnv, batch_mut, torch.tensor(labels)
def multi_drp_loader(data_set,batch_size,shuffle = True, num_workers = 4):
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate, num_workers=num_workers,pin_memory=True)
    return data_loader



if __name__ == '__main__':
    args = arg_parse()
    parameters = dict(
    lr=[0.001], ##0.001*args.use_norm_ic50 + 0.01*(1-args.use_norm_ic50)
    embed_dim= [64],
    batch_size=[1024],
    layer_num=[4],
    hidden_dim = [64],
    readout = ['mean'],
)
    drug_response_dict, drug_name, cell_name = read_dr_dict()
    ge_HN_feat, ge_sim_dict, cnv_dict, mut_dict = load_cell_feat()
    drug_atom_dict,drug_bond_dict = load_drug_feat()
    ## Splity train_test dataset
    train_idx, test_idx = create_drp_set(type= args.train_type, drug_name = drug_name, cell_name = cell_name, drug_response_dict = drug_response_dict, seed = 0)
    train_set = multi_DRP_dataset(drug_atom_dict=drug_atom_dict, drug_bond_dict =drug_bond_dict, ge_dict=ge_HN_feat, cnv_dict = cnv_dict, mut_dict = mut_dict, drug_response_dict=drug_response_dict[train_idx],use_norm_ic50= args.use_norm_ic50)
    test_set = multi_DRP_dataset(drug_atom_dict=drug_atom_dict, drug_bond_dict = drug_bond_dict, ge_dict=ge_HN_feat, cnv_dict = cnv_dict, mut_dict = mut_dict, drug_response_dict=drug_response_dict[test_idx],use_norm_ic50= args.use_norm_ic50)
    device  = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu") 
    if args.use_norm_ic50 == 'True':
        weights = 'weights_atten_norm_multi_omics'
    else: weights = 'weights_atten_multi_omics'
    if os.path.exists(f"./{weights}/{args.train_type}") is False:
        os.makedirs(f"./{weights}/{args.train_type}")
    print(f'Save model parameter under ./{weights}/{args.train_type}')
    test_loader = multi_drp_loader(test_set, batch_size= 256, shuffle = False, num_workers= args.num_workers)
    save_dir = 'tensorboard_atten_model_multi_omics'+ args.train_type
    print(save_dir)
    # if args.drug_encoder == 'Drug_Encoder':
    def train_model(parameters,model_sturcture):
        param_values = [v for v in parameters.values()]
        for run_id, (lr, embed_dim, batch_size, layer_num,hidden_dim,readout) in enumerate(product(*param_values)):      
            path = f'./Drug_response/{save_dir}/{run_id}'+'.pth'    
            early_stopping = EarlyStopping(path = path,patience= 3 , verbose=True)
            print(path)
            print("run id:", run_id + 1)
            model_config = {'embed_dim': embed_dim, 'dropout_rate': 0.4,'hidden_dim' : hidden_dim,
                            'layer_num': layer_num, 'readout': readout, 'use_cnn' : True, 'use_fp' : False, 'use_smiles' : False}
            model = model_sturcture(model_config).to(device)
            optimizer = opt.Adam(model.parameters(), lr=lr)
            # cos_lr = lambda x : ((1+math.cos(math.pi* x /args.epochs) )/2)*(1-args.lrf) + args.lrf
            # scheduler = opt.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda = cos_lr)
            criterion = torch.nn.MSELoss()
            comment = f' batch_size = {batch_size} lr = {lr} embed_dim = {embed_dim} layer_num = {layer_num} readout = {readout} hidden_dim = {hidden_dim}'
            print('Begin Training')
            print(f'Embed_dim_drug : {embed_dim}'+ '\n' +f'Hidden_dim_cell : {hidden_dim} \n' +  f'layer_num : {layer_num} \n'+ f'read_out_function : {readout}')
            tb = SummaryWriter(comment=comment, log_dir=f'./Drug_response/{save_dir}/')
            train_loader = multi_drp_loader(train_set,batch_size= batch_size,shuffle = True, num_workers = args.num_workers)
            n_epochs = args.epochs
            epoch_len = len(str(n_epochs))
            best_val_pcc = -1
            for epoch in range(n_epochs):
                train_rmse = train(model, train_loader, optimizer, criterion, tb, epoch, device)
                # scheduler.step()
                print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                            f'train_loss: {train_rmse:.5f} ')
                torch.save(model.state_dict(), f"./{weights}/{args.train_type}"+"/model_run_{}".format(run_id)+"_epoch_{}.pth".format(epoch))
                print(print_msg)
                # print(f'Epoch: {epoch:03d}, Loss: {train_rmse:.4f}')
                if epoch % 10 == 0:
                    val_rmse,val_pcc = test(model,test_loader, device)
                    if val_pcc > best_val_pcc:
                        best_val_pcc = val_pcc
                        best_val_rmse = val_rmse
                        best_epoch = epoch
                    train_rmse,train_pcc = test(model,train_loader, device)
                    print(f'Epoch: {epoch:03d}, Loss: {train_rmse:.4f}, val_rmse:{val_rmse:.4f}, Val_PCC: {val_pcc:.4f}, train_PCC: {train_pcc:.4f} ')
                    print(f'Best epoch: {best_epoch:03d}, Best PCC: {best_val_pcc:.4f}, Best RMSE: {best_val_rmse:.4f} ')
                    early_stopping(val_rmse, model)                 
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break  
                print("batch_size:", batch_size, "lr:", lr,
                    "embed_dim:", embed_dim, "layer_num:", layer_num)
            print("__________________________________________________________")
            if args.use_norm_ic50 == 'True':
                results = 'results_atten_norm_multi_omics'
            else: results = 'results_atten_multi_omics'
            if os.path.exists(f"./{results}/{args.train_type}") is False:
                os.makedirs(f"./{results}/{args.train_type}")
            reult_file_path = f"./{results}/{args.train_type}/" + "model_run_{}".format(run_id)+"_epoch_{}.csv".format(best_epoch)
            with open(reult_file_path,'w') as f:
                f.write('\n'.join(map(str,[f'embed_dim: {embed_dim}',f'hidden_dim: {hidden_dim}',f'layer_num:{layer_num}', f'read_out: {readout}',f'test_rmse: {best_val_rmse}',f'test_pcc: {best_val_pcc}']))) 
            tb.add_hparams(
                {"lr": lr, "bsize": batch_size, "embed_dim": embed_dim, "layer_num": layer_num,'hidden_dim' : hidden_dim,'readout': parameters['readout'].index(readout)},
                {
                    "accuracy": val_pcc,
                    "loss": train_rmse,
                },
            )
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
        tb.close()
    train_model(parameters,Multi_Omics_TransRegression)