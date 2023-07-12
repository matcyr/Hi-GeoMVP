import sys
sys.path.append('.')
from model.drp_model.DRP_nn import *
from model.cell_model.cell_encoder import *
from model.drug_model.drug_encoder import *
from prepare_data.DRP_loader import *
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter
import torch.optim as opt
from itertools import product
import os






class MUT_model(nn.Module):
    def __init__(self, model_config) -> None:
        super(Multi_Omics_TransRegression, self).__init__()
        self.hidden_dim = model_config.get('hidden_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.cell_conv_type = model_config.get('cell_conv_type')  ## cnn, gnn
        if self.cell_conv_type == 'cnn':
            self.cell_encoder = cell_cnn(model_config, 2560)
        elif self.cell_conv_type == 'gnn':
            self.cell_encoder = Cell_encoder_mut(model_config)
        # elif self.cell_conv_type == 'both':
        #     self.cell__cnn_encoder = cell_cnn(model_config, 2560)
        #     self.cell__gnn_encoder = Cell_encoder_mut(model_config)
    def forward(self, cell_mut):
        batch_size = int(cell_mut.x.shape[0]/636)
        if self.cell_conv_type == 'cnn':
            xt = self.cell_encoder(cell_mut.x.view(batch_size,1,636))
        











def train_single_omics_atten_step(model, train_loader, optimizer, loss_op, writer, epoch,device):
    model.train()
    y_true, preds = [], []
    optimizer.zero_grad()
    for data in tqdm(train_loader):
        drug_atom,drug_bond,cell_ge,cell_cnv,cell_mut, ic50 = data
        y_true.append(ic50.view(-1, 1).float())
        drug_atom = drug_atom.to(device)  
        drug_bond = drug_bond.to(device)
        cell_ge = cell_ge.to(device)
        # ge_sim = ge_sim.to(device)
        ic50 = ic50.to(device)  
        cell_cnv = cell_cnv.to(device)
        cell_mut = cell_mut.to(device)
        # with torch.cuda.amp.autocast():
        pred = model(drug_atom,drug_bond,cell_ge,cell_cnv,cell_mut)
        preds.append(pred.float().cpu())
        loss = loss_op(pred, ic50.view(-1, 1).float())
        # scalr.scale(loss).backward()
        # scalr.step(optimizer)
        # scalr.update()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    y_true = torch.cat(y_true, dim=0).detach().numpy()
    y_pred = torch.cat(preds, dim=0).detach().numpy()
    rmse = mean_squared_error(y_true,y_pred, squared=False)
    pcc = pearsonr(y_true.flatten(), y_pred.flatten())[0]
    r_2 = r2_score(y_true.flatten(), y_pred.flatten())
    MAE = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    writer.add_scalar("Loss", rmse, epoch)
    writer.add_scalar("Accuracy/train/rmse", rmse, epoch)
    writer.add_scalar("Accuracy/train/mae", MAE, epoch)
    writer.add_scalar("Accuracy/train/pcc", pcc, epoch)
    writer.add_scalar("Accuracy/train/r_2", r_2, epoch)
    return rmse, pcc

@torch.no_grad()
def test_multi_omics_atten_step(model,loader,device):
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
    pcc = pearsonr(y_true.flatten(), y_pred.flatten())[0]
    r_2 = r2_score(y_true.flatten(), y_pred.flatten())
    MAE = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    return rmse,pcc,r_2,MAE

def train_model(args, model_sturcture, train_set, test_set ,parameters):
    save_dir = 'atten_model_multi_omics_increase_dim_'+ args.train_type
    if args.use_norm_ic50 == 'True':
        weights = 'weights_atten_norm_multi_omics'
    else: weights = 'weights_atten_multi_omics'
    if os.path.exists(f"./{weights}/{args.train_type}") is False:
        os.makedirs(f"./{weights}/{args.train_type}")
    param_values = [v for v in parameters.values()]
    device = args.device
    for run_id, (lr, embed_dim, batch_size, layer_num,hidden_dim,readout, use_fp, attention_dim, optimizer_name) in enumerate(product(*param_values)):    
        path = f'./Drug_response/{save_dir}/{run_id}'+'.pth'    
        print(path)
        print("run id:", run_id + 1)
        model_config = {'embed_dim': embed_dim, 'dropout_rate': args.dropout_rate,'hidden_dim' : hidden_dim,
                        'layer_num': layer_num, 'readout': readout, 'use_cnn' : True, 'use_fp' : use_fp, 'use_smiles' : False, 'attention_dim' : attention_dim}
        model = model_sturcture(model_config).to(device)
        if optimizer_name == 'Adam':
            optimizer = opt.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD': 
            optimizer = opt.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)
        # cos_lr = lambda x : ((1+math.cos(math.pi* x /100) )/2)*(1-args.lrf) + args.lrf
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cos_lr)
        criterion = torch.nn.MSELoss()
        comment = f' batch_size = {batch_size} lr = {lr} embed_dim = {embed_dim} layer_num = {layer_num} readout = {readout} hidden_dim = {hidden_dim} use_fp = {use_fp} optimizer_name = {optimizer_name} attention_dim = {attention_dim}'
        print('Begin Training')
        print(f'Embed_dim_drug : {embed_dim}'+ '\n' +f'Hidden_dim_cell : {hidden_dim} \n' +  f'layer_num : {layer_num} \n'+ 
              f'read_out_function : {readout}\n' + f'use_fp : {use_fp}\n' +f'batch_size : {batch_size}\n' + f'optimizer : {optimizer_name}\n' + f'lr : {lr}\n' + f'attention_dim : {attention_dim}\n' )
        tb = SummaryWriter(comment=comment, log_dir=f'./Drug_response/{save_dir}/')
        train_loader = multi_drp_loader(train_set,batch_size= batch_size,shuffle = True, num_workers = args.num_workers)
        test_loader = multi_drp_loader(test_set,batch_size= 512,shuffle = True, num_workers = args.num_workers)
        n_epochs = args.epochs
        epoch_len = len(str(n_epochs))
        best_val_pcc = -1
        for epoch in range(n_epochs):
            train_rmse, train_pcc = train_multi_omics_atten_step(model, train_loader, optimizer, criterion, tb, epoch, device)
            # scheduler.step()
            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                        f'train_rmse: {train_rmse:.5f} ' +
                        f'train_pcc: {train_pcc:.5f} ')
            # torch.save(model.state_dict(), f"./{weights}/{args.train_type}"+"/model_run_{}".format(run_id)+"_epoch_{}.pth".format(epoch))
            print(print_msg)
            # print(f'Epoch: {epoch:03d}, Loss: {train_rmse:.4f}')
            if epoch % 10 == 0:
                val_rmse,val_pcc, val_r_2, val_mae = test_multi_omics_atten_step(model,test_loader, device)
                if val_pcc > best_val_pcc:
                    best_val_pcc = val_pcc
                    best_val_rmse = val_rmse
                    best_r_2 = val_r_2
                    best_mae = val_mae
                    best_epoch = epoch
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, path)
                tb.add_scalar('Accuracy/test/pcc', val_pcc, epoch )
                tb.add_scalar("Accuracy/test/rmse", val_rmse, epoch)
                tb.add_scalar("Accuracy/test/mae", val_mae, epoch)
                tb.add_scalar("Accuracy/test/r_2", val_r_2, epoch)
                print(f'Test for Epoch: {epoch:03d},  val_rmse:{val_rmse:.4f}, val_PCC: {val_pcc:.4f}')
                print(f'Best epoch: {best_epoch:03d}, Best PCC: {best_val_pcc:.4f}')
                print(f'Best RMSE: {best_val_rmse:.4f}, Best R_2: {best_r_2:.4f}, Best MAE: {best_mae:.4f}')
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
            f.write('\n'.join(map(str,[f'embed_dim: {embed_dim}',f'hidden_dim: {hidden_dim}',f'layer_num:{layer_num}', f'read_out: {readout}',
                                       f'test_rmse: {best_val_rmse}',f'test_pcc: {best_val_pcc}',f'test_r_2 : {best_r_2}', f'test_mae : {best_mae}']))) 
        tb.add_hparams(
            {"lr": lr, "bsize": batch_size, "embed_dim": embed_dim, "layer_num": layer_num,'hidden_dim' : hidden_dim,'readout': readout, 
             'best_epoch' : best_epoch, 'optimizer_name': optimizer_name, 'use_fp' : use_fp, 'attention_dim' : attention_dim},
            {
                "best_pcc": best_val_pcc,
                "best_r2": best_r_2,
                "best_rmse": best_val_rmse,
                "best_mae": best_mae,
                "loss": train_rmse,
            },
        )

    tb.close()
if __name__ == '__main__':
    save_dir = 'tensorboard_atten_model_multi_omics_mix'
    run_id = 0
    path = f'./Drug_response/{save_dir}/{run_id}'+'.pth'    
    print(path)