import sys
sys.path.append('.')
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import torch
import torch.cuda.amp as amp
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR
from model.drp_model.DRP_nn import DRP_multi_view
from torch.utils.data import Dataset, DataLoader
from prepare_data.create_cell_feat import *
from prepare_data.create_drug_feat import *
from prepare_data.create_drp_dict import *
from prepare_data.DRP_loader import *
from base_line.TGSA_model.tgsa_model import *
from torch_geometric.data import Batch
import torch.optim as opt
from torch_geometric.nn import  graclus
from tqdm import tqdm
from itertools import product
from train_model.utils import get_polynomial_decay_schedule_with_warmup
import argparse
       



# def train_step(model, train_loader, optimizer, loss_op, writer, epoch,device):

#     model.train()
#     y_true, preds = [], []
#     optimizer.zero_grad()
#     for data in tqdm(train_loader):
#         drug_atom, drug_bond, cell_ge, cell_cnv, cell_mut, ic50 = data
#         y_true.append(ic50.view(-1, 1).float())
#         drug_atom = drug_atom.to(device)  
#         drug_bond = drug_bond.to(device)
#         cell_ge = cell_ge.to(device)
#         cell_cnv = cell_cnv.to(device)
#         cell_mut = cell_mut.to(device)
#         ic50 = ic50.to(device)  
#         out = model(drug_atom,drug_bond, cell_ge, cell_mut, cell_cnv)
#         loss = loss_op(out, ic50.view(-1, 1).float())
#         preds.append(out.float().cpu())
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         # scheduler.step()
#     y_true = torch.cat(y_true, dim=0).detach().numpy()
#     y_pred = torch.cat(preds, dim=0).detach().numpy()
#     rmse = mean_squared_error(y_true,y_pred, squared=False)
#     pcc = pearsonr(y_true.flatten(), y_pred.flatten())[0]
#     r_2 = r2_score(y_true.flatten(), y_pred.flatten())
#     MAE = mean_absolute_error(y_true.flatten(), y_pred.flatten())
#     writer.add_scalar("Loss", rmse, epoch)
#     writer.add_scalar("Accuracy/train/rmse", rmse, epoch)
#     writer.add_scalar("Accuracy/train/mae", MAE, epoch)
#     writer.add_scalar("Accuracy/train/pcc", pcc, epoch)
#     writer.add_scalar("Accuracy/train/r_2", r_2, epoch)
#     return rmse, pcc

def cross_entropy_one_hot(input, target):
    batch_size = input.shape[0]
    num_target = input.shape[1]
    target = target.reshape(batch_size,num_target)
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)
def total_loss(out, ic50, cell_class, target, drug_class, threshold, pred_pathway,target_pathway, args):
    m = args.regular_weight
    n = args.regular_weight_drug
    p = args.regular_weight_drug_path_way
    pred_loss = nn.MSELoss()(out, ic50)
    class_l = 0.0
    drug_l = 0.0
    drug_pathway_l = 0.0
    if args.use_regulizer_drug == 'True':
        drug_l = nn.MSELoss()(drug_class, threshold.view(-1,1))
    if args.use_regulizer == 'True':
        class_l = cross_entropy_one_hot(cell_class,target)
    if args.use_drug_path_way == 'True':
        drug_pathway_l = cross_entropy_one_hot(pred_pathway, target_pathway)
    return pred_loss + m * class_l + n* drug_l + p*drug_pathway_l

def train_step(model, train_loader, optimizer, writer, epoch, device, args):
    # enable automatic mixed precision
    scaler = amp.GradScaler()

    model.train()
    y_true, preds = [], []
    optimizer.zero_grad()
    for data in tqdm(train_loader):
        drug_atom, drug_bond, cell_ge, cell_cnv, cell_mut, raw_gene, ic50 = data
        y_true.append(ic50.view(-1, 1).float())
        drug_atom = drug_atom.to(device)  
        drug_bond = drug_bond.to(device)
        cell_ge = cell_ge.to(device)
        cell_cnv = cell_cnv.to(device)
        raw_gene = raw_gene.to(device)
        cell_mut = cell_mut.to(device)
        ic50 = ic50.to(device)  
        with amp.autocast():
            out_dict = model(drug_atom,drug_bond, cell_ge, cell_mut, cell_cnv, raw_gene)
            out = out_dict['pred']
            cell_class = out_dict['cell_regulizer']
            drug_class = out_dict['drug_regulizer']
            pred_pathway = out_dict['drug_pathway']
            # loss_main = loss_op(out, ic50.view(-1, 1).float())
            loss = total_loss(out, ic50.view(-1, 1).float(),cell_class, cell_ge.cancer_type, drug_class, drug_atom.threshold, pred_pathway,drug_atom.path_way, args)
        preds.append(out.float().cpu())
        # perform backward pass and optimizer step using the scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # scheduler.step()
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
    print(optimizer.param_groups[0]['lr'])
    return rmse, pcc

@torch.no_grad()
def test_step(model,loader,device):
    model.eval()
    y_true, preds = [], []
    for data in tqdm(loader):
        drug_atom, drug_bond, cell_ge, cell_cnv, cell_mut, raw_gene, ic50 = data
        raw_gene = raw_gene.to(device)
        drug_atom, drug_bond, cell_ge, cell_cnv, cell_mut, ic50  = data = drug_atom.to(device), drug_bond.to(device), cell_ge.to(device), cell_cnv.to(device),cell_mut.to(device), ic50.to(device)
        y_true.append(ic50.view(-1, 1).float())
        out_dict = model(drug_atom,drug_bond, cell_ge, cell_mut, cell_cnv, raw_gene)
        out = out_dict['pred']
        preds.append(out.float().cpu())
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(preds, dim=0).numpy()
    rmse = mean_squared_error(y_true,y_pred, squared=False)
    pcc = pearsonr(y_true.flatten(), y_pred.flatten())[0]
    r_2 = r2_score(y_true.flatten(), y_pred.flatten())
    MAE = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    return rmse,pcc,r_2,MAE








def train_multi_view_model(args, train_set, test_set, mut_cluster, cnv_cluster, ge_cluster):
    # save_dir = 'tb_multi_view_full_'+ args.train_type
    save_dir = 'best_model_' + args.train_type
    # if args.use_norm_ic50 == 'True':
    #     weights = 'weights_atten_norm_multi_omics'
    # else: weights = 'weights_atten_multi_omics'
    # if os.path.exists(f"./{weights}/{args.train_type}") is False:
    #     os.makedirs(f"./{weights}/{args.train_type}")
    # param_values = [v for v in parameters.values()]
    device = args.device
    # for run_id, (lr, embed_dim, batch_size, layer_num,hidden_dim,readout, use_fp, optimizer_name, view_dim, use_regulizer) in enumerate(product(*param_values)):    
    lr = args.lr
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    layer_num = args.layer_num
    hidden_dim = args.hidden_dim
    readout = args.readout
    use_regulizer = args.use_regulizer
    regular_weight = args.regular_weight
    use_regulizer_drug = args.use_regulizer_drug
    regular_weight_drug = args.regular_weight_drug
    use_drug_path_way = args.use_drug_path_way
    regular_weight_drug_path_way = args.regular_weight_drug_path_way
    view_dim = args.view_dim
    use_cnn = args.use_cnn
    # run_id = f'embed_dim_{embed_dim}_'+f'hidden_dim_{hidden_dim}_'+f'view_dim_{view_dim}_'+f'regular_weight_{regular_weight}_' + f'use_regulizer_drug_{use_regulizer_drug}' + f'use_drug_path_way_{use_drug_path_way}_' +f'regular_weight_drug_{regular_weight_drug}_' + f'regular_weight_drug_path_way_{regular_weight_drug_path_way}_' + f'use_cnn_{use_cnn}' + f'_lr_{args.lr}'
    run_id = args.use_predined_gene_cluster + '_' + args.scheduler_type + '_' + f'{args.dropout_rate}'
    path = f'./Drug_response/{save_dir}/{run_id}'+'.pth'    
    n_epochs = args.epochs
    print(path)
    print("run id:", run_id)
    model_config = {'embed_dim': embed_dim, 'dropout_rate': args.dropout_rate,'hidden_dim' : hidden_dim,
                    'layer_num': layer_num, 'readout': readout, 'use_regulizer' : use_regulizer, 'use_drug_path_way' : use_drug_path_way,'use_regulizer_drug' : use_regulizer_drug,'use_smiles' : False, 'view_dim':view_dim, 'JK' : 'True', 'use_cnn' : use_cnn, 'use_predined_gene_cluster' : args.use_predined_gene_cluster}
    model = DRP_multi_view(mut_cluster, cnv_cluster, ge_cluster, model_config).to(device)
    optimizer = opt.AdamW(model.parameters(), lr=lr, weight_decay= 0.01)
        # scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,num_warmup_steps=50, num_training_steps=n_epochs, lr_end = 1e-4, power=1)
    # elif optimizer_name == 'SGD': 
        # optimizer = opt.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)
    # cos_lr = lambda x : ((1+math.cos(math.pi* x /100) )/2)*(1-args.lrf) + args.lrf
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cos_lr)
    current_time = datetime.datetime.now().time()
    print('Begin Training')
    print(f'Embed_dim_drug : {embed_dim}'+ '\n' +f'Hidden_dim_cell : {hidden_dim} \n' +  f'layer_num : {layer_num} \n'+ 
            f'read_out_function : {readout}\n'  +f'batch_size : {batch_size}\n' + f'view_dim : {view_dim}\n' + 
            f'lr : {lr}\n' + f'use_regulizer : {use_regulizer}\n' + f'use_regulizer_drug : {use_regulizer_drug}\n' + f'use_cnn : {use_cnn}')
    tb = SummaryWriter(comment=current_time, log_dir=f'./Drug_response/{save_dir}')
    torch.save([mut_cluster, cnv_cluster, ge_cluster],f'/home/yurui/Atten_Geom_DRP/Drug_response/{save_dir}/{run_id}'+'_cluster.pth' )
    train_loader = multi_drp_loader(train_set,batch_size= batch_size,shuffle = True, num_workers = args.num_workers, use_raw_gene= 'True')
    test_loader = multi_drp_loader(test_set,batch_size= 512,shuffle = True, num_workers = args.num_workers, use_raw_gene= 'True')
    epoch_len = len(str(n_epochs))
    best_val_pcc = -1
    if args.use_norm_ic50 == 'True':
        results = 'results_multi_view_full_norm'
    else: results = 'results_multi_view_full'
    if os.path.exists(f"./{results}/{args.train_type}") is False:
        os.makedirs(f"./{results}/{args.train_type}")
    reult_file_path = f"./{results}/{args.train_type}/" + "model_run_{}".format(run_id)+".csv"
    early_stop_count = 0 
    best_epoch = 0 
    best_mae = 100
    best_val_pcc = -1
    best_val_rmse = 100
    scheduler_type = args.scheduler_type
    if args.scheduler_type == 'OP':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience= 8 , verbose=True, min_lr= 0.05 * args.lr, factor= 0.1)
    elif args.scheduler_type == 'ML':
        scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1)
    for epoch in range(n_epochs):
        if early_stop_count < args.early_stop_count :
            train_rmse, train_pcc = train_step(model, train_loader, optimizer, tb, epoch, device, args)
            if args.scheduler_type == 'ML':
                scheduler.step()
            # scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] '  + 
                        f'train_rmse: {train_rmse:.5f} ' +
                        f'train_pcc: {train_pcc:.5f} ' +  f'lr : {current_lr}')
            # torch.save(model.state_dict(), f"./{weights}/{args.train_type}"+"/model_run_{}".format(run_id)+"_epoch_{}.pth".format(epoch))
            print(print_msg)
            # print(f'Epoch: {epoch:03d}, Loss: {train_rmse:.4f}')
            if epoch % args.check_step == 0:
                val_rmse,val_pcc, val_r_2, val_mae = test_step(model,test_loader, device)
                if args.scheduler_type == 'OP':
                    scheduler.step(val_rmse)
                tb.add_scalar('Accuracy/test/pcc', val_pcc, epoch )
                tb.add_scalar("Accuracy/test/rmse", val_rmse, epoch)
                tb.add_scalar("Accuracy/test/mae", val_mae, epoch)
                tb.add_scalar("Accuracy/test/r_2", val_r_2, epoch)
                tb.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
                print(f'Test for Epoch: {epoch:03d},  val_rmse:{val_rmse:.4f}, val_PCC: {val_pcc:.4f}')
                if val_rmse < best_val_rmse or val_pcc > best_val_pcc:
                    early_stop_count = 0
                    best_val_pcc = val_pcc
                    best_val_rmse = val_rmse
                    best_r_2 = val_r_2
                    best_mae = val_mae
                    best_epoch = epoch
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict' : optimizer.state_dict(),
                            }, path)
                    with open(reult_file_path,'w') as f:
                        f.write('\n'.join(map(str,[f'embed_dim: {embed_dim}',f'hidden_dim: {hidden_dim}',f'layer_num:{layer_num}', f'read_out: {readout}', f'view_dim : {view_dim}',
                                                f'test_rmse: {best_val_rmse}',f'test_pcc: {best_val_pcc}',f'test_r_2 : {best_r_2}', f'test_mae : {best_mae}', 
                                                f'best_epoch : {best_epoch}']))) 
                else: 
                    early_stop_count += 1 
                    print(f'Early stopping encounter : {early_stop_count}  times')
                if early_stop_count >= args.early_stop_count:
                    print('Early stopping!')
                    break
                print(f'Best epoch: {best_epoch:03d}, Best PCC: {best_val_pcc:.4f}')
                print(f'Best RMSE: {best_val_rmse:.4f}, Best R_2: {best_r_2:.4f}, Best MAE: {best_mae:.4f}')
    print("__________________________________________________________")
    tb.add_hparams(
        {"lr": lr, "bsize": batch_size, "embed_dim": embed_dim, "layer_num": layer_num,'hidden_dim' : hidden_dim,'readout': readout, 
            'best_epoch' : best_epoch, 'view_dim' : view_dim, "drop_out" : args.dropout_rate,
            'use_regulizer': use_regulizer, 'use_regulizer_drug': use_regulizer_drug, 'use_drug_path_way': use_drug_path_way,
            'use_cnn' : use_cnn, 'use_predined_gene_cluster' : args.use_predined_gene_cluster, 'scheduler_type' : args.scheduler_type,
            'regular_weight' : regular_weight, 'regular_weight_drug' : regular_weight_drug, 'regular_weight_drug_path_way': regular_weight_drug_path_way},
        {
            "best_pcc": best_val_pcc,
            "best_r2": best_r_2,
            "best_rmse": best_val_rmse,
            "best_mae": best_mae,
            "loss": train_rmse,
        },
    )
    tb.close()