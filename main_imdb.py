#!/usr/bin/python3

import os
import torch
import argparse
import numpy as np
from utils.utils_graph import *
from warnings import simplefilter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optims.scheduler.scheduler_plateau import LR_SchedulerPlateau

simplefilter(action='ignore', category=FutureWarning)
torch.set_num_threads(10)

dir_path = os.path.dirname(__file__)
nfs_dataset_path1 = '/nfs4-p1/ckx/datasets/'
nfs_dataset_path2 = '/mnt/nfs4-p1/ckx/datasets/'


def get_motifmaxvalue(args, dataset):
    max_motifvalue = 0
    for g in dataset:
        g_motifvalue = g[0].ndata['sg_npower'].max().item()
        if g_motifvalue > max_motifvalue:
            max_motifvalue = g_motifvalue
    args.max_motifvalue = max_motifvalue
    return args

### add new arguments
def add_args_imdb(args, dataset): 

    # args = get_motifmaxvalue(args, dataset) if args.dataset_init=='motif' else args
    if args.dataset_eval in ['acc', 'f1score']:
        args.output_dim = int(2)
    elif args.dataset_eval == 'rocauc':
        args.output_dim = int(1)
    args.eval_metric = args.dataset_eval
    args.task_type = dataset.task_type
    args.identity = (f"{args.dataset_name}-"+
                     f"{args.dataset_init}-"+ 
                     f"{args.dataset_eval}-"+ 
                     f"{args.model}-"+
                     f"{args.num_layer}-"+
                     f"{args.embed_dim}-"+
                     f"{args.norm_type}-"+
                     f"{args.norm_affine}-"+
                     f"{args.activation}-"+
                     f"{args.dropout}-"+
                     f"{args.pool_type}-"+  
                     f"{args.skip_type}-"+  
                     f"{args.batch_size}-"+      
                     f"{args.lr}-"+
                     f"{args.lr_min}-"+  
                     f"{args.lr_warmup}-"+
                     f"{args.lr_patience}-"+
                     f"{args.weight_decay}-"+            
                     f"{args.seed}"
                     )
    if not os.path.exists(args.logs_perf_dir):
        os.mkdir(args.logs_perf_dir)
    if not os.path.exists(os.path.join(args.logs_perf_dir, args.dataset_name)):
        os.mkdir(os.path.join(args.logs_perf_dir, args.dataset_name))  
    args.perf_xlsx_dir = os.path.join(args.logs_perf_dir, args.dataset_name, 'xlsx')
    args.perf_imgs_dir = os.path.join(args.logs_perf_dir, args.dataset_name, 'imgs')
    args.perf_dict_dir = os.path.join(args.logs_perf_dir, args.dataset_name, 'dict')
    args.perf_best_dir = os.path.join(args.logs_perf_dir, args.dataset_name, 'best') 

    if not os.path.exists(args.logs_stas_dir):
        os.mkdir(args.logs_stas_dir)
    if not os.path.exists(os.path.join(args.logs_stas_dir, args.dataset_name)):
        os.mkdir(os.path.join(args.logs_stas_dir, args.dataset_name))
    args.stas_xlsx_dir = os.path.join(args.logs_stas_dir, args.dataset_name, 'xlsx')
    args.stas_imgs_dir = os.path.join(args.logs_stas_dir, args.dataset_name, 'imgs')

    return args


def main(args):
    # setting seeds
    set_seed(args)

    # check nfs dataset path
    if os.path.exists(nfs_dataset_path1):
        args.dataset_path = nfs_dataset_path1
    elif os.path.exists(nfs_dataset_path2):
        args.dataset_path = nfs_dataset_path2

    dataset, train_loader, valid_loader, test_loader = load_dataset(args)
    args = add_args_imdb(args, dataset)

    model = load_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    scheduler = LR_SchedulerPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience, verbose=True, 
                                    warmup=args.lr_warmup)                                  
    modelOptm = ModelOptLoading(model=model, 
                                optimizer=optimizer,
                                scheduler=scheduler,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                test_loader=test_loader,
                                args=args)
    modelOptm.optimizing()

    print('optmi')



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    ## datasets path and name
    parser.add_argument("--dataset_path", type=str, default='datasets')
    parser.add_argument("--dataset_name", type=str, default='imdb-binary')
    parser.add_argument("--dataset_init", type=str, default='ones')
    parser.add_argument("--dataset_eval", type=str, default='rocauc', choices=['rocauc', 'acc'])

    ## model parameters
    parser.add_argument("--model", type=str, default='GIN_IMDB', choices=['MLP_IMDB', 'GIN_IMDB', 'GCN_IMDB', 
                        'GraphSage_IMDB', 'GAT_IMDB', 'SGC_IMDB'])
    parser.add_argument("--device", type=int, default=1)  
    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=128) 
    parser.add_argument("--norm_type", type=str, default='batchnorm')
    parser.add_argument("--norm_affine", type=bool, default=True)
    parser.add_argument("--activation", type=str, default='relu', choices=['relu', 'None'])
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--init_dp", type=float, default=0.0)
    parser.add_argument("--pool_type", type=str, default="mean", choices=['mean','sum','max'])
    parser.add_argument("--skip_type", type=str, default='None', 
                        choices= ['None', 'Residual', 'Initial', 'Dense', 'Jumping'])
    parser.add_argument("--econv", action="store_true")
 
    ## optimization parameters and others
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--epoch_slice", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
       
    parser.add_argument("--logs_perf_dir", type=str, default=os.path.join(dir_path,'logs_perf'), 
                        help="logs' files of the loss and performance")
    parser.add_argument("--logs_stas_dir", type=str, default=os.path.join(dir_path,'logs_stas'), 
                        help="statistics' files of the avg and std")                        
    parser.add_argument("--node_weight", default=True)
    parser.add_argument("--state_dict", action="store_true")
    parser.add_argument("--breakout", action="store_true")
    parser.add_argument("--lr_warmup", action="store_true")

    args = parser.parse_args()
 
    main(args)