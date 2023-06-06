#!/usr/bin/python3

import os
import torch
import argparse
import numpy as np
from utils.utils_graph import *
from warnings import simplefilter
from torch.optim.lr_scheduler import ReduceLROnPlateau

simplefilter(action='ignore', category=FutureWarning)
torch.set_num_threads(10)

dir_path = os.path.dirname(__file__)
nfs_dataset_path1 = '/nfs4-p1/ckx/datasets/'
nfs_dataset_path2 = '/mnt/nfs4-p1/ckx/datasets/'


def main(args):
    # setting seeds
    set_seed(args)

    # check nfs dataset path
    if os.path.exists(nfs_dataset_path1):
        args.dataset_path = nfs_dataset_path1
    elif os.path.exists(nfs_dataset_path2):
        args.dataset_path = nfs_dataset_path2

    dataset, train_loader, valid_loader, test_loader = load_dataset(args)
    args = add_args(args, dataset)

    model = load_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience, verbose=True)                                   
    modelOptm = ModelOptLoading(model=model, 
                                optimizer=optimizer,
                                scheduler=scheduler,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                test_loader=test_loader,
                                args=args)
    modelOptm.optimizing()

    metric_list = ['train-loss','train-rocauc', 'valid-rocauc', 'test-rocauc']
    print_best_log(args, eopch_slice=args.epoch_slice)

    # plot_logs(args, metric_list)

    print('optmi')



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    ## datasets path and name
    parser.add_argument("--dataset_path", type=str, default='datasets')
    parser.add_argument("--dataset_name", type=str, default='ogbg-molhiv',
                        choices=['ogbg-moltoxcast', 'zinc','ogbg-molclintox', 'ogbg-ppa', 'ogbg-mollipo','ogbg-molhiv'])
    ## model parameters
    parser.add_argument("--model", type=str, default='GCN', choices=['GCN', 'GraphSage', 'GAT', 'GIN'])
    parser.add_argument("--device", type=int, default=0)  
    parser.add_argument("--num_layer", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--norm_type", type=str, default='supernorm')
    parser.add_argument("--norm_affine", type=bool, default=True)
    parser.add_argument("--activation", type=str, default='relu', choices=['relu', 'None'])
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--pool_type", type=str, default="mean", choices=['mean','sum','max'])
    parser.add_argument("--skip_type", type=str, default='None', 
                        choices= ['None', 'Residual', 'Initial', 'Dense', 'Jumping'])
    parser.add_argument("--econv", action="store_true")

    ## optimization parameters and others
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--epoch_slice", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_patience", type=int, default=15)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
       
    parser.add_argument("--logs_perf_dir", type=str, default=os.path.join(dir_path,'logs_perf'), 
                        help="logs' files of the loss and performance")
    parser.add_argument("--logs_stas_dir", type=str, default=os.path.join(dir_path,'logs_stas'), 
                        help="statistics' files of the avg and std")                        
    parser.add_argument("--node_weight", default=True)
    parser.add_argument("--state_dict", action="store_true")
    parser.add_argument("--breakout", action="store_true")

    args = parser.parse_args()
 
    main(args)
