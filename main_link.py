#!/usr/bin/python3

import os
import torch
import argparse
import numpy as np
from utils.utils_link import *
from warnings import simplefilter
from modules.predict.predict import LinkPredictor

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

    dataset, split_edge = load_dataset(args)
    args = add_args(args, dataset)

    model = load_model(args) # GNN Module
    predictor = LinkPredictor(model.embed_dim, model.embed_dim, args.predictor_output,
                              num_layers=args.predictor_layers, dropout=args.dropout).to(args.device)   # Link Predictor 

    modelOptm = ModelOptLoading(model=model, 
                                predictor=predictor,                          
                                args=args)
    modelOptm.optimizing(dataset, split_edge)

    print('optmi')



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    ## datasets path and name
    parser.add_argument("--dataset_path", type=str, default='datasets')
    parser.add_argument("--dataset_name", type=str, default='ogbl-collab')

    ## GNN Model parameters
    parser.add_argument("--model", type=str, default='GraphSage', choices=['GCN', 'GraphSage'])
    parser.add_argument("--device", type=int, default=0)  
    parser.add_argument("--num_layer", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--norm_type", type=str, default='nodenorm-bn')
    parser.add_argument("--norm_affine", type=bool, default=True)
    parser.add_argument("--activation", type=str, default='relu', choices=['relu', 'None'])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--skip_type", type=str, default='None', 
                        choices= ['None', 'Residual', 'Initial', 'Dense', 'Jumping'])
    parser.add_argument("--econv", action="store_true")

    # Link Predictor parameters
    parser.add_argument("--predictor_layers", type=int, default=3)
    parser.add_argument("--predictor_output", type=int, default=1)

    ## optimization parameters and others
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--epoch_slice", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64*1024)
    parser.add_argument("--lr", type=float, default=1e-2)
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
