

import torch
import numpy as np
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
from xmlrpc.client import boolean
from utils.utils_statistic import add_args, get_metric, get_task_type, get_batchsize

dir_path = os.path.dirname(__file__)

### add arguments
parser = argparse.ArgumentParser()
## datasets path and name
parser.add_argument("--dataset_path", type=str, default='datasets')
parser.add_argument("--dataset_name", type=str, default='ogbl-collab')

## model parameters
parser.add_argument("--model", type=str, default='GCN', choices='GCN, GAT, GIN, SGC')
parser.add_argument("--device", type=int, default=1)  
parser.add_argument("--num_layer", type=int, default=3)
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--norm_type", type=str, default='bn')
parser.add_argument("--norm_affine", type=bool, default=True)
parser.add_argument("--activation", type=str, default='relu', choices=['relu', 'None'])
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--pool_type", type=str, default="mean", choices=['dke','mean','sum','max'])
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
# parser.add_argument("--norm_affine", action="store_true")
parser.add_argument("--node_weight", default=True)
parser.add_argument("--state_dict", action="store_true")

args = parser.parse_args()
args = add_args(args)

args.dataset_name = 'cora'
args.model = 'GraphSage'
args.embed_dim = 128
args.norm_type = 'supernorm'
args.lr = 1e-2
args.num_layer = 4
args.dropout = 0.5
args.batch_size = 128
args.weight_decay = 0.0

args.eval_metric = get_metric(args)
args.task_type = get_task_type(args)
args.batch_size = get_batchsize(args)

train_list = []
valid_list = []
test_list = []
for layer in list(range(2,5,1)):
    train_rst = []
    valid_rst = []
    test_rst = []
    args.num_layer = layer
    for i in [0, 1, 2, 3, 4, 5,6,7,8,9]:
        args.seed=i
        args = add_args(args)
        logs_table = pd.read_excel(os.path.join(args.perf_best_dir, args.identity+'.xlsx'))

        train_rst.append(logs_table.iat[2,1])
        valid_rst.append(logs_table.iat[4,1])
        test_rst.append(logs_table.iat[6,1])

    train_mean = np.mean(np.array(train_rst))
    train_std = np.std(np.array(train_rst))

    valid_mean = np.mean(np.array(valid_rst))
    valid_std = np.std(np.array(valid_rst))

    test_mean = np.mean(np.array(test_rst))
    test_std = np.std(np.array(test_rst))

    train_list.append(train_mean)
    valid_list.append(valid_mean)
    test_list.append(test_mean)

    print('testing')

    print(f"Layer: {layer} Model: {args.model} with {args.norm_type} on dataset {args.dataset_name} ")
    print(f"Valid mean : {valid_mean:.8f} with {valid_std:.8f} ")
    print(f"Test mean : {test_mean:.8f} with {test_std:.8f} ")

print(f"train list: {train_list}")
print(f"valid list: {valid_list}")
print(f"test list: {test_list}")
