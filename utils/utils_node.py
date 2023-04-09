
import os
import dgl
import time
import torch
import random
import numpy as np
import pandas as pd
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset
from datasets.dgl_planetoid_dataset import DglPlanetoidDataset
from networks.gcn import GCN_Node
from networks.gat import GAT_Node
from networks.graphsage import GraphSage_Node
from optims.optim_ogbn_acc import ModelOptLearning_OGBN_Acc
from optims.optim_ogbn_proteins import ModelOptLearning_OGBN_Proteins

from download_dataset import data_preprocess

### set random seed
def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if args.device >= 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

### load and preprocess dataset 
def load_dataset(args):    
    # load ogb dataset
    start_time = time.time()

    # load node prediction dataset 
    if 'ogbn' in args.dataset_name:
        dataset = DglNodePropPredDataset(name=args.dataset_name, root=args.dataset_path)
    else:
        dataset = DglPlanetoidDataset(name=args.dataset_name, root=args.dataset_path)
    split_idx = dataset.get_idx_split()

    if not 'motif_factor' in dataset.graph[0].ndata.keys():
        dataset = data_preprocess(args, dataset)

    if 'ogbn-proteins' == args.dataset_name:
        graph=dataset.graph[0]
        graph.update_all(fn.copy_e('feat', 'e'), fn.mean('e', 'feat'))

    end_time = time.time()
    print('- ' * 30)
    print(f'{args.dataset_name} dataset loaded, using {end_time-start_time} seconds.')
    print('- ' * 30)        
    return dataset, split_idx


### add new arguments
def add_args(args, dataset): 
    args.dataset_name = args.dataset_name.lower()
    args.input_dim = dataset.graph[0].ndata['feat'].shape[1]
    if args.dataset_name in 'ogbn-proteins':
        args.output_dim = dataset.num_tasks
    elif args.dataset_name in ['ogbn-arxiv', 'cora', 'citeseer', 'pubmed']:
        args.output_dim = dataset.num_classes
    args.task_type = dataset.task_type
    args.eval_metric = dataset.eval_metric    
    args.identity = (f"{args.dataset_name}-"+
                     f"{args.model}-"+
                     f"{args.num_layer}-"+
                     f"{args.embed_dim}-"+
                     f"{args.norm_type}-"+
                     f"{args.norm_affine}-"+
                     f"{args.activation}-"+
                     f"{args.dropout}-"+
                     f"{args.skip_type}-"+  
                     f"{args.lr}-"+  
                     f"{args.lr_min}-"+
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

### load gnn model
def load_model(args):
    if args.model == 'GCN':
        model = GCN_Node(args.input_dim, args.embed_dim, args.output_dim, args.num_layer, args).to(args.device)
    if args.model == 'GraphSage':
        model = GraphSage_Node(args.input_dim, args.embed_dim, args.output_dim, args.num_layer, args).to(args.device)
    if args.model == 'GAT':
        model = GAT_Node(args.input_dim, args.embed_dim, args.output_dim, args.num_layer, args).to(args.device)  

    print('- ' * 30)
    if torch.cuda.is_available():
        print(f'{args.model} with {args.norm_type} norm, {args.dropout} dropout, on gpu', torch.cuda.get_device_name(0))
    else:
        print(f'{args.model} with {args.norm_type} norm, {args.dropout} dropout')
    print('- ' * 30) 
    return model

### load model optimizing and learning class
def ModelOptLoading(model, optimizer, scheduler, args):

    if args.dataset_name in ['cora', 'citeseer', 'pubmed', 'ogbn-arxiv']:
        modelOptm = ModelOptLearning_OGBN_Acc(
                                model=model, 
                                optimizer=optimizer,
                                scheduler=scheduler,                          
                                args=args)
    elif 'proteins' in args.dataset_name:
        modelOptm = ModelOptLearning_OGBN_Proteins(
                                model=model, 
                                optimizer=optimizer,
                                scheduler=scheduler,                          
                                args=args)

    return modelOptm



def print_best_log(args,  eopch_slice=0, 
                   metric_list='all'):
    key_metric=f'valid-{args.eval_metric}'
    logs_table = pd.read_excel(os.path.join(args.perf_xlsx_dir, args.identity+'.xlsx'))
    metric_log = logs_table[key_metric]

    best_epoch = metric_log[eopch_slice:].idxmax()  

    best_frame = logs_table.loc[best_epoch]
    if not os.path.exists(args.perf_best_dir):
        os.mkdir((args.perf_best_dir))
    best_frame.to_excel(os.path.join(args.perf_best_dir, args.identity+'.xlsx'))

    if metric_list == 'all':
        print(best_frame)
    else:
        for metric in metric_list:
            print(f'{metric }: {best_frame[metric]}')
    return 0

