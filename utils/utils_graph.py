
import os
import dgl
import time
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from ogb.graphproppred.dataset_dgl import DglGraphPropPredDataset, collate_dgl
from networks.gcn import GCN_Graph
from networks.graphsage import GraphSage_Graph
from networks.gat import GAT_Graph
from networks.gin import GIN_Graph
from networks.imdb.mlp_imdb import MLP_IMDB
from networks.imdb.gin_imdb import GIN_IMDB
from networks.imdb.gcn_imdb import GCN_IMDB
from networks.imdb.gat_imdb import GAT_IMDB
from networks.imdb.graphsage_imdb import GraphSage_IMDB
from networks.imdb.sgc_imdb import SGC_IMDB
from optims.optim_ogbg_mol import ModelOptLearning_OGBG_MOL
from optims.optim_ogbg_ppa import ModelOptLearning_OGBG_PPA
from optims.optim_graph_zinc import ModelOptLearning_ZINC
from optims.optim_graph_imdb import ModelOptLearning_IMDB
from download_dataset import data_preprocess
from datasets.dgl_zinc_dataset import DglZincDataset
from datasets.dgl_imdb_dataset import DglIMDBDataset

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

    if args.dataset_name == 'zinc':
        dataset = DglZincDataset(name=args.dataset_name, root=args.dataset_path)
    elif args.dataset_name in ['imdb-multi', 'imdb-binary']:
        dataset = DglIMDBDataset(name=args.dataset_name, root=args.dataset_path)
    elif 'ogbg' in args.dataset_name: # load ogb graph dataset
        dataset = DglGraphPropPredDataset(name=args.dataset_name, root=args.dataset_path)

    if not 'sg_factor' in dataset.graphs[0].ndata.keys():
        dataset = data_preprocess(args, dataset)

    # split_idx for training, valid and test 
    split_idx = dataset.get_idx_split()
    
    train_loader= DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, collate_fn=collate_dgl, num_workers=0)
    valid_loader= DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, collate_fn=collate_dgl, num_workers=0)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, collate_fn=collate_dgl, num_workers=0)    
    end_time = time.time()
    print('- ' * 30)
    print(f'{args.dataset_name} dataset loaded, using {end_time-start_time} seconds.')
    print('- ' * 30)        
    return dataset, train_loader, valid_loader, test_loader


def get_output_dim(dataset, args):
    if 'zinc' == args.dataset_name:
        return int(dataset.num_tasks)
    elif 'ogbg-mol' in args.dataset_name:
        return int(dataset.num_tasks)
    elif 'ppa' in args.dataset_name:
        return int(dataset.num_classes)

### add new arguments
def add_args(args, dataset): 
    
    args.task_type = dataset.task_type
    args.eval_metric = dataset.eval_metric  
    args.output_dim = get_output_dim(dataset, args)
    args.identity = (f"{args.dataset_name}-"+
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
        model = GCN_Graph(args.embed_dim, args.output_dim, args.num_layer, args).to(args.device)
    if args.model == 'GraphSage':
        model = GraphSage_Graph(args.embed_dim, args.output_dim, args.num_layer, args).to(args.device)
    if args.model == 'GAT':
        model = GAT_Graph(args.embed_dim, args.output_dim, args.num_layer, args).to(args.device)
    if args.model == 'GIN':
        model = GIN_Graph(args.embed_dim, args.output_dim, args.num_layer, args).to(args.device)
    if args.model == 'MLP_IMDB':
        model = MLP_IMDB(args.embed_dim, args.output_dim, args.num_layer, args).to(args.device)
    if args.model == 'GIN_IMDB':
        model = GIN_IMDB(args.embed_dim, args.output_dim, args.num_layer, args).to(args.device)
    if args.model == 'GCN_IMDB':
        model = GCN_IMDB(args.embed_dim, args.output_dim, args.num_layer, args).to(args.device)
    if args.model == 'GAT_IMDB':
        model = GAT_IMDB(args.embed_dim, args.output_dim, args.num_layer, args).to(args.device)
    if args.model == 'GraphSage_IMDB':
        model = GraphSage_IMDB(args.embed_dim, args.output_dim, args.num_layer, args).to(args.device)
    if args.model == 'SGC_IMDB':
        model = SGC_IMDB(args.embed_dim, args.output_dim, args.num_layer, args).to(args.device)

    print('- ' * 30)
    if torch.cuda.is_available():
        print(f'{args.model} with {args.pool_type} pool, {args.norm_type} norm, {args.dropout} dropout, seed:{args.seed} on gpu', torch.cuda.get_device_name(0))
    else:
        print(f'{args.model} with {args.pool_type} pool, {args.norm_type} norm, {args.dropout} dropout, seed:{args.seed}')
    print('- ' * 30) 
    return model

### load model optimizing and learning class
def ModelOptLoading(model, optimizer, scheduler,
                    train_loader, valid_loader, test_loader,
                    args):
    if 'zinc' == args.dataset_name:
        modelOptm = ModelOptLearning_ZINC(
                        model=model, 
                        optimizer=optimizer,
                        scheduler=scheduler,
                        train_loader=train_loader,
                        valid_loader=valid_loader,
                        test_loader=test_loader,
                        args=args)
    elif 'imdb' in args.dataset_name:
        modelOptm = ModelOptLearning_IMDB(
                        model=model, 
                        optimizer=optimizer,
                        scheduler=scheduler,
                        train_loader=train_loader,
                        valid_loader=valid_loader,
                        test_loader=test_loader,
                        args=args)
    elif 'ogbg-mol' in args.dataset_name:
        modelOptm = ModelOptLearning_OGBG_MOL(
                                model=model, 
                                optimizer=optimizer,
                                scheduler=scheduler,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                test_loader=test_loader,
                                args=args)
    elif 'ogbg-ppa' in args.dataset_name:
        modelOptm = ModelOptLearning_OGBG_PPA(
                                model=model, 
                                optimizer=optimizer,
                                scheduler=scheduler,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                test_loader=test_loader,
                                args=args)

    return modelOptm



def print_best_log(args,  eopch_slice=0, 
                   metric_list='all'):
    key_metric=f'valid-{args.eval_metric}'
    logs_table = pd.read_excel(os.path.join(args.perf_xlsx_dir, args.identity+'.xlsx'))
    metric_log = logs_table[key_metric]
    if "classification" in args.task_type:
        best_eval_values = max(metric_log[eopch_slice:])
    else:
        best_eval_values = min(metric_log[eopch_slice:])
    best_eval_epochs = list(np.where(metric_log[eopch_slice:].to_numpy()==best_eval_values)[0]+eopch_slice)

    test_logs = logs_table[f'test-{args.eval_metric}']

    best_epoch = best_eval_epochs[0]
    best_value = test_logs[int(best_epoch)]
    for i in best_eval_epochs:
        test_value = test_logs[i]
        if "classification" in args.task_type and test_value>best_value:
            best_value = test_value
            best_epoch = i
        elif 'reg' in args.task_type and test_value<best_value:
            best_value = test_value
            best_epoch = i
            
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

