

import os
import torch
import numpy as np

def load_logs():
    return 0

def get_batchsize(args):
    if args.dataset_name in ['imdb-binary']:
        return 32
    elif args.dataset_name in ['ogbg-mollipo']: 
        return 64
    elif args.dataset_name in ['ogbg-moltoxcast','zinc','ogbg-ppa']: 
        return 128
    elif args.dataset_name in ['ogbg-molhiv','zinc']: 
        return 256
    elif args.dataset_name in ['ogbg-molpcba']:
        return 512



### add new arguments
def add_args_graph(args): 
    args.batch_size = get_batchsize(args)
    args.identity = (f"{args.dataset_name}-"+
                     f"{args.model}-"+
                     f"{args.num_layer}-"+
                     f"{args.embed_dim}-"+
                     f"{args.norm_type}-"+
                     f"{args.norm_affine}-"+
                     f"{args.activation}-"+
                     f"{args.dropout}-"+
                     f"{args.pool_type}-"+  
                     f"{args.residual}-"+
                     f"{args.plain}-"+   
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


def get_identity(args):

    if args.dataset_name in ['ogbn-arxiv', 'ogbn-proteins', 'cora', 'pubmed', 'citeseer']:
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
    elif args.dataset_name in ['ogbl-collab', 'ogbl-ddi']:
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
    elif 'ogbg' or 'imdb' in args.dataset_name or args.dataset_name in ['zinc']:
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
    return args

def add_args(args):
    args = get_identity(args)

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


def get_metric(args):

    if args.dataset_name in ['ogbg-molhiv','ogbg-molbace',
                            'ogbg-molbbbp','ogbg-molclintox',
                            'ogbg-molsider','ogbg-moltox21',
                            'ogbg-moltoxcast',
                            ]:
        return 'rocauc'
    elif args.dataset_name in ['ogbg-molpcba','ogbg-molmuv']:
        return 'ap'
    elif args.dataset_name in ['ogbg-molesol','ogbg-molfreesolv',
                            'ogbg-mollipo']:
        return 'rmse'
    elif 'ogbg-ppa' or 'imdb' in args.dataset_name:
        return 'acc'


def get_task_type(args):

    if args.dataset_name in ['ogbg-molpcba','ogbg-ppa', 'ogbg-moltoxcast','imdb-m',
                            'ogbn-arxiv','ogbn-proteins',
                            ]:
        return 'classification'
    else:
        return 'regression'


def smoothing(array, width):
    length = len(array)
    output = np.zeros([length], dtype=float)

    ind_begin = 0
    for i in range(length):
        ind_end = i + 1
        if ind_end > width:
            ind_begin = ind_end - width
        output[i] = array[ind_begin:ind_end].mean()
    return output



