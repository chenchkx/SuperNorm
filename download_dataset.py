#!/usr/bin/python3

import os
import dgl
import time
import torch
import argparse
import dgl.data
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset
from ogb.graphproppred import DglGraphPropPredDataset
from datasets.dgl_imdb_dataset import DglIMDBDataset
from datasets.dgl_zinc_dataset import DglZincDataset
from datasets.dgl_planetoid_dataset import DglPlanetoidDataset
from datasets.preprocess import data_preprocess
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


dir_path = os.path.dirname(__file__)
nfs_dataset_path1 = '/nfs4-p1/ckx/datasets/'
nfs_dataset_path2 = '/mnt/nfs4-p1/ckx/datasets/'
# torch.set_num_threads(10) 

def main(args):

    # check nfs dataset path
    if os.path.exists(nfs_dataset_path1):
        args.dataset_path = nfs_dataset_path1
    elif os.path.exists(nfs_dataset_path2):
        args.dataset_path = nfs_dataset_path2
    
    start_time = time.time()
    # download dataset
    
    if 'ogbn' in args.dataset_name:
        dataset = DglNodePropPredDataset(name=args.dataset_name, root=args.dataset_path)
    elif 'ogbg' in args.dataset_name:
        dataset = DglGraphPropPredDataset(name=args.dataset_name, root=args.dataset_path)
    elif 'ogbl' in args.dataset_name:
        dataset = DglLinkPropPredDataset(name=args.dataset_name, root=args.dataset_path)
    elif args.dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = DglPlanetoidDataset(name=args.dataset_name, root=args.dataset_path)
    elif args.dataset_name == 'zinc':
        dataset = DglZincDataset(name=args.dataset_name, root=args.dataset_path)
    elif args.dataset_name in ['imdb-multi','imdb-binary']:
        dataset = DglIMDBDataset(name=args.dataset_name, root=args.dataset_path)

    # data preprocessing 
    dataset = data_preprocess(args, dataset, bool_preprocessed=True)

    end_time = time.time()

    print('- ' * 30)
    print(f'{args.dataset_name} dataset loaded, preprocessing {end_time-start_time} seconds.')
    print('- ' * 30)   


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    ## datasets path and name
    parser.add_argument("--dataset_path", type=str, default='datasets')
    parser.add_argument("--dataset_name", type=str, default='ogbl-collab')
    ## graph-level dataset
    # zinc
    # imdb-binary
    # imdb-multi
    # ogbg-mollipo
    # ogbg-moltoxcast
    # ogbg-molpcba
    # ogbg-ppa

    ## node-level dataset
    # ogbn-proteins
    # cora
    # citeseer
    # pubmed

    ## link-level dataset
    # ogbl-collab

    args = parser.parse_args()
    main(args)
