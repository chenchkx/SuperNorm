
import os
import torch
from dgl.data.utils import load_graphs, save_graphs, Subset
from ogb.graphproppred.dataset_dgl import DglGraphPropPredDataset

class DglZincDataset(DglGraphPropPredDataset):
    def __init__(self, name, root = 'dataset', meta_dict = None): 

        self.name = name
        self.root = 'datasets/zinc/' 
        self.num_atom_type = 28
        self.num_bond_type = 4
        self.task_type = 'regression'
        self.num_tasks = 1
        self.eval_metric = 'mae'
        self.graphs, label_dict = load_graphs(os.path.join(self.root, 'processed', 'dgl_data_processed'))
        self.labels = label_dict['labels']

    def get_idx_split(self, split_type = None):
        train_idx = list(range(0, 10000))
        valid_idx = list(range(10000,11000))
        test_idx = list(range(11000,12000))

        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}
