import os
import dgl
import torch
from dgl.data.utils import load_graphs, save_graphs
from ogb.nodeproppred import DglNodePropPredDataset
from torch_geometric.datasets import Planetoid


def pyg_to_dgl(dataset, root_file):
    label_dict = {}
    u, v = dataset.data.edge_index
    graph = dgl.graph((u, v))
    graph.ndata['feat'] = dataset.data.x
    label_dict['labels']=dataset.data.y
    save_graphs(root_file, graph, label_dict)
    return graph, label_dict

# DGL Cora  CiteSeer  PubMed datasets 
class DglPlanetoidDataset(DglNodePropPredDataset):
    def __init__(self, name, root = 'dataset', meta_dict = None): 

        self.name = name.lower()
        self.root = os.path.join(root, self.name)
        self.task_type = 'classification'
        self.num_tasks = 1
        self.eval_metric = 'acc'

        data_preprocessed = os.path.join(self.root, 'processed', 'dgl_data_processed')
        if not os.path.exists(data_preprocessed):
            dataset = Planetoid(name=self.name, root=self.root)
            pyg_to_dgl(dataset, data_preprocessed)

        graph, label_dict = load_graphs(data_preprocessed)

        self.graph, self.labels = graph, label_dict['labels'].view([-1,1])
        
        self.num_classes = self.labels.max().item()+1

    def get_idx_split(self, split_type = None):


        if self.name == 'cora':
            train_idx = list(range(0, 139))
            valid_idx = list(range(140, 639))
            test_idx = list(range(1708, 2707))
        elif self.name == 'citeseer':
            train_idx = list(range(0, 119))
            valid_idx = list(range(120, 619))
            test_idx = list(range(2312, 3326))
        elif self.name == 'pubmed':
            train_idx = list(range(0, 59))
            valid_idx = list(range(60, 559))
            test_idx = list(range(18717, 19716))
            
        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}
