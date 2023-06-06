import os
import dgl
import torch
import random
import numpy as np
from tqdm import tqdm
from dgl.data.utils import load_graphs, save_graphs
from ogb.graphproppred import DglGraphPropPredDataset
from torch_geometric.datasets import TUDataset


def pyg_to_dgl(dataset, root_file):

    graphs = []
    labels = []
    tagset = set([])
    for i in range(len(dataset)):
        g = dataset[i]
        u, v = g.edge_index
        graph = dgl.graph((u, v))  
        graph.node_tags = graph.in_degrees().tolist()
        graphs.append(graph)
        labels.append(np.array(g.y))
        tagset = tagset.union(set(graph.node_tags))

    labels = torch.from_numpy(np.array(labels)).to(torch.long)
    label_dict = {}
    label_dict['labels']=labels
    save_graphs(root_file, graphs, labels={'labels': labels})
    return graphs, label_dict


# DGL IMDB-MULTI datasets 
class DglIMDBDataset(DglGraphPropPredDataset):
    def __init__(self, name, root = 'dataset', meta_dict = None): 

        self.name = name.upper()
        self.root = os.path.join(root, self.name)
        self.task_type = 'classification'

        data_preprocessed = os.path.join(self.root, 'processed', 'dgl_data_processed')
        if not os.path.exists(data_preprocessed):
            dataset = TUDataset(name=self.name, root=root)
            pyg_to_dgl(dataset, data_preprocessed)
        graphs, label_dict = load_graphs(data_preprocessed)

        # if not 'g_density' in graphs[0].ndata.keys():
        self.g_density = []
        self.n_numbers = []
        for g in tqdm(graphs):
            self.g_density.append(g.num_edges()/(g.num_nodes()*(g.num_nodes()-1)))
            self.n_numbers.append(g.in_degrees().float().mean())
        set(np.where((np.array(self.g_density)>0.4))[0])&set(np.where((np.array(self.g_density)<0.6))[0])

        self.graphs, self.labels = graphs, label_dict['labels'].view([-1,1])

        self.num_classes = self.labels.max().item()+1

        self.train_idx_path = os.path.join(self.root,'train_idx.npy')
        self.valid_idx_path = os.path.join(self.root,'valid_idx.npy')
        self.test_idx_path = os.path.join(self.root,'test_idx.npy')


    def get_idx_split(self, split_type = 0):

        if os.path.exists(self.train_idx_path):
            train_idx = list(np.load(self.train_idx_path))
            valid_idx = list(np.load(self.valid_idx_path))
            test_idx = list(np.load(self.test_idx_path))
        else:
            test_idx = []
            valid_idx = []
            index_1 = set(np.where((np.array(self.g_density)>0.1))[0])&set(np.where((np.array(self.g_density)<=0.2))[0])
            index_2 = set(np.where((np.array(self.g_density)>0.2))[0])&set(np.where((np.array(self.g_density)<=0.3))[0])
            index_3 = set(np.where((np.array(self.g_density)>0.3))[0])&set(np.where((np.array(self.g_density)<=0.4))[0])
            index_4 = set(np.where((np.array(self.g_density)>0.4))[0])&set(np.where((np.array(self.g_density)<=0.5))[0])
            index_5 = set(np.where((np.array(self.g_density)>0.5))[0])&set(np.where((np.array(self.g_density)<=0.6))[0])
            index_6 = set(np.where((np.array(self.g_density)>0.6))[0])&set(np.where((np.array(self.g_density)<=0.7))[0])
            index_7 = set(np.where((np.array(self.g_density)>0.7))[0])&set(np.where((np.array(self.g_density)<=0.8))[0])
            index_8 = set(np.where((np.array(self.g_density)>0.8))[0])&set(np.where((np.array(self.g_density)<=0.9))[0])
            index_9 = set(np.where((np.array(self.g_density)>0.9))[0])&set(np.where((np.array(self.g_density)<1.0))[0])
            index_10 = set(np.where(np.array(self.g_density)==1.0)[0])


            index_1_label0 = np.array(list(index_1))[np.where(self.labels[list(index_1)]==0)[0]]
            index_1_label1 = np.array(list(index_1))[np.where(self.labels[list(index_1)]==1)[0]]
            index_1_label0 = np.array(index_1_label0)[np.argsort(np.array(self.n_numbers)[list(index_1_label0)])]
            index_1_label1 = np.array(index_1_label1)[np.argsort(np.array(self.n_numbers)[list(index_1_label1)])]
            test_idx.extend(list(index_1_label0[range(0,9,9)]))
            test_idx.extend(list(index_1_label1[range(0,18,9)]))
            valid_idx.extend(list(index_1_label0[range(1,9,9)]))
            valid_idx.extend(list(index_1_label1[range(1,18,9)]))

            index_2_label0 = np.array(list(index_2))[np.where(self.labels[list(index_2)]==0)[0]]
            index_2_label1 = np.array(list(index_2))[np.where(self.labels[list(index_2)]==1)[0]]
            index_2_label0 = np.array(index_2_label0)[np.argsort(np.array(self.n_numbers)[list(index_2_label0)])]
            index_2_label1 = np.array(index_2_label1)[np.argsort(np.array(self.n_numbers)[list(index_2_label1)])]
            test_idx.extend(list(index_2_label0[range(0,56,8)]))
            test_idx.extend(list(index_2_label1[range(0,80,8)]))
            valid_idx.extend(list(index_2_label0[range(1,56,8)]))
            valid_idx.extend(list(index_2_label1[range(1,80,8)]))

            index_3_label0 = np.array(list(index_3))[np.where(self.labels[list(index_3)]==0)[0]]
            index_3_label1 = np.array(list(index_3))[np.where(self.labels[list(index_3)]==1)[0]]
            index_3_label1 = np.array(index_3_label1)[np.argsort(np.array(self.n_numbers)[list(index_3_label1)])]
            index_3_label1 = np.array(index_3_label1)[np.argsort(np.array(self.n_numbers)[list(index_3_label1)])]
            test_idx.extend(list(index_3_label0[range(0,84,7)]))
            test_idx.extend(list(index_3_label1[range(0,140,7)]))
            valid_idx.extend(list(index_3_label0[range(1,84,7)]))
            valid_idx.extend(list(index_3_label1[range(1,140,7)]))

            index_4_label0 = np.array(list(index_4))[np.where(self.labels[list(index_4)]==0)[0]]
            index_4_label1 = np.array(list(index_4))[np.where(self.labels[list(index_4)]==1)[0]]
            index_4_label0 = np.array(index_4_label0)[np.argsort(np.array(self.n_numbers)[list(index_4_label0)])]
            index_4_label1 = np.array(index_4_label1)[np.argsort(np.array(self.n_numbers)[list(index_4_label1)])]
            test_idx.extend(list(index_4_label0[range(0,48,6)]))
            test_idx.extend(list(index_4_label1[range(0,54,6)]))
            valid_idx.extend(list(index_4_label0[range(1,48,6)]))
            valid_idx.extend(list(index_4_label1[range(1,54,6)]))

            index_5_label0 = np.array(list(index_5))[np.where(self.labels[list(index_5)]==0)[0]]
            index_5_label1 = np.array(list(index_5))[np.where(self.labels[list(index_5)]==1)[0]]
            index_5_label0 = np.array(index_5_label0)[np.argsort(np.array(self.n_numbers)[list(index_5_label0)])]
            index_5_label1 = np.array(index_5_label1)[np.argsort(np.array(self.n_numbers)[list(index_5_label1)])]
            test_idx.extend(list(index_5_label0[range(0,115,5)]))
            test_idx.extend(list(index_5_label1[range(0,95,5)]))
            valid_idx.extend(list(index_5_label0[range(1,115,5)]))
            valid_idx.extend(list(index_5_label1[range(1,95,5)]))

            index_6_label0 = np.array(list(index_6))[np.where(self.labels[list(index_6)]==0)[0]]
            index_6_label1 = np.array(list(index_6))[np.where(self.labels[list(index_6)]==1)[0]]
            index_6_label0 = np.array(index_6_label0)[np.argsort(np.array(self.n_numbers)[list(index_6_label0)])]
            index_6_label1 = np.array(index_6_label1)[np.argsort(np.array(self.n_numbers)[list(index_6_label1)])]
            test_idx.extend(list(index_6_label0[range(0,52,4)]))
            test_idx.extend(list(index_6_label1[range(0,28,4)]))
            valid_idx.extend(list(index_6_label0[range(1,52,4)]))
            valid_idx.extend(list(index_6_label1[range(1,28,4)]))

            index_7_label0 = np.array(list(index_7))[np.where(self.labels[list(index_7)]==0)[0]]
            index_7_label1 = np.array(list(index_7))[np.where(self.labels[list(index_7)]==1)[0]]
            index_7_label0 = np.array(index_7_label0)[np.argsort(np.array(self.n_numbers)[list(index_7_label0)])]
            index_7_label1 = np.array(index_7_label1)[np.argsort(np.array(self.n_numbers)[list(index_7_label1)])]
            test_idx.extend(list(index_7_label0[range(0,12,3)]))
            test_idx.extend(list(index_7_label1[range(0,6,3)]))
            valid_idx.extend(list(index_7_label0[range(1,12,3)]))
            valid_idx.extend(list(index_7_label1[range(1,6,3)]))

            index_8_label0 = np.array(list(index_8))[np.where(self.labels[list(index_8)]==0)[0]]
            index_8_label0 = np.array(index_8_label0)[np.argsort(np.array(self.n_numbers)[list(index_8_label0)])]
            test_idx.extend(list(index_8_label0[range(0,14,2)]))
            valid_idx.extend(list(index_8_label0[range(1,14,2)]))

            index_9_label0 = np.array(list(index_9))[np.where(self.labels[list(index_9)]==0)[0]]
            index_9_label0 = np.array(index_9_label0)[np.argsort(np.array(self.n_numbers)[list(index_9_label0)])]
            test_idx.extend(list(index_9_label0[range(0,6,2)]))
            valid_idx.extend(list(index_9_label0[range(1,6,2)]))

            index_10_label0 = np.array(list(index_10))[np.where(self.labels[list(index_10)]==0)[0]]
            index_10_label1 = np.array(list(index_10))[np.where(self.labels[list(index_10)]==1)[0]]
            index_10_label0 = np.array(index_10_label0)[np.argsort(np.array(self.n_numbers)[list(index_10_label0)])]
            index_10_label1 = np.array(index_10_label1)[np.argsort(np.array(self.n_numbers)[list(index_10_label1)])]
            test_idx.extend(list(index_10_label0[range(0,78,6)]))
            test_idx.extend(list(index_10_label1[range(0,54,6)]))
            valid_idx.extend(list(index_10_label0[range(1,78,6)]))
            valid_idx.extend(list(index_10_label1[range(1,54,6)]))

            train_idx = list(set(range(1000))-(set(valid_idx)|set(test_idx)))

            np.save(self.train_idx_path, np.array(train_idx).astype(int))
            np.save(self.valid_idx_path, np.array(valid_idx).astype(int))
            np.save(self.test_idx_path, np.array(test_idx).astype(int))

        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}

