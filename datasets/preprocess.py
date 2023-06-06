
import os
import dgl
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from warnings import simplefilter
from dgl.data.utils import save_graphs
from scipy.sparse import csr_array
simplefilter(action='ignore', category=FutureWarning)
torch.set_num_threads(15)

dir_path = os.path.dirname(__file__)


### preprocess dataset for graph level prediction
def preprocess_graph(args, dataset, bool_preprocessed=False):
    
    folder_preprocessed =  os.path.join(dataset.root, 'processed')
    if not os.path.exists(folder_preprocessed):
        os.mkdir(folder_preprocessed)    
    data_preprocessed = os.path.join(dataset.root, 'processed', 'dgl_data_processed')

    # preprocess the node features in ogbg-ppa dataset
    if 'ppa' in args.dataset_name:
        bool_preprocessed = True
        print('Initialing node feature in graphs ...')
        for g in tqdm(dataset):
            g[0].ndata['feat'] = torch.zeros(g[0].num_nodes(), dtype=int)

    # preprocess node weight in graph
    if not 'sg_factor' in dataset.graphs[0].ndata.keys() or bool_preprocessed:  
        bool_preprocessed = True
        print('Preprocess subgraph information in graph ...')
        for g in tqdm(dataset):
            # coefficients in terms of mean value
            graph = g[0]
            row, col = graph.edges()
            num_nodes = graph.num_nodes()
            adj = torch.zeros(num_nodes, num_nodes)
            for i in range(row.shape[0]):
                adj[row[i]][col[i]]=1.0           

            A_array = adj.detach().numpy()
                
            G = nx.from_numpy_array(A_array)
            sg_factor = torch.zeros(num_nodes,1)
            p_constant = 0.01
            
            for i in range(len(A_array)):
                s_indexes = []
                for j in range(len(A_array)):
                    s_indexes.append(i)
                    if(A_array[i][j]==1):
                        s_indexes.append(j)      
                subgraph_nodes = list(G.subgraph(s_indexes).nodes)
                subgraph_edges = G.subgraph(s_indexes).edges()
 
                if len(subgraph_nodes) == 1:
                    sg_factor[i] = 1
                else:
                    sg_adj = torch.zeros(len(subgraph_nodes), len(subgraph_nodes))
                    for edge in subgraph_edges:
                        head_index = subgraph_nodes.index(edge[0])
                        tail_index = subgraph_nodes.index(edge[1])
                        sg_adj[head_index, tail_index] = 1.0
                        sg_adj[tail_index, head_index] = 1.0

                    U, S, V = torch.tensor(sg_adj).svd()
                    eig_factor = 0
                    for eig_th in range(len(S)):
                        q = S[eig_th].item()
                        eig_factor = eig_factor + q*p_constant**eig_th  

                    sg_dense = 2*len(subgraph_edges)/(len(subgraph_nodes)*(len(subgraph_nodes)-1))

                    sg_factor[i] = sg_dense*len(subgraph_nodes)*len(subgraph_nodes) + eig_factor*p_constant**1 + len(subgraph_nodes)*p_constant**2 

            g[0].ndata['square_n'] = torch.FloatTensor(num_nodes).fill_(1/num_nodes**0.5) 

            g[0].ndata['sg_factor'] = sg_factor
            g[0].ndata['sg_factor_norm'] = g[0].ndata['sg_factor']/g[0].ndata['sg_factor'].sum()


        if bool_preprocessed:
            save_graphs(data_preprocessed, dataset.graphs, labels={'labels': dataset.labels})

    return dataset


### preprocess dataset for node level prediction
def preprocess_node(args, dataset, bool_preprocessed=False):

    folder_preprocessed =  os.path.join(dataset.root, 'processed')
    if not os.path.exists(folder_preprocessed):
        os.mkdir(folder_preprocessed)    
    data_preprocessed = os.path.join(dataset.root, 'processed', 'dgl_data_processed')

    if not 'sg_factor' in dataset.graph[0].ndata.keys() or bool_preprocessed:  
        bool_preprocessed = True
        print('Preprocess subgraph information in graph ...')
        
        graph, labels = dataset[0]
        if 'arxiv' in args.dataset_name:
            graph_undirected = dgl.add_reverse_edges(graph) # add reverse edges
        else: 
            graph_undirected = graph

        row, col = graph_undirected.edges()
        num_nodes = graph_undirected.num_nodes()

        sg_factor = torch.zeros(num_nodes,1)
        p_constant = 0.01

        for i in tqdm(range(num_nodes)):
            s_indexes = []
            s_indexes.append(i)
            s_indexes.extend(list(col[torch.where(row==i)].numpy())) 
            subgraph_nodes = s_indexes
            subgraph_edges = dgl.node_subgraph(graph,subgraph_nodes).edges()

            if len(subgraph_nodes) == 1:
                sg_factor[i] = 1
            else:
                sg_adj = torch.zeros(len(subgraph_nodes), len(subgraph_nodes))
                for edge_th in range(len(subgraph_edges[0])):
                    sg_adj[int(subgraph_edges[0][edge_th]), int(subgraph_edges[1][edge_th])] = 1.0

                sg_adj = sg_adj + 0.001*torch.eye(len(subgraph_nodes), len(subgraph_nodes))
                U, S, V = torch.tensor(sg_adj).svd()
                eig_factor = 0
                for eig_th in range(len(S)):
                    q = S[eig_th].item()
                    eig_factor = eig_factor + q*p_constant**eig_th  

                sg_dense = len(subgraph_edges[0])/(len(subgraph_nodes)*(len(subgraph_nodes)-1))

                sg_factor[i] = sg_dense*len(subgraph_nodes)*len(subgraph_nodes) + eig_factor*p_constant**1 + len(subgraph_nodes)*p_constant**2
                # sg_factor[i] = sg_dense*len(subgraph_nodes) + eig_factor*p_constant**1 + len(subgraph_nodes)*p_constant**2

        graph.ndata['square_n'] = torch.FloatTensor(num_nodes).fill_(1/num_nodes**0.5) 

        graph.ndata['sg_factor'] = sg_factor
        graph.ndata['sg_factor_norm'] = graph.ndata['sg_factor']/graph.ndata['sg_factor'].sum()


        # row, col = graph_undirected.edges()
        # num_nodes = graph_undirected.num_nodes()
        # adj = torch.zeros(num_nodes, num_nodes)
        # for i in range(row.shape[0]):
        #     adj[row[i]][col[i]]=1.0           

        # A_array = adj.detach().numpy()
        # print('test')
        # G = nx.from_numpy_array(A_array)
        # sg_factor = torch.zeros(num_nodes,1)
        # p_constant = 0.01

        # for i in tqdm(range(len(A_array))):

        #     s_indexes = []
        #     for j in range(len(A_array)):
        #         s_indexes.append(i)
        #         if(A_array[i][j]==1):
        #             s_indexes.append(j)      
        #     subgraph_nodes = list(G.subgraph(s_indexes).nodes)
        #     subgraph_edges = G.subgraph(s_indexes).edges()

        #     if len(subgraph_nodes) == 1:
        #         sg_factor[i] = 1
        #     else:
        #         sg_adj = torch.zeros(len(subgraph_nodes), len(subgraph_nodes))
        #         for edge in subgraph_edges:
        #             head_index = subgraph_nodes.index(edge[0])
        #             tail_index = subgraph_nodes.index(edge[1])
        #             sg_adj[head_index, tail_index] = 1.0
        #             sg_adj[tail_index, head_index] = 1.0

        #         U, S, V = torch.tensor(sg_adj).svd()
        #         eig_factor = 0
        #         for eig_th in range(len(S)):
        #             q = S[eig_th].item()
        #             eig_factor = eig_factor + q*p_constant**eig_th  

        #         sg_dense = 2*len(subgraph_edges)/(len(subgraph_nodes)*(len(subgraph_nodes)-1))

        #         sg_factor[i] = sg_dense*len(subgraph_nodes)*len(subgraph_nodes) + eig_factor*p_constant**1 + len(subgraph_nodes)*p_constant**2
        #         # sg_factor[i] = sg_dense*len(subgraph_nodes) + eig_factor*p_constant**1 + len(subgraph_nodes)*p_constant**2

        # graph.ndata['square_n'] = torch.FloatTensor(num_nodes).fill_(1/num_nodes**0.5) 

        # graph.ndata['sg_factor'] = sg_factor
        # graph.ndata['sg_factor_norm'] = graph.ndata['sg_factor']/graph.ndata['sg_factor'].sum()

    if bool_preprocessed:
        save_graphs(data_preprocessed, graph, labels={'labels': dataset.labels})

    return dataset


### preprocess dataset for node level prediction
def preprocess_link(args, dataset, bool_preprocessed=False):

    folder_preprocessed =  os.path.join(dataset.root, 'processed')
    if not os.path.exists(folder_preprocessed):
        os.mkdir(folder_preprocessed)    
    data_preprocessed = os.path.join(dataset.root, 'processed', 'dgl_data_processed')

    if not 'motif_factor' in dataset.graph[0].ndata.keys() or bool_preprocessed:  
        bool_preprocessed = True
        print('Preprocess subgraph information in graph ...')
        
        graph = dataset.graph[0]
        if args.dataset_name in ['ogbl-citation2']:
            graph_undirected = dgl.add_reverse_edges(graph) # add reverse edges
        else: 
            graph_undirected = graph # add reverse edges
        row, col = graph_undirected.edges()
        num_nodes = graph_undirected.num_nodes()
        adj = torch.zeros(num_nodes, num_nodes)
        for i in range(row.shape[0]):
            adj[row[i]][col[i]]=1.0           

        A_array = adj.detach().numpy()
        G = nx.from_numpy_array(A_array)
        sg_nodes = torch.zeros(num_nodes,1)
        sg_edges = torch.zeros(num_nodes,1)
        sg_dense = torch.zeros(num_nodes,1)
        for i in tqdm(range(len(A_array))):
            s_indexes = []
            for j in range(len(A_array)):
                s_indexes.append(i)
                if(A_array[i][j]==1):
                    s_indexes.append(j)      
            subgraph_nodes = len(list(G.subgraph(s_indexes).nodes))
            subgraph_edges = G.subgraph(s_indexes).number_of_edges()
            # subgraph information 
            if subgraph_nodes == 1:
                sg_nodes[i] = 1
                sg_edges[i] = 1
                sg_dense[i] = 1
            else:
                sg_nodes[i] = subgraph_nodes
                sg_edges[i] = subgraph_edges
                sg_dense[i] = 2*subgraph_edges/(subgraph_nodes*(subgraph_nodes-1))
        
        graph.ndata['square_n'] = torch.FloatTensor(num_nodes).fill_(1/num_nodes**0.5)     

        graph.ndata['motif_factor'] = sg_dense*sg_nodes*sg_nodes
        graph.ndata['motif_factor_norm'] = graph.ndata['motif_factor']/graph.ndata['motif_factor'].sum()

    if bool_preprocessed:
        save_graphs(data_preprocessed, graph, {})

    return dataset


def data_preprocess(args, dataset, bool_preprocessed=True):
    
    if 'ogbg' in args.dataset_name or args.dataset_name in ['zinc', 'imdb-binary']:
        dataset = preprocess_graph(args, dataset, bool_preprocessed=bool_preprocessed)
    if 'ogbn' in args.dataset_name or args.dataset_name in ['cora','citeseer','pubmed']:  
        dataset = preprocess_node(args, dataset, bool_preprocessed=bool_preprocessed)
    if 'ogbl' in args.dataset_name:
        dataset = preprocess_link(args, dataset, bool_preprocessed=bool_preprocessed)

    return dataset