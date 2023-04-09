
import os
import dgl
import torch
import networkx as nx
from tqdm import tqdm
from warnings import simplefilter
from dgl.data.utils import save_graphs
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
    if not 'motif_factor' in dataset.graphs[0].ndata.keys() or bool_preprocessed:  
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
            G = nx.from_numpy_matrix(A_array)
            sg_nodes = torch.zeros(num_nodes,1)
            sg_edges = torch.zeros(num_nodes,1)
            sg_dense = torch.zeros(num_nodes,1)

            for i in range(len(A_array)):
                s_indexes = []
                for j in range(len(A_array)):
                    s_indexes.append(i)
                    if(A_array[i][j]==1):
                        s_indexes.append(j)      
                subgraph_nodes = len(list(G.subgraph(s_indexes).nodes))
                subgraph_edges = G.subgraph(s_indexes).number_of_edges()
                if subgraph_nodes == 1:
                    sg_nodes[i] = 1
                    sg_edges[i] = 1
                    sg_dense[i] = 1
                else:
                    sg_nodes[i] = subgraph_nodes
                    sg_edges[i] = subgraph_edges
                    sg_dense[i] = 2*subgraph_edges/(subgraph_nodes*(subgraph_nodes-1))

            g[0].ndata['square_n'] = torch.FloatTensor(num_nodes).fill_(1/num_nodes**0.5) 

            g[0].ndata['motif_factor'] = sg_dense*sg_nodes*sg_nodes
            g[0].ndata['motif_factor_norm'] = g[0].ndata['motif_factor']/g[0].ndata['motif_factor'].sum()

            if 'imdb' in args.dataset_name:
                g[0].ndata['motif_init'] = g[0].ndata['motif_factor']
                g[0].ndata['degree_init'] = sg_nodes


        if bool_preprocessed:
            save_graphs(data_preprocessed, dataset.graphs, labels={'labels': dataset.labels})

    return dataset


### preprocess dataset for node level prediction
def preprocess_node(args, dataset, bool_preprocessed=False):

    folder_preprocessed =  os.path.join(dataset.root, 'processed')
    if not os.path.exists(folder_preprocessed):
        os.mkdir(folder_preprocessed)    
    data_preprocessed = os.path.join(dataset.root, 'processed', 'dgl_data_processed')

    if not 'motif_factor' in dataset.graph[0].ndata.keys() or bool_preprocessed:  
        bool_preprocessed = True
        print('Preprocess subgraph information in graph ...')
        
        graph, labels = dataset[0]
        if 'arxiv' in args.dataset_name:
            graph_undirected = dgl.add_reverse_edges(graph) # add reverse edges
        else: 
            graph_undirected = graph # add reverse edges
        row, col = graph_undirected.edges()
        num_nodes = graph_undirected.num_nodes()
        adj = torch.zeros(num_nodes, num_nodes)
        for i in range(row.shape[0]):
            adj[row[i]][col[i]]=1.0           

        A_array = adj.detach().numpy()
        G = nx.from_numpy_matrix(A_array)
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
        G = nx.from_numpy_matrix(A_array)
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