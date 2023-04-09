
import os
import dgl
import time
import torch
import pandas as pd
import torch.nn.functional as F
from cmath import inf
from ogb.nodeproppred import Evaluator

from modules.norm.norm_graph import GraphNorm

# Multi-class Task. Metric: Accuracy
# including OGBN Multi-class datasets and Planetoid datasets
class ModelOptLearning_OGBN_Acc:
    def __init__(self, model, optimizer, scheduler, args):
        # initizing ModelOptLearning class
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.evaluator = Evaluator('ogbn-arxiv')        
        self.args = args

        if args.dataset_name in ['ogbn-arxiv']:
            # add reverse edges because ogbn-arxiv dataset is a directed graph
            self.add_reverse_edges = True 
        else: 
            self.add_reverse_edges = False

    @torch.no_grad()
    def eval(self, graph, labels, nfeat, efeat, split_idx):
        # torch.cuda.empty_cache()
        self.model.eval()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        outputs = self.model(graph, nfeat, efeat)
        outputs = outputs.log_softmax(dim=-1)
        y_pred = outputs.argmax(dim=-1, keepdim=True)

        train_rst, valid_rst, test_rst = {},{},{} 
        train_rst['loss'] = F.nll_loss(outputs[train_idx], labels.squeeze(1)[train_idx]).item()
        train_rst[self.args.eval_metric] = self.evaluator.eval({
                                            'y_true': labels[train_idx],
                                            'y_pred': y_pred[train_idx],
                                            })['acc']

        valid_rst['loss'] = F.nll_loss(outputs[valid_idx], labels.squeeze(1)[valid_idx]).item()
        valid_rst[self.args.eval_metric] = self.evaluator.eval({
                                            'y_true': labels[valid_idx],
                                            'y_pred': y_pred[valid_idx],
                                            })['acc']

        test_rst['loss'] = F.nll_loss(outputs[test_idx], labels.squeeze(1)[test_idx]).item()
        test_rst[self.args.eval_metric] = self.evaluator.eval({
                                            'y_true': labels[test_idx],
                                            'y_pred': y_pred[test_idx],
                                            })['acc']
        return train_rst, valid_rst, test_rst


    def log_epoch(self, logs_table, train_rst, valid_rst, test_rst, log_lr, log_time):
        table_head = []
        table_data = []
        for keys in train_rst.keys():
            table_head.append(f'train-{keys}')
            table_data.append(train_rst[keys])
        for keys in valid_rst.keys():
            table_head.append(f'valid-{keys}')
            table_data.append(valid_rst[keys])
        for keys in test_rst.keys():
            table_head.append(f'test-{keys}')
            table_data.append(test_rst[keys])
        for keys in log_lr.keys():
            table_head.append(f'{keys}')
            table_data.append(log_lr[keys])
        for keys in log_time.keys():
            table_head.append(f'{keys}')
            table_data.append(log_time[keys])
        
        return logs_table.append(pd.DataFrame([table_data], columns=table_head), ignore_index=True)

   
    def optimizing(self, dataset, split_idx):
                 
        valid_best_cls = 0
        valid_best_reg = inf
        logs_table = pd.DataFrame()

        graph, labels = dataset.graph[0].to(self.args.device), dataset.labels.to(self.args.device) 
        graph = dgl.add_reverse_edges(graph) if self.add_reverse_edges else graph

        nfeat = graph.ndata['feat']
        efeat = graph.edata['feat'] if len(graph.edata) else torch.zeros(len(graph.edges()[0]), nfeat.shape[1]).to(self.args.device) 

        for epoch in range(self.args.epochs):

            t0 = time.time()
            # training model 
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(graph, nfeat, efeat)[split_idx["train"]] 
            outputs = outputs.log_softmax(dim=-1)
            loss = F.nll_loss(outputs, labels.squeeze(1)[split_idx["train"]])
            loss.backward()
            
            self.optimizer.step()

            # model eval
            train_rst, valid_rst, test_rst=self.eval(graph, labels, nfeat, efeat, split_idx)

            train_loss = train_rst['loss']
            train_perf = train_rst[self.args.eval_metric]
            valid_perf = valid_rst[self.args.eval_metric]
            test_perf = test_rst[self.args.eval_metric]

            eopch_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            eopch_time = time.time() - t0
            log_lr = {'lr': eopch_lr}
            log_time = {'time': eopch_time}
            print(f"epoch: {epoch}, train_loss {train_loss:.6f}, train perf {train_perf:.6f}, valid perf {valid_perf:.6f}, test perf {test_perf:.6f}, {eopch_lr}, {eopch_time:.2f}")
            logs_table = self.log_epoch(logs_table, train_rst, valid_rst, test_rst, log_lr, log_time)
        

            if "classification" in self.args.task_type:
                is_best_valid = bool((self.args.state_dict) & (valid_best_cls < valid_perf) & (self.args.epoch_slice < epoch))
                valid_best_cls = valid_perf
            else: 
                is_best_valid = bool((self.args.state_dict) & (valid_best_reg > valid_perf) & (self.args.epoch_slice < epoch))
                valid_best_reg = valid_perf
            if is_best_valid:
                if not os.path.exists(self.args.perf_dict_dir):
                    os.mkdir(self.args.perf_dict_dir)
                dict_file_path = os.path.join(self.args.perf_dict_dir, self.args.identity+'.pth')
                torch.save(self.model.state_dict(), dict_file_path)

            self.scheduler.step(valid_rst['loss'])

            if self.optimizer.param_groups[0]['lr']<self.args.lr_min and self.args.breakout:
                print("\n!! LR LESS THAN THE MIN LR SET.")
                break
            
        representation = self.model.representation
        row_dis_list = []
        col_dis_list = []
        for i in range(len(representation)):
            tensor = representation[i][split_idx["train"]] 
            row, col = tensor.size()
            row_dis = 0
            for row_i in range(row):
                row_i_rep = tensor[row_i,:]
                for row_j in range(row_i, row):
                    row_j_rep = tensor[row_j,:]
                    row_dis = row_dis+torch.pow(row_i_rep-row_j_rep,2).sum()/row_j_rep.size()[0]
            col_dis = 0
            for col_i in range(col):
                col_i_rep = tensor[col_i,:]
                for col_j in range(col_i, col):
                    col_j_rep = tensor[col_j,:]
                    col_dis = col_dis+torch.pow(col_i_rep-col_j_rep,2).sum()/col_j_rep.size()[0]
            row_dis_list.append(row_dis/row)
            col_dis_list.append(col_dis/col)

        row_similar_list = []
        col_similar_list = []
        for i in range(len(representation)):
            tensor = representation[i][split_idx["train"]] 
            row_norm = F.normalize(tensor, dim=1)
            col_norm = F.normalize(tensor, dim=0)

            row_spd = row_norm.matmul(row_norm.T)
            col_spd = col_norm.matmul(col_norm.T)

            row_similar_list.append(row_spd.mean())
            col_similar_list.append(col_spd.mean())


        if not os.path.exists(self.args.perf_xlsx_dir):
            os.mkdir(self.args.perf_xlsx_dir)
        logs_table.to_excel(os.path.join(self.args.perf_xlsx_dir, self.args.identity+'.xlsx'))
        if not os.path.exists(self.args.stas_xlsx_dir):
            os.mkdir(self.args.stas_xlsx_dir)


