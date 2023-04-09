
import os
import dgl
import time
import torch
import pandas as pd
import torch.nn as nn
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modules.norm.norms import NormalizeGNN
# class of model optimizing & learning (ModelOptLearning) for ogb link(ogbl) property prediction
class ModelOptLearning_OGBL_DDI():
    def __init__(self, model, predictor, args):
        # initizing ModelOptLearning class
        self.model = model
        self.predictor = predictor
        self.embedding = nn.Embedding(args.num_nodes, args.embed_dim).to(args.device)
        self.norm = NormalizeGNN(args.norm_type, args.embed_dim, affine=args.norm_affine).to(args.device)

        self.optimizer = torch.optim.Adam(list(self.model.parameters())+list(self.predictor.parameters())+list(self.embedding.parameters()), 
                                          lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=args.lr_patience, verbose=True)   

        self.evaluator = Evaluator(args.dataset_name)        
        self.args = args

    @torch.no_grad()
    def eval(self, graph, nfeat, efeat, split_edge, edge_weight=None):
        # torch.cuda.empty_cache()
        self.model.eval()
        self.predictor.eval()

        h = self.model(graph, nfeat, efeat, edge_weight=edge_weight)

        pos_train_edge = split_edge['train']['edge'].to(h.device)
        pos_valid_edge = split_edge['valid']['edge'].to(h.device)
        neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
        pos_test_edge = split_edge['test']['edge'].to(h.device)
        neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

        pos_train_preds = []
        train_loss = train_examples = 0
        for perm in DataLoader(range(pos_train_edge.size(0)), self.args.batch_size):
            edge = pos_train_edge[perm].t()
            
            pos_out = self.predictor(h[edge[0]], h[edge[1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()
            # Just do some trivial random sampling.
            edge = torch.randint(0, graph.batch_num_nodes().item(), edge.size(), dtype=torch.long,
                                device=h.device)
            neg_out = self.predictor(h[edge[0]], h[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            num_examples = pos_out.size(0)
            train_loss += (pos_loss + neg_loss).item() * num_examples
            train_examples += num_examples

            pos_train_preds += [pos_out.squeeze().cpu()]

        pos_train_pred = torch.cat(pos_train_preds, dim=0)
        train_loss = train_loss/num_examples

        pos_valid_preds = []
        valid_loss = valid_examples = 0
        for perm in DataLoader(range(pos_valid_edge.size(0)), self.args.batch_size):
            edge = pos_valid_edge[perm].t()

            pos_out = self.predictor(h[edge[0]], h[edge[1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()
            # Just do some trivial random sampling.
            edge = torch.randint(0, graph.batch_num_nodes().item(), edge.size(), dtype=torch.long,
                                device=h.device)
            neg_out = self.predictor(h[edge[0]], h[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            num_examples = pos_out.size(0)
            valid_loss += (pos_loss + neg_loss).item() * num_examples
            valid_examples += num_examples

            pos_valid_preds += [pos_out.squeeze().cpu()]

        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
        valid_loss = valid_loss/valid_examples

        neg_valid_preds = []
        for perm in DataLoader(range(neg_valid_edge.size(0)), self.args.batch_size):
            edge = neg_valid_edge[perm].t()
            neg_valid_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)


        pos_test_preds = []
        test_loss = test_examples = 0
        for perm in DataLoader(range(pos_test_edge.size(0)), self.args.batch_size):
            edge = pos_test_edge[perm].t()

            pos_out = self.predictor(h[edge[0]], h[edge[1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()
            # Just do some trivial random sampling.
            edge = torch.randint(0, graph.batch_num_nodes().item(), edge.size(), dtype=torch.long,
                                device=h.device)
            neg_out = self.predictor(h[edge[0]], h[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            num_examples = pos_out.size(0)
            test_loss += (pos_loss + neg_loss).item() * num_examples
            test_examples += num_examples

            pos_test_preds += [pos_out.squeeze().cpu()]

        pos_test_pred = torch.cat(pos_test_preds, dim=0)
        test_loss = test_loss/test_examples


        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), self.args.batch_size):
            edge = neg_test_edge[perm].t()
            neg_test_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)


        self.evaluator.K = 50
        train_hits = self.evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{50}']
        valid_hits = self.evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{50}']
        test_hits = self.evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{50}']

        train_rst = {}
        train_rst['loss'] = train_loss
        train_rst[self.args.eval_metric] = train_hits

        valid_rst = {}
        valid_rst['loss'] = valid_loss
        valid_rst[self.args.eval_metric] = valid_hits

        test_rst = {}
        test_rst['loss'] = test_loss
        test_rst[self.args.eval_metric] = test_hits  

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


    def optimizing(self, dataset, split_edge):
                 
        valid_best_cls = 0
        logs_table = pd.DataFrame()

        graph = dataset.graph[0].to(self.args.device)

        nfeat = nn.init.xavier_uniform_(self.embedding.weight)    
        nfeat = self.embedding.weight
        # nfeat = self.norm(graph, self.embedding.weight)     

        efeat = []

        for epoch in range(self.args.epochs):

            t0 = time.time()
            # training model 
            self.model.train()
            self.predictor.train()

            pos_train_edge = split_edge['train']['edge'].to(nfeat.device)
            total_loss = total_examples = 0
            for perm in DataLoader(range(pos_train_edge.size(0)), self.args.batch_size, shuffle=True):
                self.optimizer.zero_grad()

                outputs = self.model(graph, nfeat, efeat)

                edge = pos_train_edge[perm].t()
                pos_out = self.predictor(outputs[edge[0]], outputs[edge[1]])
                pos_loss = -torch.log(pos_out + 1e-15).mean()

                # Just do some trivial random sampling.
                edge = torch.randint(0, graph.batch_num_nodes().item(), edge.size(), dtype=torch.long,
                                    device=outputs.device)
                neg_out = self.predictor(outputs[edge[0]], outputs[edge[1]])
                neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

                loss = pos_loss + neg_loss
                loss.backward()      
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
                self.optimizer.step()
                num_examples = pos_out.size(0)
                total_loss += loss.item() * num_examples
                total_examples += num_examples

            # eval the performance of the model
            train_rst, valid_rst, test_rst = self.eval(graph, nfeat, efeat, split_edge)

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
        
            is_best_valid = bool((self.args.state_dict) & (valid_best_cls < valid_perf) & (self.args.epoch_slice < epoch))
            valid_best_cls = valid_perf
            if is_best_valid:
                if not os.path.exists(self.args.perf_dict_dir):
                    os.mkdir(self.args.perf_dict_dir)
                dict_file_path = os.path.join(self.args.perf_dict_dir, self.args.identity+'.pth')
                torch.save(self.model.state_dict(), dict_file_path)


            self.scheduler.step(valid_rst['loss'])

            if self.optimizer.param_groups[0]['lr']<self.args.lr_min and self.args.breakout:
                print("\n!! LR LESS THAN THE MIN LR SET.")
                break

        if not os.path.exists(self.args.perf_xlsx_dir):
            os.mkdir(self.args.perf_xlsx_dir)
        logs_table.to_excel(os.path.join(self.args.perf_xlsx_dir, self.args.identity+'.xlsx'))
        if not os.path.exists(self.args.stas_xlsx_dir):
            os.mkdir(self.args.stas_xlsx_dir)


