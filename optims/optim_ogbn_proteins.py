
import os
import dgl
import time
import torch
import pandas as pd
import torch.nn.functional as F
from cmath import inf
from ogb.nodeproppred import Evaluator

criterion = torch.nn.BCEWithLogitsLoss()

# class of model optimizing & learning (ModelOptLearning) for ogb node(ogbn)-proteins property prediction
class ModelOptLearning_OGBN_Proteins:
    def __init__(self, model, optimizer, scheduler, args):
        # initizing ModelOptLearning class
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.evaluator = Evaluator(args.dataset_name)        
        self.args = args

    @torch.no_grad()
    def eval(self, graph, labels, nfeat, efeat, split_idx):
        # torch.cuda.empty_cache()
        self.model.eval()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        outputs = self.model(graph, nfeat, efeat)
        y_pred = outputs

        train_rst, valid_rst, test_rst = {},{},{} 
        train_rst['loss'] = criterion(outputs[train_idx], labels[train_idx].to(torch.float)).item()
        train_rst[self.args.eval_metric] = self.evaluator.eval({
                                            'y_true': labels[train_idx],
                                            'y_pred': y_pred[train_idx],
                                            })['rocauc']

        valid_rst['loss'] = criterion(outputs[valid_idx], labels[valid_idx].to(torch.float)).item()
        valid_rst[self.args.eval_metric] = self.evaluator.eval({
                                            'y_true': labels[valid_idx],
                                            'y_pred': y_pred[valid_idx],
                                            })['rocauc']

        test_rst['loss'] = criterion(outputs[test_idx], labels[test_idx].to(torch.float)).item()
        test_rst[self.args.eval_metric] = self.evaluator.eval({
                                            'y_true': labels[test_idx],
                                            'y_pred': y_pred[test_idx],
                                            })['rocauc']
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
        nfeat = graph.ndata['feat']
        efeat = None

        for epoch in range(self.args.epochs):

            t0 = time.time()
            # training model 
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(graph, nfeat, efeat)[split_idx["train"]] 
            loss = criterion(outputs, labels[split_idx["train"]].to(torch.float))
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

        if not os.path.exists(self.args.perf_xlsx_dir):
            os.mkdir(self.args.perf_xlsx_dir)
        logs_table.to_excel(os.path.join(self.args.perf_xlsx_dir, self.args.identity+'.xlsx'))
        if not os.path.exists(self.args.stas_xlsx_dir):
            os.mkdir(self.args.stas_xlsx_dir)


