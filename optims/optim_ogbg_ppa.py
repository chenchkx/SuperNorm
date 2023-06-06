
import os
import time
import torch
import pandas as pd
import torch.nn as nn
import dgl.function as fn
from tqdm import tqdm
from cmath import inf
from ogb.graphproppred import Evaluator

cls_criterion = nn.BCEWithLogitsLoss()
reg_criterion = nn.MSELoss()
multicls_criterion = torch.nn.CrossEntropyLoss()

# class of model optimizing & learning (ModelOptLearning) for ogb graph(ogbg) property prediction
class ModelOptLearning_OGBG_PPA:
    def __init__(self, model, optimizer, scheduler,
                train_loader, valid_loader, test_loader,
                args):
        # initizing ModelOptLearning class
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.evaluator = Evaluator(args.dataset_name)        
        self.args = args
        
    @torch.no_grad()
    def eval(self, model, loader):
        # torch.cuda.empty_cache()
        model.eval()
        total, total_loss = 0, 0  
        y_true, y_pred = [], []
        
        for graphs, labels in loader:    
            graphs, labels = graphs.to(self.args.device), labels.to(self.args.device)
            graphs.update_all(fn.copy_e('feat', 'e'), fn.mean('e', 'feat'))
            nfeats = graphs.ndata['feat']
            efeats = graphs.edata['feat']
            with torch.no_grad():
                outputs = model(graphs, nfeats, efeats)
            y_true.append(labels.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(outputs.detach(), dim = 1).view(-1,1).cpu())

            total += len(labels)

            loss = multicls_criterion(outputs.to(torch.float32), labels.view(-1,))

            total_loss += loss * len(labels)
                    
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        input_dict = {'y_true': y_true, 'y_pred': y_pred}
        # eval results 
        rst = {}
        rst['loss'] = (1.0 * total_loss / total).item()
        rst[self.args.eval_metric] = self.evaluator.eval(input_dict)[self.args.eval_metric]

        return rst


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

        
    def optimizing(self):
                 
        valid_best_cls = 0
        valid_best_reg = inf
        logs_table = pd.DataFrame()
        train_stas_table = pd.DataFrame()
        valid_stas_table = pd.DataFrame()
        test_stas_table = pd.DataFrame()

        epochs_no_improve = 0  # used for early stopping
        for epoch in range(self.args.epochs):
            # training model 
            self.model.train()
            t0 = time.time()
            for _, batch in enumerate(tqdm(self.train_loader, desc='Iteration')):              
            # for graphs, labels in self.train_loader:
                graphs, labels = batch
                graphs, labels = graphs.to(self.args.device), labels.to(self.args.device)

                graphs.update_all(fn.copy_e('feat', 'e'), fn.mean('e', 'feat'))
                nfeats = graphs.ndata['feat']
                efeats = graphs.edata['feat']

                outputs = self.model(graphs, nfeats, efeats)
                self.optimizer.zero_grad()

                loss = multicls_criterion(outputs.to(torch.float32), labels.view(-1,))
                
                loss.backward()
                self.optimizer.step()

            # eval the performance of the model
            train_rst = self.eval(self.model, self.train_loader)
            valid_rst = self.eval(self.model, self.valid_loader)
            test_rst = self.eval(self.model, self.test_loader)  

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

