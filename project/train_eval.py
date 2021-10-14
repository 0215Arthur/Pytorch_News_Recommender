# coding: UTF-8
from enum import EnumMeta
import os
import numpy as np
import torch
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import pickle
  
from torch.utils.data import Dataset,DataLoader
from lr_scheduler import GradualWarmupScheduler
import time
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from tools import get_time_dif,plot_loss,log_exec_time
from evaluation import ndcg_score,mrr_score,auc_score

import functools
from config import Config
from data_handler import *
# from model.nrms import Model
from model.GNUD import Model

from joblib import Parallel, delayed

def train(config, model, train_iter, dev_iter, news_iter ):    
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    #writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    
    AUC_best=0.56
    loss_list=[]
    STEP_SIZE=100
    improve='*'
    criterion =nn.CrossEntropyLoss()
    if config.warm_up:
        print('warm-up training...')
        scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config.warm_up_steps
                #after_scheduler=scheduler
            )
        optimizer.zero_grad()
        optimizer.step()
        
            #auc=evaluate(config,model,dev_iter,AUC_best,title_id_dict,abst_id_dict)
    # print('Epoch [{}/{}]'.format(epoch + 1, config.warm_up_epochs))
            # scheduler.step() # 学习率衰减
        loss_records=[]
        for i, datas in enumerate(train_iter):
            outputs = model(datas)
            # print(outputs)
            # break
            model.zero_grad()
            y = torch.zeros(len(outputs)).long().to(config.device)
            loss = criterion(outputs, y)
            #loss = torch.stack([x[0] for x in -F.log_softmax(outputs, dim=1)]).mean()
            #loss=torch.log(1+torch.exp(outputs[:,1:]-outputs[:,:1])).mean()
            loss_list.append(loss.item())
            loss_records.append(loss.item())
            loss.backward()
            optimizer.step()
            if i%100==0:
                time_dif = get_time_dif(start_time)
                msg = 'Warm-up Steps: {0:>6},  Train Loss: {1:>5.6},  Time: {2} {3}'
                print(msg.format(i, np.mean( loss_list), time_dif, improve))
                loss_list=[]
            if i>500:
                break
            scheduler.step(i)
           # evaluate(config,model,dev_iter,AUC_best)
            
           
    
    for epoch in range(config.num_epochs):
        #auc=evaluate(config,model,dev_iter,AUC_best,title_id_dict,abst_id_dict)
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        loss_records=[]
        # auc=evaluate(model,dev_iter, news_iter)
        eval_res=""
        for i, datas in enumerate(train_iter):
            
            outputs = model(datas)

            # print(outputs)
            # break
            model.zero_grad()
            y = torch.zeros(len(outputs)).long().to(config.device)
            loss = criterion(outputs, y)
            #loss = torch.stack([x[0] for x in -F.log_softmax(outputs, dim=1)]).mean()
            #loss=torch.log(1+torch.exp(outputs[:,1:]-outputs[:,:1])).mean()
            loss_list.append(loss.item())
            loss_records.append(loss.item())
            #model.train()
            #break
            
            #loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
           # evaluate(config,model,dev_iter,AUC_best)
            total_batch += 1
            if total_batch % STEP_SIZE == 0: 
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.6},  Time: {2} {3}'
                print(msg.format(total_batch, np.mean( loss_list), time_dif, improve))
                loss_list=[]

            if total_batch % config.eval_step == 0 and total_batch > 0: 
                auc,eval_res=evaluate(model,dev_iter,news_iter)
                log_res(config,eval_res,"step-{}".format(total_batch))
                # if auc>AUC_best:
                #     AUC_best=auc
                #     if config.save_flag:
                #         torch.save(model.state_dict(), config.save_path+'T{}_{}_epoch{}_iter_{}_auc_{:.3f}.ckpt'.format(time.strftime('%m-%d_%H.%M'),config.model_name,config.num_epochs,total_batch,AUC_best))

        auc, eval_res=evaluate(model,dev_iter,news_iter)
        log_res(config,eval_res,'epoch_{}'.format(epoch))
        if auc>AUC_best:
            AUC_best=auc
            if config.save_flag:
                torch.save(model.state_dict(), config.save_path+'T{}_{}_epoch{}_iter_{}_auc_{:.3f}.ckpt'.format(time.strftime('%m-%d_%H.%M'),config.model_name,config.num_epochs,total_batch,AUC_best))

        if flag:
            break
    plot_loss(config.log_path,loss_records,step_size=STEP_SIZE)
 

def _cal_score(y_true, pred, real_length):
    auc = auc_score(y_true[:real_length], pred[:real_length])
    mrr = mrr_score(y_true[:real_length], pred[:real_length])
    ndcg5 = ndcg_score(y_true[:real_length], pred[:real_length], 5)
    ndcg10 = ndcg_score(y_true[:real_length], pred[:real_length], 10)
    return [auc, mrr, ndcg5, ndcg10]

def evaluate(model, data_iter, news_iter):
    model.eval()
    res=[]
    scores=[]
    with torch.no_grad():
        print("update news vectors")
        candidate_lens=[]
        model.update_rep(news_iter)
        with tqdm(total=(data_iter.__len__()), desc='Predicting') as p:
            for i, datas in enumerate(data_iter):
                #print(datas)
                outputs = model.predict(datas).cpu()
                candidate_lens+=list(datas["candidate_lens"])
                # print(outputs)
                scores.append(outputs)
                p.update(1)
        rank_score=np.concatenate(scores)
        y_true = np.array([1] + [0 for _ in range(rank_score.shape[1] - 1)])
        print("calculating metrics...")
        res = Parallel(n_jobs=4)(
            delayed(_cal_score)(y_true, _, candidate_lens[i]) for i,_ in enumerate(rank_score)
        )
        final_scores = np.array(res).mean(axis=0)
        eval_res = "auc:{:.4f}\tmrr:{:.4f}\tndcg@5:{:.4f}\tndcg@10:{:.4f}".format(final_scores[0],\
                                final_scores[1],\
                                final_scores[2],\
                                final_scores[3])
        print(eval_res)
    return final_scores[0], eval_res 
def log_res(config,auc,step,mode="val"):
    if not  os.path.exists(config.log_path):
        os.mkdir(config.log_path) 
    with open(config.log_path+'/res.txt','a+') as f:
        f.write('{}\t{}\t{}\t{}\n'.format(time.strftime('%m-%d_%H.%M'),step,auc,mode))


if __name__=='__main__':
    print(os.getpid())
    torch.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)
    #build_dataset(config,_type=0)
    torch.backends.cudnn.deterministic = True
    # config=Config('NRMS', 'MIND')
    # config.__nrms__()
    config=Config('GNUD', 'MIND')
    config.__gnud__()
    with open(os.path.join(config.data_path,config.train_data), 'rb') as f:
        train_data=pickle.load(f)
    with open(os.path.join(config.data_path,config.val_data), 'rb') as f:
        val_data=pickle.load(f)
    with open(os.path.join(config.data_path, "MIND/news.pkl"),'rb') as f:
        news_dict=pickle.load(f)
    
    data=MyDataset(config,train_data, news_dict)
    train_iter = DataLoader(dataset=data, 
                              batch_size=config.batch_size, 
                              num_workers=4,
                              drop_last=False,
                              shuffle=True,
                              pin_memory=False)

    val_data=MyDataset(config,val_data, news_dict, type=1)
    val_iter = DataLoader(dataset=val_data, 
                              batch_size=config.batch_size, 
                              num_workers=4,
                              drop_last=False,
                              shuffle=False,
                              pin_memory=False)
    
    data=NewsDataset(news_dict)
    news_iter = DataLoader(dataset=data, 
                              batch_size=1280, 
                              num_workers=4,
                              drop_last=False,
                              shuffle=False,
                              pin_memory=False)
    model=Model(config).to(config.device)

    train(config,model,train_iter,val_iter, news_iter)


 