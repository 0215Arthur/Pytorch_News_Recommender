# coding: UTF-8
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


#from tensorboard import SummaryWriter
from multiprocessing import Pool
from multiprocessing import cpu_count 


global  behaviors

global _y_true 
#_y_true=[[int(_) for _ in x.split(' ')] for x in behaviors['y_true'].tolist()]

def train(config, model, train_iter, dev_iter ):
    behaviors = pd.read_csv('./data_processed/'+'dev_behaviors.csv')
    
    global _y_true
    _y_true = [[int(_) for _ in x.split(' ')] for x in behaviors['y_true'].tolist()]

     

    #assert dev_iter.__len__()==len(_y_true)

    
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
            
            if total_batch % STEP_SIZE == 0: 
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.6},  Time: {2} {3}'
                print(msg.format(total_batch, np.mean( loss_list), time_dif, improve))
                loss_list=[]
            total_batch += 1
            if total_batch % config.eval_step == 0 and total_batch>0: 
                auc=evaluate(config,model,dev_iter,AUC_best)
                log_res(config,auc,total_batch)
                if auc>AUC_best:
                    AUC_best=auc
                    if config.save_flag:
                        torch.save(model.state_dict(), config.save_path+'T{}_{}_epoch{}_iter_{}_auc_{:.3f}.ckpt'.format(time.strftime('%m-%d_%H.%M'),config.model_name,config.num_epochs,total_batch,AUC_best))

        auc=evaluate(config,model,dev_iter,AUC_best)
        log_res(config,auc,'epoch_{}'.format(epoch))
        if auc>AUC_best:
            AUC_best=auc
            if config.save_flag:
                torch.save(model.state_dict(), config.save_path+'T{}_{}_epoch{}_iter_{}_auc_{:.3f}.ckpt'.format(time.strftime('%m-%d_%H.%M'),config.model_name,config.num_epochs,total_batch,AUC_best))

        if flag:
            break
    plot_loss(config.log_path,loss_records,step_size=STEP_SIZE)
 

def train_demo(config, model, train_iter, dev_iter ):
    behaviors = pd.read_csv('./data_processed/'+'small_dev_behaviors.csv')
    global _y_true
    _y_true = [[int(_) for _ in x.split(' ')] for x in behaviors['y_true'].tolist()]
    print('result_length::::',len(_y_true))

    # assert dev_iter.__len__()==len(_y_true)


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
    
    AUC_best=0
    loss_list=[]
    STEP_SIZE=100
    improve='*'
    criterion =nn.CrossEntropyLoss()
    for epoch in range(config.num_epochs):
        #auc=evaluate(config,model,dev_iter,AUC_best,title_id_dict,abst_id_dict)
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
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
            #model.train()
            #break
            
            #loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
           # evaluate(config,model,dev_iter,AUC_best)
            
            if total_batch % STEP_SIZE == 0: 
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.6},  Time: {2} {3}'
                print(msg.format(total_batch, np.mean( loss_list), time_dif, improve))
                loss_list=[]
            total_batch += 1
            #if total_batch % 500 == 0 and total_batch>0: 
        auc=evaluate(config,model,dev_iter,AUC_best)
        


def _cal_scores(i):
    y_true=_y_true[i]
    # rank=[0 for _ in range(len(y_true))]
    # for _,v in enumerate(rank_score[i][:len(y_true)]):
    #     rank[v]=_+1
    # y_score=1./np.array(rank)
    auc = auc_score(y_true,rank_score[i][:len(y_true)])
 
    return auc

def evaluate(config, model, data_iter,AUC_best):
    model.eval()
    AUC_list=[]
    MRR_list=[]
    nDCG5_list=[]
    nDCG10_list=[]
    res=[]
    global rank_score
    scores=[]
    with torch.no_grad():
        with tqdm(total=(data_iter.__len__()), desc='Predicting') as p:
            for i, datas in tqdm(enumerate(data_iter)):
                #print(datas)
                outputs = model(datas).cpu()
              #  print(model.get_news_vectors())
                #[:,:,1].cpu()
                #print(outputs)
                #print(outputs.size())
                #[:,:,1].squeeze(-1).cpu()
                
                #_res=(np.argsort(-np.array(outputs)))#.tolist()
                scores.append(outputs)
                p.update(1)

 
        
        rank_score=np.concatenate(scores)
        pool = Pool(cpu_count()//4)
        #final_scores = pool.map(_cal_scores, range(len(rank_score)))
        final_scores = pool.map(_cal_scores, range(rank_score.shape[0]))
        pool.close()
        pool.join()
 

        # AUC_list.append(auc_score(y_true, predict))
        # MRR_list.append(mrr_score(y_true, predict))
        # nDCG5_list.append(ndcg_score(y_true, predict, 5))
        # nDCG10_list.append(ndcg_score(y_true, predict, 10))
        print('AUC:', np.mean(final_scores))
        # print('MRR:', np.mean([_[1]  for _ in final_scores]))
        # print('nDCG@5:', np.mean([_[2]  for _ in final_scores]))
        # print('nDCG@10:', np.mean([_[3]  for _ in final_scores]))
        AUC=np.mean( final_scores)
 
    return AUC
def log_res(config,step,auc):
    if not  os.path.exists(config.log_path):
        os.mkdir(config.log_path) 
    with open(config.log_path+'/res.txt','a+') as f:
        f.write('{}_{}_:auc_{}\n'.format(time.strftime('%m-%d_%H.%M'),auc,step))

def _cal_test(i):
    _len=test_list_nums[i]
    res=np.argsort(-np.array(test_rank_score[i][:_len]))
    rank=[0 for _ in range(_len)]
    for _,v in enumerate(res):
        rank[v]=_+1
    return rank#mrr,ndcg5,ndcg10
def get_Test_List(config):
    if os.path.exists(config.data_path+'test_imps_list.pkl'):
        with open(config.data_path+'test_imps_list.pkl','rb') as f:
            contents=pickle.load(f)
            return contents
    test_behaviors = pd.read_table(config.test_path+'behaviors.tsv', header=None, 
                                    names=['user_id','time','history','impressions'])
    test_behaviors['impressions'].apply(lambda x: len(x.split(' ')))
    test_imps_lens=test_behaviors['impressions'].apply(lambda x: len(x.split(' '))).tolist()
    with open(config.data_path+'test_imps_list.pkl','wb') as f:
        pickle.dump(test_imps_lens,f)
        return test_imps_lens

def test(config, model, data_iter,ckpt_file):
    if ckpt_file is None:
        auc_best=0.5
        for ckpt in os.listdir(config.save_path):
            if config.model_name in ckpt:
                tmp_auc=eval(ckpt.split('_')[-1].strip('.ckpt'))
                if tmp_auc>auc_best:
                    auc_best=tmp_auc
                    ckpt_file=ckpt
    model.load_state_dict(torch.load('./save_model/'+ckpt_file))
    print('load the ckpt_file:{}'.format(ckpt_file))


    model.eval()
    global  test_list_nums
    test_list_nums=get_Test_List(config)
     
    res=[]
    global test_rank_score
    scores=[]
    with torch.no_grad():
        with tqdm(total=(data_iter.__len__()), desc='Predicting') as p:
            for i, datas in tqdm(enumerate(data_iter)):
                outputs = model(datas).cpu()
                #_res=(np.argsort(-np.array(outputs)))#.tolist()
                scores.append(outputs)
                p.update(1)
                # if i>10:
                #     break
        test_rank_score=np.concatenate(scores)
        pool = Pool(cpu_count()//4)
        #final_scores = pool.map(_cal_scores, range(len(rank_score)))
        final_scores = pool.map(_cal_test, range(test_rank_score.shape[0]))
        pool.close()
        pool.join()
    file_name='sumbit_{}_{}.txt'.format(config.model_name,time.strftime('%m-%d_%H.%M', time.localtime()))
     
    with open(file_name,'w') as f:
        for i,_ in enumerate(final_scores):
            f.write(str(i+1)+' ')
            f.write(str(_).replace(' ','')+'\n')
    print('saved to {}'.format(file_name))


if __name__=='__main__':

    print(os.getpid())
    torch.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)
    #build_dataset(config,_type=0)
    torch.backends.cudnn.deterministic = True
    config=Config('NRMS+Rank-score_sum')
     




    # test_data=load_dataset(config,'test_datas.pkl',config.data_path,_type=1)
    
    # test_data=MyDataset(config,test_data,type=1)

    # test_iter = DataLoader(dataset=test_data, 
    #                           batch_size=1024, 
    #                           num_workers=6,
    #                           drop_last=False,
    #                           shuffle=False,
    #                          pin_memory=False)
    model=NRMS(config).to(config.device)


    # test(config,model,test_iter,'NRMS+Rank-score_sum.ckptT09-03_01.44_NRMS+Rank-score_sum_epoch10_iter_15000_auc_0.657.ckpt')
    

    dataset=load_dataset(config,'train_datas.pkl',config.data_path,_type=0)
    train_data=MyDataset(config,dataset ,type=0)
    train_iter = DataLoader(dataset=train_data, 
                              batch_size=config.batch_size, 
                              num_workers=6,
                              drop_last=False,
                              shuffle=False,
                              pin_memory=False)
     
  
    
    dev_data=load_dataset(config,'dev_datas.pkl',config.data_path,_type=1)
    
    dev_data=MyDataset(config,dev_data ,type=1)

    dev_iter = DataLoader(dataset=dev_data, 
                              batch_size=config.batch_size, 
                              num_workers=6,
                              drop_last=False,
                              shuffle=False,
                              pin_memory=False)


    # # dev_iter=build_iterator(dev_data,config,title_id_dict,abst_id_dict,train_flag=False)
    # # #dev_iter=None
    # model=NRMS(config).to(config.device)
    # print(model.parameters)
    # #init_network(model)
    train(config,model,train_iter,dev_iter)


 