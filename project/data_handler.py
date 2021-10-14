# coding: UTF-8


"""
script: 数据预处理、模型加载数据迭代器


"""
from ast import literal_eval
import os
import torch
import numpy as np
  
from torch.utils.data import Dataset,DataLoader

from tqdm import tqdm
import time
from datetime import timedelta
 
import pickle
 
import gc
import json
import pandas as pd 
#from sklearn.utils import shuffle
import random 
from multiprocessing import Pool
from multiprocessing import cpu_count 
#from random  import shuffle
 
import functools
from tools import log_exec_time,get_time_dif

from config import Config
# from data_processor import News_Processor,Demo_News_Processor

class NewsDataset(Dataset):
    def __init__(self, news_dict):
        super(NewsDataset, self).__init__()
        self.bacthes=list(news_dict.items())
        print("news num: ",len(self.bacthes)) 
    def __len__(self):
        return len(self.bacthes)

    def __getitem__(self, index):
        # 用户侧
        data=self.bacthes[index][1]
        browsed_ids=data["index"]
        # 初始化
        browsed_titles=np.array(data["Title"],dtype=np.int)
        browsed_absts=np.array(data["Abstract"],dtype=np.int)
        browsed_categ_ids=torch.LongTensor([data["Category"]])
        browsed_subcateg_ids=torch.LongTensor([data["SubCategory"]])

        return {'ids': browsed_ids,
                'titles':browsed_titles,\
                'absts':browsed_absts,\
                #'browsed_entity_ids':browsed_entity_ids,\
                'categ_ids':browsed_categ_ids,\
                'subcateg_ids':browsed_subcateg_ids}

class MyDataset(Dataset):
    def __init__(self, config, datas, news_dict, type=0):
        super(MyDataset, self).__init__()
        self.config = config
        self.data_type = type
        self.bacthes=datas
        self.news_dict=news_dict
        #self.entity_dict=np.load('../data_processed/entitiy_ids.npz')['embeddings'].astype('int')
        #self.entity_nums=config.entity_nums
        if type<1:
            self.sample_size=self.config.negsample_size+1
        else:
            self.sample_size=self.config.max_candidate_size
        print('dataset batch nums: ',len(datas)//config.batch_size)
        
    def __len__(self):
        return len(self.bacthes)

    def __getitem__(self, index):
        # 用户侧
        data=self.bacthes[index]
         
        browsed_ids=np.zeros((self.config.history_len),dtype=np.int)
        candidate_ids=np.zeros((self.sample_size),dtype=np.int)

        #print(data[0])
        # 初始化
        browsed_titles=np.zeros((self.config.history_len,self.config.n_words_title),dtype=np.int)
        browsed_absts=np.zeros((self.config.history_len,self.config.n_words_abst),dtype=np.int)
        browsed_categ_ids=np.zeros((self.config.history_len),dtype=np.int)
        browsed_subcateg_ids=np.zeros((self.config.history_len),dtype=np.int)

        candidate_titles=np.zeros((self.sample_size,self.config.n_words_title),dtype=np.int)
        candidate_absts=np.zeros((self.sample_size,self.config.n_words_abst),dtype=np.int)
        candidate_categ_ids=np.zeros((self.sample_size),dtype=np.int)
        #browsed_entity_ids=self.entity_dict[browsed_ids][:,:self.entity_nums]
        candidate_subcateg_ids=np.zeros((self.sample_size),dtype=np.int)
 
        x=len(data[0])
        browsed_lens=x
        browsed_ids[:x] = np.array([self.news_dict[i]["index"] for i in data[0]] )
        browsed_mask=torch.ByteTensor([1 for _ in range(x)]+[0 for _ in range(self.config.history_len-x )])
        browsed_titles[:x,:]=np.array([self.news_dict[i]["Title"] for i in data[0]] )
        browsed_absts[:x,:]=np.array([self.news_dict[i]["Abstract"]  for i in data[0]] )

        browsed_categ_ids[:x]=np.array([self.news_dict[i]["Category"]  for i in data[0]] )
        browsed_subcateg_ids[:x]=np.array([self.news_dict[i]["SubCategory"]  for i in data[0]] )
            
            # 对训练集而言： 需要构造新闻imps的数据特征； 
            # 而测试和验证集，均不需要，直接统一填充即可
        y=len(data[1][:self.sample_size])
        candidate_ids[:y]=np.array([self.news_dict[i]["index"] for i in data[1][:self.sample_size]])
        candidate_titles[:y,:]=np.array([self.news_dict[i]["Title"] for i in data[1][:self.sample_size]])
        candidate_absts[:y,:]=np.array([self.news_dict[i]["Abstract"]  for i in data[1][:self.sample_size]] )

        candidate_categ_ids[:y]=np.array([self.news_dict[i]["Category"]  for i in data[1][:self.sample_size]])
        candidate_subcateg_ids[:y]=np.array([self.news_dict[i]["SubCategory"]  for i in data[1][:self.sample_size]] )

        #candidate_entity_ids=self.entity_dict[candidate_ids][:,:self.entity_nums]
        candidate_mask=torch.ByteTensor([1 for _ in range(y)]+[0 for _ in range(self.sample_size-y)])
        user_id = np.array([data[2]], dtype=np.int)
        return {'user_id': user_id,\
                'browsed_ids': browsed_ids,
                'browsed_lens':browsed_lens,\
                'browsed_titles':browsed_titles,\
                'browsed_absts':browsed_absts,\
                #'browsed_entity_ids':browsed_entity_ids,\
                'browsed_categ_ids':browsed_categ_ids,\
                'browsed_subcateg_ids':browsed_subcateg_ids,\
                'browsed_mask':browsed_mask,\
                'candidate_ids': candidate_ids,
                'candidate_lens':y,\
                'candidate_titles':candidate_titles,\
                'candidate_absts':candidate_absts,\
                #'candidate_entity_ids':candidate_entity_ids,\
                'candidate_categ_ids':candidate_categ_ids,
                'candidate_subcateg_ids':candidate_subcateg_ids,
                'candidate_mask':candidate_mask}


if __name__ == "__main__":
    # word_dict = dict(pd.read_csv( ('../data_processed/'+ 'word_dict.csv'), sep='\t', na_filter=False, header=0).values.tolist())
    # for k, v in word_dict.items():
    #     print(k,v)
    #     break
    config = Config()
    config.__MIND__()
    
    
    with open("./dataset_processed/MIND/train_datas.pkl", 'rb') as f:
        train_data=pickle.load(f)
    
    with open(os.path.join(config.data_path, "MIND/news.pkl"),'rb') as f:
        news_dict=pickle.load(f)
    print(len(news_dict))
    data=MyDataset(config,train_data, news_dict)
    train_iter = DataLoader(dataset=data, 
                              batch_size=256, 
                              num_workers=4,
                              drop_last=False,
                              shuffle=False,
                              pin_memory=False)

    data=NewsDataset(news_dict)
    news_iter = DataLoader(dataset=data, 
                              batch_size=256, 
                              num_workers=4,
                              drop_last=False,
                              shuffle=False,
                              pin_memory=False)
    # for i,_ in enumerate(news_iter):
    #     print(i)
    #     print(_["ids"])
        #print(i)
        #break
    #print(train_iter.__len__())

    # dev_data=load_dataset(config,'dev_datas.pkl',config.data_path,_type=1)
    # print(len(dev_data))
    # test_data=load_dataset(config,'test_datas.pkl',config.data_path,_type=2)
    # print(len(test_data))
    # train_data=load_dataset(config,'train_datas.pkl',config.data_path,_type=0)
    # print(len(train_data))


    #title_id_dict,abst_id_dict=get_Words_Infos(config)

     
    # dataset=load_dataset(config,'word_train_datas.pkl',config.data_path,_type=0)
    # train_data=MyDataset(config,dataset,title_id_dict,abst_id_dict,type=0)

    # train_loader = DataLoader(dataset=train_data, 
    #                           batch_size=config.batch_size, 
    #                           num_workers=6,
    #                           drop_last=False,
    #                           shuffle=False,
    #                           pin_memory=False)

    #dataset=load_dataset(config,'word_dev_datas.pkl',config.data_path,_type=1)
    
    # dev_data=MyDataset(config,dataset,title_id_dict,abst_id_dict,type=1)

    # dev_iter = DataLoader(dataset=dev_data, 
    #                           batch_size=config.batch_size, 
    #                           num_workers=6,
    #                           drop_last=False,
    #                           shuffle=False,
    #                           pin_memory=False)
 

    # dataset=load_dataset(config,'dev_datas.pkl',config.data_path,_type=1)
    # 
 




 


