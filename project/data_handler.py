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
        title_mask = np.where(browsed_titles > 0, 1, 0)
        abst_mask = np.where(browsed_absts > 0, 1, 0)
        return {'ids': browsed_ids,
                'titles':browsed_titles,\
                'absts':browsed_absts,\
                'title_mask':title_mask,\
                'abst_mask':abst_mask,\
                #'browsed_entity_ids':browsed_entity_ids,\
                'categ_ids':browsed_categ_ids,\
                'subcateg_ids':browsed_subcateg_ids}

class AdrNewsDataset(Dataset):
    def __init__(self, config, news_dict):
        super(AdrNewsDataset, self).__init__()
        self.bacthes=list(news_dict.items())
        self.config = config
        print("news num: ",len(self.bacthes)) 

    def __len__(self):
        return len(self.bacthes)

    def __getitem__(self, index):
        # 用户侧
        data=self.bacthes[index][1]
        browsed_ids=index
        # 初始化
        browsed_titles=np.array(data["titleid"][:self.config.n_words_title] + [0]*(self.config.n_words_title-len(data["titleid"][:self.config.n_words_title])),dtype=np.int)
        browsed_categ_ids=torch.LongTensor([data["categoryid"]])
        browsed_subcateg_ids=torch.LongTensor([data["subcategoryid"]])
        title_mask = np.where(browsed_titles > 0, 1, 0)
        abst_mask = np.where(browsed_absts > 0, 1, 0)
        return {'ids': browsed_ids,
                'titles':browsed_titles,\
                # 'absts':browsed_absts,\
                'title_mask':title_mask,\
                # 'abst_mask':abst_mask,\
                #'browsed_entity_ids':browsed_entity_ids,\
                'categ_ids':browsed_categ_ids,\
                'subcateg_ids':browsed_subcateg_ids}


class MyDataset(Dataset):
    def __init__(self, config, datas, news_dict, is_train=True):
        super(MyDataset, self).__init__()
        self.config = config
        self.data_type = type
        self.bacthes=datas
        self.news_dict=news_dict
        #self.entity_dict=np.load('../data_processed/entitiy_ids.npz')['embeddings'].astype('int')
        #self.entity_nums=config.entity_nums
        if is_train:
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
        browsed_title_mask = np.where(browsed_titles > 0, 1, 0)
        browsed_abst_mask = np.where(browsed_absts > 0, 1, 0)

        browsed_categ_ids[:x]=np.array([self.news_dict[i]["Category"]  for i in data[0]] )
        browsed_subcateg_ids[:x]=np.array([self.news_dict[i]["SubCategory"]  for i in data[0]] )
            # 对训练集而言： 需要构造新闻imps的数据特征； 
            # 而测试和验证集，均不需要，直接统一填充即可
        y=len(data[1][:self.sample_size])
        candidate_ids[:y]=np.array([self.news_dict[i]["index"] for i in data[1][:self.sample_size]])
        candidate_titles[:y,:]=np.array([self.news_dict[i]["Title"] for i in data[1][:self.sample_size]])
        candidate_absts[:y,:]=np.array([self.news_dict[i]["Abstract"]  for i in data[1][:self.sample_size]] )
        candidate_title_mask = np.where(candidate_titles > 0, 1, 0)
        candidate_abst_mask = np.where(candidate_absts > 0, 1, 0)


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
                'browsed_title_mask':browsed_title_mask,\
                'browsed_abst_mask':browsed_abst_mask,\
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
                'candidate_title_mask':candidate_title_mask,\
                'candidate_abst_mask':candidate_abst_mask,\
                'candidate_mask':candidate_mask}



class AdrDataset(Dataset):
    def __init__(self, config, datas, news_dict, is_train=True):
        super(AdrDataset, self).__init__()
        self.config = config
        self.data_type = type
        self.bacthes=datas
        self.news_dict=news_dict
        #self.entity_dict=np.load('../data_processed/entitiy_ids.npz')['embeddings'].astype('int')
        #self.entity_nums=config.entity_nums
        if is_train:
            self.sample_size=self.config.negsample_size+1
        else:
            self.sample_size=self.config.max_candidate_size
        print('dataset batch nums: ',len(datas)//config.batch_size)
        
    def __len__(self):
        return len(self.bacthes)



    def __timematrix__(self, intervals):
        time_mats = []
        for th in self.config.time_thresh:
            p = [1 if _ < th else self.config.time_p for _ in intervals]
            lens = len(p)
            w=np.ones((lens,lens))
            w=np.triu(w)
            for i in range(lens):
                for j in range(i + 1, lens):
                    w[i][j]=w[i][j-1]*p[j]
            w += w.T - np.diag(w.diagonal())
            w = np.pad(w, (0, self.config.history_len - lens), 'constant')
            time_mats.append(w)
        
        return torch.FloatTensor(np.array(time_mats))


    def __getitem__(self, index):
        # userId, hist_list, time_list, cand_list
        data=self.bacthes[index]
         
        browsed_ids=np.zeros((self.config.history_len),dtype=np.int)
        candidate_ids=np.zeros((self.sample_size),dtype=np.int)

        #print(data[0])
        # 初始化
        browsed_titles=np.zeros((self.config.history_len,self.config.n_words_title),dtype=np.int)
        browsed_categ_ids=np.zeros((self.config.history_len),dtype=np.int)
        browsed_subcateg_ids=np.zeros((self.config.history_len),dtype=np.int)

        candidate_titles=np.zeros((self.sample_size,self.config.n_words_title),dtype=np.int)
        candidate_categ_ids=np.zeros((self.sample_size),dtype=np.int)
        #browsed_entity_ids=self.entity_dict[browsed_ids][:,:self.entity_nums]
        candidate_subcateg_ids=np.zeros((self.sample_size),dtype=np.int)
   
        hist=data[1][-self.config.history_len:]
        x = len(hist)
        # print(x)
        browsed_lens=len(hist)
        browsed_ids[:x] = np.array(hist)
        browsed_mask=torch.ByteTensor([1 for _ in range(x)]+[0 for _ in range(self.config.history_len-x )])
        # print("hist",hist)
        # print([self.news_dict[i]["titleid"] + [0] *(self.config.n_words_title - len(self.news_dict[i]["titleid"])) for i in hist])
        browsed_titles[:x,:]=np.array([self.news_dict[i]["titleid"][:self.config.n_words_title ] +\
                                        [0] *(self.config.n_words_title - len(self.news_dict[i]["titleid"])) for i in hist] )
        browsed_title_mask = np.where(browsed_titles > 0, 1, 0)
      
        browsed_categ_ids[:x]=np.array([self.news_dict[i]["categoryid"]  for i in hist] )
        browsed_subcateg_ids[:x]=np.array([self.news_dict[i]["subcategoryid"]  for i in hist] )
            # 对训练集而言： 需要构造新闻imps的数据特征； 
            # 而测试和验证集，均不需要，直接统一填充即可
        cands=data[3][:self.sample_size]
        y = len(cands)
        candidate_ids[:y]=np.array(cands)
        candidate_titles[:y,:]=np.array([self.news_dict[i]["titleid"][:self.config.n_words_title ] +\
                                            [0] *(self.config.n_words_title - len(self.news_dict[i]["titleid"])) for i in cands])
        candidate_title_mask = np.where(candidate_titles > 0, 1, 0)
      

        candidate_categ_ids[:y]=np.array([self.news_dict[i]["categoryid"]  for i in cands])
        candidate_subcateg_ids[:y]=np.array([self.news_dict[i]["subcategoryid"]  for i in cands] )

        #candidate_entity_ids=self.entity_dict[candidate_ids][:,:self.entity_nums]
        candidate_mask=torch.ByteTensor([1 for _ in range(y)]+[0 for _ in range(self.sample_size-y)])
        user_id = np.array([data[0]], dtype=np.int)
        time_mat = 0
        if "time_thresh" in self.config.__dict__:
            intervals = data[2][-self.config.history_len:]
            time_mat = self.__timematrix__(intervals)
        
        return {
                'user_id': user_id,\
                'browsed_ids': browsed_ids,
                'browsed_lens':browsed_lens,\
                'browsed_titles':browsed_titles,\
                'browsed_title_mask':browsed_title_mask,\
                #'browsed_entity_ids':browsed_entity_ids,\
                'browsed_categ_ids':browsed_categ_ids,\
                'browsed_subcateg_ids':browsed_subcateg_ids,\
                'browsed_mask':browsed_mask,\
                'candidate_ids': candidate_ids,
                'candidate_lens':y,\
                'candidate_titles':candidate_titles,\
                #'candidate_entity_ids':candidate_entity_ids,\
                'candidate_categ_ids':candidate_categ_ids,
                'candidate_subcateg_ids':candidate_subcateg_ids,
                'candidate_title_mask':candidate_title_mask,\
                'candidate_mask':candidate_mask,
                'time_mat':time_mat
                }


class GloboDataset(Dataset):
    def __init__(self, config, datas, is_train=True):
        super(GloboDataset, self).__init__()
        self.config = config
        self.data_type = type
        self.bacthes=datas
        # self.news_dict=news_dict
        #self.entity_dict=np.load('../data_processed/entitiy_ids.npz')['embeddings'].astype('int')
        #self.entity_nums=config.entity_nums
        if is_train:
            self.sample_size=self.config.negsample_size + 1
        else:
            self.sample_size=self.config.max_candidate_size
        print('dataset batch nums: ',len(datas)//config.batch_size)
        
    def __len__(self):
        return len(self.bacthes)

    def __timematrix__(self, intervals):
        time_mats = []
        for th in self.config.time_thresh:
            p = [1 if _ < th else self.config.time_p for _ in intervals]
            lens = len(p)
            w=np.ones((lens,lens))
            w=np.triu(w)
            for i in range(lens):
                for j in range(i + 1, lens):
                    w[i][j]=w[i][j-1]*p[j]
            w += w.T - np.diag(w.diagonal())
            w = np.pad(w, (0, self.config.history_len - lens), 'constant')
            time_mats.append(w)
        
        return torch.FloatTensor(np.array(time_mats))

    def __getitem__(self, index):
        # 用户侧
        #  # ["user_id", "hist", "intervals", 'cands']
        data=self.bacthes[index]
         
        browsed_ids=np.zeros((self.config.history_len),dtype=np.int)
        candidate_ids=np.zeros((self.sample_size),dtype=np.int)

        # browsed_categ_ids=np.zeros((self.config.history_len),dtype=np.int)
        # browsed_subcateg_ids=np.zeros((self.config.history_len),dtype=np.int)

        # candidate_titles=np.zeros((self.sample_size,self.config.n_words_title),dtype=np.int)
        # candidate_absts=np.zeros((self.sample_size,self.config.n_words_abst),dtype=np.int)
        # candidate_categ_ids=np.zeros((self.sample_size),dtype=np.int)
        # candidate_subcateg_ids=np.zeros((self.sample_size),dtype=np.int)
                                       
       
        browsed_lens=min(len(data[1]), self.config.history_len)
        browsed_ids[:browsed_lens] = np.array(data[1][-self.config.history_len:])
        browsed_mask=torch.ByteTensor([1 for _ in range(browsed_lens)]+[0 for _ in range(self.config.history_len-browsed_lens )])
        time_mat = 0
        if "time_thresh" in self.config.__dict__:
            intervals = data[2][-self.config.history_len:]
            time_mat = self.__timematrix__(intervals)

        # browsed_categ_ids[:x]=np.array([self.news_dict[i]["Category"]  for i in data[0]] )
        # browsed_subcateg_ids[:x]=np.array([self.news_dict[i]["SubCategory"]  for i in data[0]] )
            # 对训练集而言： 需要构造新闻imps的数据特征； 
            # 而测试和验证集，均不需要，直接统一填充即可
        y=len(data[3][:self.sample_size])
        candidate_ids[:y]=np.array(data[3][:self.sample_size])

        # candidate_categ_ids[:y]=np.array([self.news_dict[i]["Category"]  for i in data[1][:self.sample_size]])
        # candidate_subcateg_ids[:y]=np.array([self.news_dict[i]["SubCategory"]  for i in data[1][:self.sample_size]] )

        #candidate_entity_ids=self.entity_dict[candidate_ids][:,:self.entity_nums]
        candidate_mask=torch.ByteTensor([1 for _ in range(y)]+[0 for _ in range(self.sample_size-y)])
        user_id = np.array([data[0]], dtype=np.int)
        return {'user_id': user_id,\
                'browsed_ids': browsed_ids,
                'browsed_lens':browsed_lens,\
                #'browsed_entity_ids':browsed_entity_ids,\
                # 'browsed_categ_ids':browsed_categ_ids,\
                # 'browsed_subcateg_ids':browsed_subcateg_ids,\
                'browsed_mask':browsed_mask,\
                'candidate_ids': candidate_ids,
                'candidate_lens':y,\
                #'candidate_entity_ids':candidate_entity_ids,\
                # 'candidate_categ_ids':candidate_categ_ids,
                # 'candidate_subcateg_ids':candidate_subcateg_ids,
                'candidate_mask':candidate_mask,
                'time_mat':time_mat}


def get_data_loader(config, data, is_train = True):
    if config.dataset == "GLOBO":
        data = GloboDataset(config, data, is_train)
        data_iter = DataLoader(dataset=data, 
                                batch_size=config.batch_size, 
                                num_workers=4,
                                drop_last=False,
                                shuffle=is_train,
                                pin_memory=False)
    elif config.dataset == "MIND":
        with open(os.path.join(config.data_path, "MIND/news.pkl"),'rb') as f:
            news_dict=pickle.load(f)
        data=MyDataset(config, data, news_dict, is_train)
        data_iter = DataLoader(dataset=data, 
                                batch_size=config.batch_size, 
                                num_workers=4,
                                drop_last=False,
                                shuffle=is_train,
                                pin_memory=False)
    else:
        # with open(os.path.join(config.data_path, "ADR/news.json"),'r') as f:
        #     news_dict=json.load(f)
        with open(os.path.join(config.data_path, "ADR/news.pkl"),'rb') as f:
            news_dict=pickle.load(f)
        data=AdrDataset(config, data, news_dict, is_train)
        data_iter = DataLoader(dataset=data, 
                                batch_size=config.batch_size, 
                                num_workers=4,
                                drop_last=False,
                                shuffle=is_train,
                                pin_memory=False)
    return data_iter


def get_news_loader(config):
    if config.model_name.lower() in ["fim", 'lstur', 'tmgm']:
        return None
    if config.dataset in ["MIND"]:
        with open(os.path.join(config.data_path, f"{config.dataset}/news.pkl"),'rb') as f:
            news_dict=pickle.load(f)
        if config.dataset == "MIND":
            data=NewsDataset(config, news_dict)
        else:
            data=AdrNewsDataset(config, news_dict)
        news_iter = DataLoader(dataset=data, 
                        batch_size=1280, 
                        num_workers=4,
                        drop_last=False,
                        shuffle=False,
                        pin_memory=False)
    else:
        return None
    return news_iter


if __name__ == "__main__":
    # word_dict = dict(pd.read_csv( ('../data_processed/'+ 'word_dict.csv'), sep='\t', na_filter=False, header=0).values.tolist())
    # for k, v in word_dict.items():
    #     print(k,v)
    #     break
    config = Config(dataset="GLOBO", model_name="tmgm")
    
    with open(os.path.join(config.data_path,config.train_data), 'rb') as f:
        train_data=pickle.load(f)
    
    train_iter = get_data_loader(config, train_data, is_train=True)
    
    for i,_ in enumerate(train_iter):
        print(i)
        print(_)
        # print(i)
        break
    print(train_iter.__len__())

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
 




 


