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


"""
加载数据到内存中：

#数据格式 [history_idx,categ_idx,subcateg_idx,imp_idx,imp_categ_idx,imp_subcateg_idx]

"""
# @log_exec_time
# def load_dataset(config,file,path,_type=0):
#     if os.path.exists(path+'idx_'+file):
#         with open(path+'idx_'+file,'rb') as f:
#             contents=pickle.load(f)
#         return contents


#     with open(path+file,'rb') as f:
#         data_list=pickle.load(f)
#     contents=[]
#     # 每条数据的格式： 
#     # 训练集： [history_news ],[ impression_logs]
#     # 验证集:
#     if config.mode=='demo':
#         NP=Demo_News_Processor(config)
#     else:
#         NP=News_Processor(config)
#     news_info=NP._get_all_news_info()

#     key=news_info['Category'].unique().tolist()
#     value=[_ for _ in range(len(key))]
#     categ_dict=dict(zip(key, value))

#     key=news_info['SubCategory'].unique().tolist()
#     value=[_ for _ in range(len(key))]
#     subcateg_dict=dict(zip(key, value))

#     categ_info=news_info[['News_ID','Category']] 
#     categ_info=categ_info.set_index('News_ID')
#     id2categ_dict=categ_info.T.to_dict('list')

#     subcateg_info=news_info[['News_ID','SubCategory']] 
#     subcateg_info=subcateg_info.set_index('News_ID')
#     id2subcateg_dict=subcateg_info.T.to_dict('list')

#     news_info=news_info['News_ID'].reset_index()
#     news_info=news_info.set_index('News_ID')
#     news_dict=news_info.T.to_dict('list')

#     for _chunk in  tqdm(data_list):
#         for sample in _chunk:
#             #print(sample)
#             # sample[0]: history  sample[1]: impression_list
#             #history_idx=news_info[ news_info['News_ID'].isin(sample[0])].index.tolist()
#             history_idx=[news_dict[_][0]+1 for _ in sample[0]]

#             if _type==0:
#                 # 训练集中剔除 历史记录比较少的数据
#                 if len(history_idx)<5:
#                     continue
#             categ_idx=[categ_dict[id2categ_dict[_][0]]+1 for _ in sample[0]]
#             subcateg_idx=[subcateg_dict[id2subcateg_dict[_][0]]+1 for _ in sample[0]]

#             for imp_logs in sample[1]:

#                 #imp_idx=news_info[ news_info['News_ID'].isin(imp_logs)].index.tolist()
#                 imp_idx=[news_dict[_][0]+1 for _ in imp_logs]
#                 imp_categ_idx=[categ_dict[id2categ_dict[_][0]]+1 for _ in imp_logs]
#                 imp_subcateg_idx=[subcateg_dict[id2subcateg_dict[_][0]]+1 for _ in imp_logs]
#                 contents.append([history_idx,categ_idx,subcateg_idx,imp_idx,imp_categ_idx,imp_subcateg_idx])
#                 # if _type>0:
#                 #     break
#         #break
#     with open(path+'idx_'+file,'wb') as f:
#         pickle.dump(contents,f)
         
#     return contents  # [([...], 0), ([...], 1), ...]


# def get_Words_Infos(config):
#     if os.path.exists(config.data_path+'news_title.pkl'):
#         with open(config.data_path+'news_title.pkl','rb') as f:
#             title_dict=pickle.load(f)
#         with open(config.data_path+'news_abst.pkl','rb') as f:
#             abst_dict=pickle.load(f)
#         return title_dict,abst_dict
#     news_df=pd.read_csv(config.data_path+'news_words.csv',header=None)
#     news_df.columns=['news_id','title','abstract']
#     #news_df=news_df.set_index('index')
#     title_id_dict={}
#     abst_id_dict={}
#     for i,row in tqdm(news_df.iterrows()):
#         title_id_dict[i]=literal_eval(row['title'])
#     for i,row in tqdm(news_df.iterrows()):
#         abst_id_dict[i]=literal_eval(row['abstract'])


#     with open(config.data_path+'news_title.pkl','wb') as f:
#         pickle.dump(title_id_dict,f)
#     with open(config.data_path+'news_abst.pkl','wb') as f:
#         pickle.dump(abst_id_dict,f)
#     return title_id_dict,abst_id_dict

# def get_Demo_Words_Infos(config):
#     if os.path.exists(config.data_path+'demo_news_title.pkl'):
#         with open(config.data_path+'demo_news_title.pkl','rb') as f:
#             title_dict=pickle.load(f)
#         with open(config.data_path+'demo_news_abst.pkl','rb') as f:
#             abst_dict=pickle.load(f)
#         return title_dict,abst_dict
#     news_df=pd.read_csv(config.data_path+'demo_news_words.csv',header=None)
#     news_df.columns=['news_id','title','abstract']
#     #news_df=news_df.set_index('index')
#     title_id_dict={}
#     abst_id_dict={}
#     for i,row in tqdm(news_df.iterrows()):
#         title_id_dict[i]=literal_eval(row['title'])
#     for i,row in tqdm(news_df.iterrows()):
#         abst_id_dict[i]=literal_eval(row['abstract'])


#     with open(config.data_path+'demo_news_title.pkl','wb') as f:
#         pickle.dump(title_id_dict,f)
#     with open(config.data_path+'demo_news_abst.pkl','wb') as f:
#         pickle.dump(abst_id_dict,f)
#     return title_id_dict,abst_id_dict

class MyDataset(Dataset):
    def __init__(self, config, datas, news_dict, type=0):
        super(MyDataset, self).__init__()
        self.config = config
        self.data_type = type
        self.bacthes=datas
        self.news_dict=news_dict
        #self.entity_dict=np.load('../data_processed/entitiy_ids.npz')['embeddings'].astype('int')
        #self.entity_nums=config.entity_nums
        #self.id2abst_dict=abst_id_dict#abst_info.T.to_dict('list')
        if type<1:
            self.sample_size=self.config.negsample_size+1
        else:
            self.sample_size=self.config.max_candidate_size
        # print('dataset batch nums: ',len(datas)//config.batch_size)
        
         
        
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
        browsed_mask=torch.ByteTensor([1 for _ in range(x)]+[0 for _ in range(self.config.history_len-x )])
        browsed_titles[:x,:]=np.array([self.news_dict[i]["Title"] for i in data[0]] )
        browsed_absts[:x,:]=np.array([self.news_dict[i]["Abstract"]  for i in data[0]] )

        browsed_categ_ids[:x]=np.array([self.news_dict[i]["Category"]  for i in data[0]] )
        browsed_subcateg_ids[:x]=np.array([self.news_dict[i]["SubCategory"]  for i in data[0]] )
            
            # 对训练集而言： 需要构造新闻imps的数据特征； 
            # 而测试和验证集，均不需要，直接统一填充即可
        y=len(data[1][:self.sample_size])
        # candidate_ids[:y]=np.array(data[1][:self.sample_size])
        candidate_titles[:y,:]=np.array([self.news_dict[i]["Title"] for i in data[1][:self.sample_size]])
        candidate_absts[:y,:]=np.array([self.news_dict[i]["Abstract"]  for i in data[1][:self.sample_size]] )

        candidate_categ_ids[:y]=np.array([self.news_dict[i]["Category"]  for i in data[1][:self.sample_size]])
        candidate_subcateg_ids[:y]=np.array([self.news_dict[i]["SubCategory"]  for i in data[1][:self.sample_size]] )

        #candidate_entity_ids=self.entity_dict[candidate_ids][:,:self.entity_nums]
        candidate_mask=torch.ByteTensor([1 for _ in range(y)]+[0 for _ in range(self.sample_size-y)])

        return {'browsed_lens':browsed_lens,\
                'browsed_titles':browsed_titles,\
                'browsed_absts':browsed_absts,\
                #'browsed_entity_ids':browsed_entity_ids,\
                'browsed_categ_ids':browsed_categ_ids,\
                'browsed_subcateg_ids':browsed_subcateg_ids,\
                'browsed_mask':browsed_mask,\
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
    data=MyDataset(config,train_data, news_dict)
    train_iter = DataLoader(dataset=data, 
                              batch_size=256, 
                              num_workers=4,
                              drop_last=False,
                              shuffle=False,
                              pin_memory=False)
    for i,_ in enumerate(train_iter):
        print(i)
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
 




 


