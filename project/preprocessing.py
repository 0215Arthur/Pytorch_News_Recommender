"""
pre-process the raw datasets to get the proper format data for model.

"""

from ast import literal_eval
import os
import torch
import numpy as np

from tqdm import tqdm
import time
from datetime import timedelta
 
import pickle
 
import gc
import pandas as pd 
#from sklearn.utils import shuffle
import random 
 
from nltk.tokenize import word_tokenize,RegexpTokenizer
from string import digits
from multiprocessing import Pool
from multiprocessing import cpu_count 
#from random  import shuffle
try:
    from bert_serving.client import BertClient
except:
    pass

import functools
from tools import *

from config import Config

class News_Processor(object):
    def __init__(self,config):
        self.dataset="MIND"
        self.conf=config
        self.root="./data/MIND/"
        self.train_path=self.root + "train/"
        self.dev_path=self.root + "dev/"
        self.stop_words=['the',',','.',':','...','``','\'',"''",'\'s'
                         #'for','it','and','from','in','as','he','his','be',
                         #'this','will','after','by','have','you',
                         #'was','a','an','at','with','that','to','of','on','is'
                         ]
    @log_exec_time
    def _get_bert_embeds(self,embed_size=1024):
        """
        build bert embedding for title/abstrats
        """
        news_info=self._get_all_news_info(cols=['News_ID','Title','Abstract'])
        ## 填充缺失摘要的新闻
        news_info.fillna(method='ffill', axis=1,inplace=True)
        bc=BertClient(check_length=False) # check_length=False
        titles_vec=bc.encode(news_info['Title'].tolist())
        absts_vec= bc.encode(news_info['Abstract'].tolist())
        embeds=(titles_vec+absts_vec)/2
        print('News original size:',embeds.shape)
    # flags=['train','dev','test']
    # file_name=   '{}_news_ebmeds.npz'.format(flags[data_type])
        x=np.zeros((1,embed_size))
        
        embeds=np.concatenate([x,embeds])
        print('News merge size:',embeds.shape)
        np.savez_compressed('./data_processed/news_embeds_{}.npz'.format(embed_size), embeddings=embeds)
        print('npz saved to ./data_processed/news_embeds.npz')

    @log_exec_time
    def _get_word_embeds(self):
        """
        build word embedding matrix for the words in dataset
        
        """
        print('Start word embedding...')
        word_dict = dict(pd.read_csv(os.path.join(self.conf.data_path, 'word_dict.csv'), sep='\t', na_filter=False, header=0).values.tolist())

        glove_embedding = pd.read_table('./glove.840B.300d.txt', sep=' ', header=None, index_col=0, quoting=3)
        embedding_result = np.random.normal(size=(len(word_dict) , 300))
        embedding_result=np.concatenate((np.zeros((1,300)),embedding_result))
        print(embedding_result.shape)
        word_missing = 0
       # return 
        #embedding_result=glove_embedding.loc[word_dict.items()]

        with tqdm(total=len(word_dict), desc="Generating word embedding") as p:
            for k, v in word_dict.items():
            # print(k,v)
                if k in glove_embedding.index:
                    embedding_result[v] = glove_embedding.loc[k].tolist()
                else:
                    word_missing += 1
                p.update(1)
        print('\ttotal_missing_word:', word_missing)
        print('Finish word embedding')
        
        #embeds=np.load(os.path.join(config.data_, 'word_embedding.npy'))
        np.savez_compressed(os.path.join(self.conf.data_path, 'all_word_embedding_v3.npz'), embeddings=embedding_result)
        print('npz saved to ./data_processed/all_word_embeds_v3.npz')

    @log_exec_time
    def _get_all_news_info(self,cols=['News_ID','Category','SubCategory']):
        """
        params:
              cols: the detail columns (list)
        return all news info (DataFrame format)
        """
        news_info=pd.read_table(self.train_path+'news.tsv',header=None,sep='\t',
                                    names=['News_ID','Category','SubCategory',
                                            'Title','Abstract','URL',
                                            'Title Entities','Abstract Entites'])[cols]
        news_info_2=pd.read_table(self.dev_path+'news.tsv',header=None,sep='\t',
                                        names=['News_ID','Category','SubCategory',
                                                'Title','Abstract','URL',
                                                'Title Entities','Abstract Entites'])[cols]

        news_info=pd.concat([news_info_2,news_info])
    
        train_lens=news_info.shape[0]
        news_info.drop_duplicates(['News_ID' ],inplace=True,keep='first')
        print('train_set + dev_set : the number of News',news_info.shape)
        news_info=news_info.dropna(axis=0,subset = ["Title"])
        #print(new_lens-old_lens)
        print(news_info.shape[0])
        print(news_info["News_ID"].nunique())
        return news_info[cols]
    
    def _count_news_words(self):
        """
        func: count the words of the all news to remap the words to IDs.
        """
        train_news=self._get_all_news_info(cols=['News_ID','Title','Abstract'])
        # print(train_news['Title'].head())
        # print(word_tokenize("I Was An NBA Wife. Here's How It Affected My M"))
        #return 
        def clean_words(x):
            try:
                return x.lower().translate(str.maketrans('', '', digits))
            except:
                return None
        tokenizer = RegexpTokenizer(r"\w+")

        train_news['Title']=train_news['Title'].apply(lambda x :clean_words(x))
        train_news['Abstract']=train_news['Abstract'].apply(lambda x :clean_words(x))
        category_dict, word_freq_dict, word_dict = {}, {}, {}
        with tqdm(total=len(train_news), desc='Processing news') as p:
            for row in train_news.itertuples():
                for word in tokenizer.tokenize(row.Title):
                    # word=word.strip('-+=.\'')
                    # if word not in self.stop_words:
                    word_freq_dict[word] = word_freq_dict.get(word, 0) + 1

                try:
                    #abst=row.Abstract.lower().translate(str.maketrans('', '', digits))
                    for word in tokenizer.tokenize(row.Abstract):
                        # word=word.strip('-+=.\'')
                        # if word not in self.stop_words:
                        word_freq_dict[word] = word_freq_dict.get(word, 0) + 1
                except:
                    pass
                p.update(1)
        # word_freq_dict = sorted(word_freq_dict.items(), key=lambda d:d[1], reverse = True)
        # print(word_freq_dict[:40])
        # return 

        for k, v in word_freq_dict.items():
            if v >= self.conf.word_freq_threshold:
                word_dict[k] = len(word_dict) + 1
        #print(word_dict.items())
    
        pd.DataFrame(word_dict.items(), columns=['word', 'index']).to_csv(os.path.join(config.data_path, 'word_dict.csv'),
                                                                            sep='\t', index=False)
        

        n_words = len(word_dict) + 1
        
        print('\ttotal_word:', len(word_dict))
        print('\tRemember to  n_words = {} '.format( len(word_dict) + 1))
        
        def get_title_word_idxs(x):
            title = []
            try:
                for i, word in enumerate(tokenizer.tokenize(x.lower())):
                    #word=word.strip('-+=.\'')
                    if word in word_dict:
                        title.append(word_dict[word])
                title=title[:self.conf.n_words_title]+[0 for _ in range(self.conf.n_words_title-len(title))]
            except:
                title = [0 for _ in range(self.conf.n_words_title)]

            return title
        def get_abst_word_idxs(x):
            abstract=[]
            try:
                for i, word in enumerate(tokenizer.tokenize(x.lower())):
                    if word in word_dict:
                        abstract.append(word_dict[word])
                abstract=abstract[:self.conf.n_words_abst]+[0 for _ in range(self.conf.n_words_abst-len(abstract))]
            except:
                abstract = [0 for _ in range(self.conf.n_words_abst)]

            return abstract
        train_news['Title']=train_news['Title'].apply(lambda x :get_title_word_idxs(x))
        train_news['Abstract']=train_news['Abstract'].apply(lambda x :get_abst_word_idxs(x))

        train_news[['News_ID', 'Title', 'Abstract']].to_csv(os.path.join(config.data_path, 'news_words.csv'),index=False,header=None)
        print('Finish news preprocessing for training')

class Log_Processor(object):
    def __init__(self,config):
        self.BATCH_SIZE=10000
        self.impression_df=None
        self.conf=config
        self._type=0
    
    def _get_dev_label(self):
        dev_behaviors = pd.read_table(os.path.join(self.conf.dev_path,'behaviors.tsv'), header=None, 
                                    names=['user_id','time','history','impressions'])
        dev_behaviors['y_true']=dev_behaviors['impressions'].apply(lambda x : ' '.join([(_[-1]) for _ in x.split(' ') ]) )
        dev_behaviors[['user_id','y_true']].to_csv(os.path.join(self.conf.data_path, 'dev_behaviors.csv'), index=False)
    
    def _get_small_dev_label(self):
        dev_behaviors = pd.read_table(os.path.join(self.conf.small_dev_path,'behaviors.tsv'), header=None, 
                                    names=['user_id','time','history','impressions'])
        dev_behaviors['y_true']=dev_behaviors['impressions'].apply(lambda x : ' '.join([(_[-1]) for _ in x.split(' ') ]) )
        dev_behaviors[['user_id','y_true']].to_csv(os.path.join(self.conf.data_path, 'small_dev_behaviors.csv'), index=False)

    def _count_news_ids(self):
        data_paths=[self.conf.train_path,self.conf.dev_path,self.conf.test_path,self.conf.small_train_path,self.conf.small_dev_path]
        df=pd.read_csv(self.conf.train_path+'behaviors.tsv',header=None,sep='\t',
                                names=['user_id','time','history','impressions'] )
        train_ids=set()
        # for h in tqdm(df['history'].unique()):
        #     if isinstance(h,float):
        #         continue
        #     for _id in h.split(' '):
        #         train_ids.add(_id)

        for h in tqdm(df['impressions'].tolist()):
            for _id in h.split(' '):
                train_ids.add(_id[:-2])
        print('train news nums:',len(train_ids))

  

        dev_df=pd.read_csv(self.conf.dev_path+'behaviors.tsv',header=None,sep='\t',
                                names=['user_id','time','history','impressions'] )

        dev_ids=set()
        # for h in tqdm(dev_df['history'].unique()):
        #     if isinstance(h,float):
        #         continue
        #     for _id in h.split(' '):
        #         dev_ids.add(_id)
        for h in tqdm(dev_df['impressions'].tolist()):
            for _id in h.split(' '):
                dev_ids.add(_id[:-2])

        print('dev news nums:',len(dev_ids))
        print('new doc:',len(dev_ids-train_ids))

        # test_ids=set()
        # dev_df=pd.read_csv(self.conf.test_path+'behaviors.tsv',header=None,sep='\t',
        #                         names=['user_id','time','history','impressions'] )
        # for h in tqdm(dev_df['impressions'].tolist()):
        #     for _id in h.split(' '):
        #         dev_ids.add(_id[:-2])
        # print('train news nums:',len(train_ids))



                            

    @log_exec_time
    def build_dataset(self,_type=0):
        """
        select the proper logs and build the basic data format 

        train_type:  每条数据：history取最新条记录id， imp: 取1个正例+4个负例
        数据集结构：每条impression数据用一个嵌套list存储[[历史新闻序列],[点击序列]]: 
            -  历史新闻序列(list格式，<=15); 
            -  点击序列(list格式，列表每个元素为正例+负例的形式(10个元素))：[[pos1,neg1,...],[pos2,neg2,....],......,[]] 

        """
        # 0 1 2 3 4 
        data_paths=[self.conf.train_path,self.conf.dev_path,self.conf.test_path,self.conf.small_train_path,self.conf.small_dev_path]

        self.impression_df=pd.read_table(data_paths[_type]+'behaviors.tsv',header=None,sep='\t',
                                names=['user_id','time','history','impressions'] )
                            
        # 训练集的数据需要删除空行/ 验证集数据进行简单填充
        old_type=_type#s.copy()
        _type=_type%3

        if _type==0:
            self.impression_df=self.impression_df.dropna()
        else:
            self.impression_df.fillna(method='backfill',inplace=True)
        self._type=_type

        num_batch=self.impression_df.shape[0]//self.BATCH_SIZE+1
        random.seed(config.random_seed)
        
        pool = Pool(cpu_count()//4)
        res = pool.map(self._build_data_sample, range(num_batch))
        pool.close()
        pool.join()
        print(old_type)
        datas=[self.conf.train_data,self.conf.dev_data,self.conf.test_data,'small_train.pkl','small_dev.pkl']
        
    
        #res=get_Train_Samples(impression_df.iloc[:100])
        with open(self.conf.data_path+datas[old_type],'wb') as f:
            pickle.dump(res,f)
    
    def _build_data_sample(self,_id):
        df=self.impression_df.iloc[_id*self.BATCH_SIZE:(_id+1)*self.BATCH_SIZE]
        
        user_ids=df['user_id'].tolist()

        his=df['history'].apply(lambda x: x.split(' ')[-self.conf.history_len:])

        imps=df['impressions'].apply(lambda x: x.split(' ')).tolist()
        
        res=[]
        for i,_h in enumerate(his.tolist()):
            tmp=[]
            tmp.append(user_ids[i])
            tmp.append(_h)
            if self._type==0:
                
                neg_idx=[_[:-2] for _ in imps[i] if _[-1]=='0']
                pos_idx=[_[:-2] for _ in imps[i] if _[-1]=='1' ]
                neg_lens=len(neg_idx)
                random.shuffle(neg_idx)
                imps_list=[]
                for i,_p in enumerate(pos_idx):
                    imps_list.append([_p]+neg_idx[i*self.conf.sample_size:(i+1)*self.conf.sample_size])                
                tmp.append(imps_list)

            elif self._type==1: # dev dataset

                tmp.append([[_[:-2] for _ in imps[i]]])

            else:

                tmp.append([[_ for _ in imps[i]]])
            
            res.append(tmp)

    
        return   res


if __name__=='__main__':

    config=Config()
    #config.train_data='list_train_datas.pkl'
    #config.sample_size=15
    NP=News_Processor(config)
    #NP=Demo_News_Processor(config)
    NP._count_news_words()
    # NP._get_word_embeds()
    #NP._get_bert_embeds()
    #news_info=NP._get_all_news_info()
    
    # LP=Log_Processor(config)
    # LP.build_dataset(4)
    # LP.build_dataset(3)



