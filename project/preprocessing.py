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
import json
import gc
import pandas as pd 
#from sklearn.utils import shuffle
import random 
 
from nltk.tokenize import word_tokenize,RegexpTokenizer
from string import digits
from multiprocessing import Pool
from multiprocessing import cpu_count 
from joblib import Parallel, delayed
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
        word_dict = dict(pd.read_csv(os.path.join(self.conf.data_path, self.dataset + '/word_dict.csv'), sep='\t', na_filter=False, header=0).values.tolist())
        # print(word_dict)
        # glove_embedding = pd.read_table('./data/glove.840B.300d.txt', sep=' ', header=None, index_col=0, quoting=3)
        embeds={}
        with open('./data/glove.840B.300d.txt', 'r') as f:
            for i, line in tqdm(enumerate(f)):
                array = line.strip().split(' ')
                if array[0] in word_dict:
                    embeds[array[0]] = array[1:]

        embedding_result = np.random.normal(size=(len(word_dict) , 300))
        embedding_result=np.concatenate((np.zeros((1,300)),embedding_result))
        print(embedding_result.shape)
        word_missing = 0

        with tqdm(total=len(word_dict), desc="Generating word embedding") as p:
            for k, v in word_dict.items():
            # print(k,v)
                if k in embeds:
                    embedding_result[v] = embeds[k]
                else:
                    word_missing += 1
                p.update(1)
        print('\ttotal_missing_word:', word_missing)
        print('Finish word embedding')
        
        np.savez_compressed(os.path.join(self.conf.data_path, self.dataset + '/' + self.conf.word_embedding_pretrained), embeddings=embedding_result)
        print('npz saved to ./data_processed/{}/{}'.format(self.dataset, self.conf.word_embedding_pretrained))

    def _update_id(self,df,field):
        labels = pd.factorize(df[field])[0]
        kwargs = {field: labels}
        df = df.assign(**kwargs)
        return df
    
    @log_exec_time
    def _get_all_news_info(self,cols=['News_ID','Category','SubCategory']):
        """
        params:
              cols: the detail columns (list)
        return all news info (DataFrame format)
        """
        cols = ['News_ID','Category','SubCategory',
                                                'Title','Abstract',
                                                'Title Entities','Abstract Entites']
        news_info=pd.read_table(self.train_path+'news.tsv',header=None,sep='\t',
                                    names=['News_ID','Category','SubCategory',
                                            'Title','Abstract','URL',
                                            'Title Entities','Abstract Entites'])[cols]
        news_info_2=pd.read_table(self.dev_path+'news.tsv',header=None,sep='\t',
                                        names=['News_ID','Category','SubCategory',
                                                'Title','Abstract','URL',
                                                'Title Entities','Abstract Entites'])[cols]

        news_info=pd.concat([news_info_2,news_info])
        def parse_entity(x):
            if isinstance(x, float):
                return None
            x=json.loads(x)#["WikidataId"]
            entities=[]
            for _ in x:
                if _["Confidence"] > 0.6:
                    entities.append(_["WikidataId"])
            return entities
        news_info.drop_duplicates(['News_ID' ],inplace=True,keep='first')
        print('train_set + dev_set : the number of News',news_info.shape)
        news_info=news_info.dropna(axis=0,subset = ["Title"])
        #print(new_lens-old_lens)
        print(news_info.shape[0])
        news_info=self._update_id(news_info, "Category")
        news_info=self._update_id(news_info, "SubCategory")
        news_info["Title Entities"]=news_info["Title Entities"].apply(lambda x: parse_entity(x))
        news_info["Abstract Entites"]=news_info["Abstract Entites"].apply(lambda x: parse_entity(x))

        return news_info[cols]
    
    def _count_news_words(self):
        """
        func: count the words of the all news to remap the words to IDs.
        """
        train_news=self._get_all_news_info()
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
                    word=word.strip('-+=.\'')
                    if word not in self.stop_words:
                        word_freq_dict[word] = word_freq_dict.get(word, 0) + 1
                try:
                    for word in tokenizer.tokenize(row.Abstract):
                        word=word.strip('-+=.\'')
                        if word not in self.stop_words:
                            word_freq_dict[word] = word_freq_dict.get(word, 0) + 1
                except:
                    pass
                p.update(1)
        for k, v in word_freq_dict.items():
            if v >= self.conf.word_freq_threshold:
                word_dict[k] = len(word_dict) + 1
    
        pd.DataFrame(word_dict.items(), columns=['word', 'index']).to_csv(os.path.join(self.conf.data_path, self.dataset + '/word_dict.csv'),
                                                                            sep='\t', index=False)
        

        n_words = len(word_dict) + 1
        
        print('\ttotal_word:', len(word_dict))
        print('\tRemember to  n_words = {} '.format( len(word_dict) + 1))
        
        def get_title_word_idxs(x):
            title = []
            try:
                for i, word in enumerate(tokenizer.tokenize(x.lower())):
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
        tl_e = train_news['Title Entities'].tolist()
        ab_e = train_news['Abstract Entites'].tolist()
        entities = []
        for i, _ in enumerate(tl_e):
            if _ is None or len(_) < 1 :
                entities.append(ab_e[i])
                continue
            if ab_e[i] is None or len(ab_e[i]) < 1:
                entities.append(_)
                continue
            entities.append(list(set(_) | set(ab_e[i])))
        train_news["Entities"] = entities
        train_news["Category"] += 1
        train_news["SubCategory"] += 1
        train_news[['News_ID', 'Title', 'Abstract', 'Category','SubCategory', 'Entities']]\
            .to_csv(os.path.join(self.conf.data_path,  self.dataset + '/news.csv'),index=False)
        print("Category num ", train_news["Category"].max())
        print("SubCategory num ", train_news["SubCategory"].max())
        # 为方便取数据，将其改为json格式存储
        train_news=train_news[['News_ID', 'Title', 'Abstract', 'Category','SubCategory', 'Entities']]
        train_news=train_news.reset_index()
        train_news["index"] += 1
        # print(train_news.columns)
        news_json = json.loads(train_news.set_index("News_ID").to_json(orient='index'))
        print(type(news_json))
        with open(os.path.join(self.conf.data_path, self.dataset + "/news.pkl"),'wb') as f:
            pickle.dump(news_json,f)
        print('Finish news preprocessing for training')

class MIND_Log_Processor(object):
    def __init__(self,config):
        self.BATCH_SIZE=50000
        self.conf=config
        self.data_type=0
    
    def _get_dev_label(self):
        dev_behaviors = pd.read_table(os.path.join(self.conf.dev_path,'behaviors.tsv'), header=None, 
                                    names=['user_id','time','history','impressions'])
        dev_behaviors['y_true']=dev_behaviors['impressions'].apply(lambda x : ' '.join([(_[-1]) for _ in x.split(' ') ]) )
        dev_behaviors[['user_id','y_true']].to_csv(os.path.join(self.conf.data_path, 'dev_behaviors.csv'), index=False)                            

    @log_exec_time
    def build_dataset(self,train_ratio = 0.3):
        """
        select the proper logs and build the basic data format 
        K: 负采样数量 
        :  每条数据：history取最新条记录id， imp: 取1个正例+4个负例
        数据集结构：每条impression数据用一个嵌套list存储[[历史新闻序列],[点击序列]]: 
            -  历史新闻序列(list格式，<=15); 
            -  点击序列(list格式，列表每个元素为正例+负例的形式(1+K个元素))：[[pos1,neg1,...],[pos2,neg2,....],......,[]] 

        """
        # 0 1 2 3 4 

        impression_df=pd.read_table('./data/MIND/train/behaviors.tsv',header=None,sep='\t',
                                names=['user_id','time','history','impressions'] )
        
        dev_df=pd.read_table('./data/MIND/dev/behaviors.tsv',header=None,sep='\t',
                                names=['user_id','time','history','impressions'] )

        impression_df=impression_df.dropna().reset_index(drop=True)
        dev_df=dev_df.dropna().reset_index(drop=True)
        train_df=impression_df.sample(frac=train_ratio).reset_index(drop=True)
        random.seed(config.random_seed)
        print(len(train_df))
        print(len(dev_df))
        print("build dataset...")
        # pool = Pool(cpu_count()//2) #cpu_count()//2
        # res = pool.map(self._build_data_sample, range(num_batch))
        # pool.close()
        # pool.join()
        res = Parallel(n_jobs=16)(
            delayed(self._build_data_sample)(train_df, i, 0) for i in range(train_df.shape[0]//self.BATCH_SIZE+1)
        )
        val_num = int(0.5 * len(dev_df))
        val_df = dev_df[:val_num]
        test_df = dev_df[val_num:]
        print("build val dataset...")
        val_res = Parallel(n_jobs=16)(
            delayed(self._build_data_sample)(val_df, i, 1) for i in range(val_df.shape[0]//self.BATCH_SIZE+1)
        )

        test_res = Parallel(n_jobs=16)(
            delayed(self._build_data_sample)(test_df, i, 1) for i in range(test_df.shape[0]//self.BATCH_SIZE+1)
        )
        with open(self.conf.data_path+self.conf.test_data,'wb') as f:
            pickle.dump(self.parse_data(test_res, 1),f)

        with open(self.conf.data_path+self.conf.train_data,'wb') as f:
            pickle.dump(self.parse_data(res),f)

        with open(self.conf.data_path+self.conf.val_data,'wb') as f:
            pickle.dump(self.parse_data(val_res, 1),f)
        


    def _build_data_sample(self, df, _id, data_type):
        """
        [user_id, [his_list], [[cand_list],[cand_list]]]
        """
        df=df.iloc[_id*self.BATCH_SIZE:(_id+1)*self.BATCH_SIZE]
        
        user_ids=df['user_id'].tolist()

        his=df['history'].apply(lambda x: x.split(' ')[-self.conf.history_len:])

        imps=df['impressions'].apply(lambda x: x.split(' ')).tolist()
        
        res=[]
        for i,_h in enumerate(his.tolist()):
            tmp=[]
            tmp.append(user_ids[i])
            tmp.append(_h)
            neg_idx=[_[:-2] for _ in imps[i] if _[-1]=='0']
            pos_idx=[_[:-2] for _ in imps[i] if _[-1]=='1' ]
            # random.shuffle(neg_idx)
            if data_type == 0:
                sample_size = self.conf.negsample_size
            else:
                sample_size = self.conf.max_candidate_size - 1
            imps_list=[]
            for i,_p in enumerate(pos_idx):
                imps_list.append([_p]+neg_idx[i*sample_size:(i+1)*sample_size])                
            tmp.append(imps_list)
            res.append(tmp)
        if _id % 5 == 0 and _id > 0:
            print("precoessed {} samples".format(_id * self.BATCH_SIZE))
        return   res
    
    def parse_data(self,data,type=0):
        """
        将数据拆分为训练格式
        ([historical news_ids],[impression_ids])
        """
        final_samples=[]
        for _ in data:
            for row in _:
                if type == 0 and len(row[1]) < 5:
                    continue
                for samples in row[2]:
                    final_samples.append((row[1], samples))
                    if len(final_samples) == 10000000:
                        return final_samples
        return final_samples



if __name__=='__main__':

    config=Config()
    config.__MIND__()
    #config.train_data='list_train_datas.pkl'
    #config.sample_size=15
    NP=News_Processor(config)
    #NP=Demo_News_Processor(config)
    # NP._count_news_words()
    #NP._get_word_embeds()
    
    LP=MIND_Log_Processor(config)
    LP.build_dataset()
    # LP.build_dataset(3)



