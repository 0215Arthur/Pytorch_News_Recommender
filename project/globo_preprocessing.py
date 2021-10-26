from ast import literal_eval
import os
from pandas.core import groupby
import torch
import numpy as np

from tqdm import tqdm
import time
from datetime import timedelta
from collections import Counter
import pickle
import json
import gc
import pandas as pd 
import datetime
#from sklearn.utils import shuffle
import random 

 
from nltk.tokenize import word_tokenize,RegexpTokenizer
from string import digits
from multiprocessing import Pool
from multiprocessing import cpu_count 
from joblib import Parallel, delayed

root_path = "./data/Globo/"

def merge_log(df, start, mid, end):
    """
    指定时间戳，进行窗口内合并操作
    """
    # print(start, mid, end)
    his_df = df[(df["click_timestamp"] >= start * 1000) & (df["click_timestamp"] < mid * 1000)].groupby("user_id")["click_article_id"].agg(list)
    his_df = his_df.reset_index()
    print(his_df.shape)

    target_df = df[(df["click_timestamp"] >= mid * 1000) & (df["click_timestamp"] <= end * 1000)].groupby("user_id")["click_article_id"].agg(list)
    target_df = target_df.reset_index()
    print(target_df.shape)

    res_df = pd.merge(target_df, his_df, how='inner', on='user_id')
    print(res_df.shape)
    return res_df

def process_log():
    """
    func: 整合globo的点击记录
    从中取7day数据构建用户的历史记录，预测来下一天
    取14天数据， 按照滑动窗口的方法进行单条数据构建
    单条数据： [historical news],[read news]
    """
    hours = []
    for i in range(336):
        hour = str(i).zfill(3)
        df = pd.read_csv(root_path + "clicks/clicks/clicks_hour_{}.csv".format(hour))
        hours.append(df)
    df = pd.concat(hours)
    start_time = time.mktime(time.strptime("2017-10-01", '%Y-%m-%d'))
    res = []
    for i in range(7):
        mid = datetime.datetime.fromtimestamp(start_time) + datetime.timedelta(days = 7)
        mid = (time.mktime(mid.timetuple()))
        end = datetime.datetime.fromtimestamp(start_time) + datetime.timedelta(days = 8)
        end = (time.mktime(end.timetuple()))
        res.append(merge_log(df, start_time, mid, end))
        # break
        start_time = datetime.datetime.fromtimestamp(start_time) + datetime.timedelta(days = 1)
        start_time = (time.mktime(start_time.timetuple()))
    res_df = pd.concat(res)
    res_df.columns = ["user_id", "historical", "target"]
    res_df.to_csv("./dataset_processed/Globo/data.csv",index=False)

def split_data(val_ratio = 0.1, test_ratio = 0.1):
    """
    news id重新映射
    user_id, [historical], target_news_id

    21481 条新闻
    """
    df = pd.read_csv("./dataset_processed/GLOBO/data.csv")
    print(df.shape)
    users = df["user_id"].tolist()
    targets = df["target"].tolist()
    res = []
    news_ids = Counter()
    user_ids = set()
    hist_cnt = []
    for i, line in tqdm(enumerate(df["historical"].tolist())):
        hist = literal_eval(line)
        if len(hist) < 5:
            continue 
        target = literal_eval(targets[i])
        news_ids.update(target)
        news_ids.update(hist)
        tmp = []
        tmp.append(users[i])
        user_ids.add(users[i])
        tmp.append(hist)
        tmp.append(target)
        res.append(tmp)
        hist_cnt.append(len(hist))
    # print(pd.DataFrame(hist_cnt).describe())
    # 历史记录长度
    # mean       7.551559
    # std        4.104858
    # min        5.000000
    # 25%        5.000000
    # 50%        6.000000
    # 75%        8.000000
    # max      141.000000
        
    print("news num: ", len(news_ids))
    print("user num: ", len(user_ids)) # 38363
    user_dict = dict(zip(list(user_ids), range(len(user_ids))))
    news_dict = dict(zip(news_ids.keys(), range(1, len(news_ids) + 1)))
    final_res = []
    for line in tqdm(res):
        tmp = [user_dict[line[0]],\
                        [news_dict[_] for _ in line[1]]]
        pos = [news_dict[_] for _ in line[2]]
        for _ in pos[:min(len(line[1]), 5)]:
            final_res.append(tmp + [_] + [pos])
            
    
    print(len(final_res))
    data_num = len(final_res)
    val_num = int (data_num * val_ratio)
    test_num = int (data_num * test_ratio)
    train_num = data_num - val_num - test_num
    
    train_res = Parallel(n_jobs=8)(
            delayed(neg_sampling)(sample, i, K = 3) for i, sample in enumerate((final_res[:train_num]))
        )
    
    val_res = Parallel(n_jobs=8)(
            delayed(neg_sampling)(sample, i, K = 199) for i, sample in enumerate(final_res[train_num: train_num + val_num])
        )

    test_res = Parallel(n_jobs=8)(
            delayed(neg_sampling)(sample, i, K = 199) for i, sample in enumerate(final_res[train_num + val_num:])
        )
    

    with open("./dataset_processed/GLOBO/train_datas.pkl",'wb') as f:
        pickle.dump(train_res, f)


    with open("./dataset_processed/GLOBO/val_datas.pkl",'wb') as f:
        pickle.dump(val_res, f)

    with open("./dataset_processed/GLOBO/test_datas.pkl",'wb') as f:
        pickle.dump(test_res, f)

    with open("./dataset_processed/GLOBO/news.json", 'w') as f:
        json.dump(news_dict, f)

    print("train_num: ", train_num)
    print("val_num: ", val_num)
    print("test_num: ", test_num)
    # train_num:  233180
    # val_num:  29147
    # test_num:  29147


def neg_sampling(sample, i, news_num=21482, K=3):
    # print(sample)
    exclusion = sample[1] + [sample[3]]
    # print("neg_sampling")
    neg_samples = random.sample([_ for _ in range(1, news_num) if _ not in exclusion], K)
    # print(neg_samples)
    if i % 10000 == 0 and i > 0:
        print("processed {} samples".format(i))
    return [sample[0], sample[1], [sample[2]] + neg_samples]

def load_article_embedding():
    """
    从原始数据所给的embedding中，选取训练数据所涉及到的article
    """
    with open("./data/Globo/articles_embeddings.pickle",'rb') as f:
        embed = pickle.load(f)
    with open("./dataset_processed/Globo/news.json", 'r') as f:
        news = json.load(f)

    embedding_result = embed[[int(_) for _ in list(news.keys())], :] 
    #np.random.normal(size=(1 , 250))
    embedding_result=np.concatenate((np.zeros((1,250)),embedding_result))
    # print((embed[[0,1,2,3], :]).shape)
    np.savez_compressed("./dataset_processed/Globo/article_embed.npz", embeddings=embedding_result)


if __name__ == "__main__":
    # process_log()
    split_data()
    # load_article_embedding()