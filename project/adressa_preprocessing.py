"""
预处理Adressa数据
"""
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
from gensim.models import Word2Vec
import logging
from collections import Counter


data="ADR"
def word_embedding(file="./data/ADR/articles_filtered.json"):
    """
    训练生成word embedding
    title平均长度： 6.822279746496474
    类别数量：18
    子类别数量：102
    新闻数量： 
    """
    with open(file, 'r') as f:
        articles = json.load(f)
    title_lens=0
    categ_cnt={}
    subcateg_cnt={}
    sentences=[]
    words=Counter()
    print(len(articles))
    for _ in articles:
        #print(articles[_]["title"])
        sentences.append(articles[_]['title'])
        words.update(articles[_]['title'])
        # title_lens+=len(articles[_]["title"])
        categ = articles[_]["category"].split('|')
        if len(categ) > 1:
            categ_cnt.setdefault(categ[0], 0)
            subcateg_cnt.setdefault(categ[1], 0)
            categ_cnt[categ[0]]+=1
            subcateg_cnt[categ[1]]+=1
    word_num = len(words)
    word_dict = dict(zip(words.keys(), range(1, word_num + 1)))
    categ_dict = dict(zip(categ_cnt.keys(), range(1, len(categ_cnt) + 1)))
    subcateg_dict = dict(zip(subcateg_cnt.keys(), range(1, len(subcateg_cnt) + 1)))
    # 单词重新映射 从1编码
    print("vocab size:", word_num + 1)
    print("categ size:", len(categ_cnt) + 1)
    print("subcateg size:", len(subcateg_cnt) + 1)
    print("article size:", len(articles))
    # print(title_lens/len(articles))
    # print(len(categ_cnt),categ_cnt)
    # print(len(subcateg_cnt),subcateg_cnt)
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = Word2Vec(sentences,\
                    min_count=1,\
                    sg=1,\
                    vector_size=50,\
                    epochs =20)
    # word_vectors = model.wv
    # print(len(word_vectors))
    # for _ in word_vectors:
    #     print(_)
    # print(model.wv.vocab)
    model.save("./data/ADR/word2vec.model")
    word_mat = np.zeros((word_num + 1, 50))
    for word in word_dict:
        word_mat[word_dict[word]] = model.wv[word]

    np.savez_compressed("./dataset_processed/ADR/all_word_embedding.npz", embeddings=word_mat)
    # print(word_vectors)
    # update origin json
    for i, _ in enumerate(articles):
        articles[_]["titleid"] = [word_dict[w] for w in articles[_]['title']]
        categ = articles[_]["category"].split('|')
        articles[_]['categoryid'] = 0
        articles[_]['subcategoryid'] = 0
        if len(categ) > 1:
            articles[_]['categoryid'] = categ_dict[categ[0]]
            articles[_]['subcategoryid'] = subcateg_dict[categ[1]]
    
    with open("./dataset_processed/ADR/articles.json", 'w') as f:
        json.dump(articles, f)


def split_data():
    """
    数据分割
    前5天数据构成训练集
    前6天数据
    平均历史长度为14.5598
    """
    df = pd.read_csv("./data/ADR/one_week_events.csv")
    print(df.shape)
    # 过滤操作
    with open("./dataset_processed/ADR/articles.json", 'r') as f:
        articles = json.load(f)
    df = df[df["newsUrl"].isin(list(articles.keys()))]
    print(df.shape)
    vc = df["userId"].value_counts() 
    vc = vc[vc >= 5]
    df = df.loc[df["userId"].isin(vc.index)]
    print(df.shape)
    df = df.sort_values(by="time").reset_index(drop=True)
    
    df['last'] = df.groupby('userId')['time'].diff(1)
    df.fillna(0, inplace=True)
    # news_reid = dict(zip(df["newsUrl"].unique().tolist(), range(1, df["newsUrl"].nunique() + 1)))
    # print(df.describe())
    # df=df.replace(news_reid)
    labels, uniques = pd.factorize(df['newsUrl'])
    ulabels, _ = pd.factorize(df['userId'])
    kwargs = {'newsUrl': labels, "userId":ulabels}
    news_reid = {oid: i + 1 for i, oid in enumerate(uniques)}
    df = df.assign(**kwargs)
    df["userId"] = df["userId"] + 1
    df["newsUrl"] = df["newsUrl"] + 1
    
    hist_data_train = df[df["day"] < 20170106].groupby("userId")["newsUrl"].agg(list).reset_index()
    interval_train = df[df["day"] < 20170106].groupby("userId")["last"].agg(list).reset_index()
    hist_data_test = df[df["day"] < 20170107].groupby("userId")["newsUrl"].agg(list).reset_index()
    interval_test = df[df["day"] < 20170107].groupby("userId")["last"].agg(list).reset_index()
    hist_data_train.columns=["userId", "hist"]
    hist_data_test.columns=["userId", "hist"]
    df = df[["userId","newsUrl","day"]]
    lens=0
    for _ in hist_data_train["hist"].tolist():
        lens+=len(_)
    
    for _ in hist_data_test["hist"].tolist():
        lens+=len(_)
    print("hist avg len:", lens/(hist_data_train.shape[0] + hist_data_test.shape[0]))
    

    # print(df.describe())
    # print(df["useId"].nunique())
    # print(df.shape)
    train_data = pd.merge(df[df['day'] == 20170106], hist_data_train, how='inner', on='userId')
    train_data = pd.merge(train_data, interval_train, how='inner', on='userId')
    test_data = pd.merge(df[df['day'] == 20170107], hist_data_test, how='inner', on='userId')
    test_data = pd.merge(test_data, interval_test, how='inner', on='userId')
    print(train_data.shape, test_data.shape)
    print("news num:", len(news_reid))
    print("user num:", df["userId"].nunique())
    train_data.to_csv("./dataset_processed/ADR/train.csv", index=False)
    test_data.to_csv("./dataset_processed/ADR/test.csv", index=False)
    train_res = []
    # train_data=train_data[["userId",""]]
    # for i in range()
    news_num = len(news_reid) + 1
    
    
    news_dict={}
    missing_cnt = 0
    for url in news_reid:
        if url not in articles:
            missing_cnt += 1
            continue
        news_dict[news_reid[url]]=articles[url]
    print("missing news count: ", missing_cnt)
    # with open("./dataset_processed/ADR/news.json", 'w') as f:
    #     json.dump(news_dict, f)
    with open(os.path.join( "dataset_processed/ADR/news.pkl"),'wb') as f:
        pickle.dump(news_dict,f)
    # for _ in train_data:
        # userId  hist_list time_list 
    val_num = int(test_data.shape[0] * 0.2)
 
    train_res = Parallel(n_jobs=8)(
            delayed(neg_sampling)(sample, i, news_num = news_num, K = 4) for i, sample in enumerate(train_data.values.tolist())
        )
    
    val_res = Parallel(n_jobs=8)(
            delayed(neg_sampling)(sample, i, news_num = news_num, K = 99) for i, sample in enumerate(test_data.values.tolist()[:val_num])
        )

    test_res = Parallel(n_jobs=8)(
            delayed(neg_sampling)(sample, i, news_num = news_num, K = 99) for i, sample in enumerate(test_data.values.tolist()[val_num:])
        )
    

    with open("./dataset_processed/ADR/train_datas.pkl",'wb') as f:
        pickle.dump(train_res, f)


    with open("./dataset_processed/ADR/val_datas.pkl",'wb') as f:
        pickle.dump(val_res, f)

    with open("./dataset_processed/ADR/test_datas.pkl",'wb') as f:
        pickle.dump(test_res, f)



    print("train_num: ", train_data.shape[0])
    print("val_num: ", val_num)
    print("test_num: ", test_data.shape[0] - val_num)


def neg_sampling(sample, i, news_num=21482, K=3):
    # print(sample)
    """
    随机负采样：
    userId, hist_list, time_list, cand_list
    """
    exclusion = [sample[1]] + [sample[3]]
    # print("neg_sampling")
    neg_samples = random.sample([_ for _ in range(1, news_num) if _ not in exclusion], K)
    # print(neg_samples)
    if i % 10000 == 0 and i > 0:
        print("processed {} samples".format(i))
    return [sample[0], sample[3], sample[4], [sample[1]] + neg_samples]




if __name__=="__main__":
    # word_embedding()
    split_data()
