import sys, os, os.path
import numpy as np
import json
from sklearn.metrics import roc_auc_score

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    #print(order)
    y_true = np.take(y_true, order[:k])
    # print(y_true)
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)
    

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
   
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    # print(np.where(np.array(y_score) <= y_score[0], 0, 1).sum())
    return np.sum(rr_score) / np.sum(y_true)

def auc_score(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


if __name__=="__main__":
    score = ndcg_score([1,0,0,0,0,0,0,0,0,0],[0.5, 0.6,0.3,0.4, 0.8, 0.8, 0.8, 0.8,0.8,0.8],k=5)
    auc = auc_score([1,0,0,0,0,0,0,0,0,0],[0.8, 0.6,0.3,0.4, 0.8, 0.8, 0.8, 0.8,0.8,0.8])
    mrr = mrr_score([1,0,0,0,0,0,0,0,0,0],[0.8, 0.6,0.3,0.4, 0.8, 0.8, 0.8, 0.8,0.8,0.8])
    # print(score,auc)
    print(mrr)