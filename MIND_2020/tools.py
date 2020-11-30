import time
import functools
import numpy as np
import pandas as pd 
from config import Config
#from bert_serving.client import BertClient
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random
import datetime
from tqdm import tqdm
from datetime import timedelta
import pickle 

# 获取执行时间 装饰器函数
def log_exec_time(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        print('Current Func : {}...'.format(func.__name__))
        start=time.perf_counter()
        res=func(*args,**kwargs)
        end=time.perf_counter()
        print('Func {} took {:.2f}s'.format(func.__name__,(end-start)))
        return res
    return wrapper


def get_entity_embeds(train_path,dev_path):
    embed_df=pd.read_table(train_path+'entity_embedding.vec',header=None,sep='\t' )#[['News_ID','Category','SubCategory']]
    embed_df_2=pd.read_table(dev_path+'entity_embedding.vec',header=None,sep='\t' )#[['News_ID','Category','SubCategory']]
    embed_df.columns=['Q_id']+['_'+str(i) for i in range(101)]
    embed_df_2.columns=['Q_id']+['_'+str(i) for i in range(101)]
    entity_embed_df=pd.concat([embed_df,embed_df_2])

    entity_embed_df.drop_duplicates('Q_id',keep='first',inplace=True)   
    _dict=entity_embed_df[['Q_id']].reset_index() 
    entity_id=_dict.set_index('Q_id')
    entity_id['index']=entity_id['index']+1
    entity_dict=entity_id.T.to_dict('list')
    with open('./data_processed/'+'entity_ids_dict.pkl','wb') as f:
        pickle.dump(entity_dict,f)

    entitiy_embeds=np.concatenate([np.zeros((1,100)),entity_embed_df.iloc[:,1:-1].values])
    print(entitiy_embeds.shape)
    np.savez_compressed('./data_processed/entitiy_embeds.npz', embeddings=entitiy_embeds)
    print('npz saved to ./data_processed/entitiy_embeds.npz')






def plot_loss(log_path,loss_list,step_size):
   # t=time.time()
    if not  os.path.exists(log_path):
        os.mkdir(log_path) 

    print('Plotting loss figure...')
    plt.plot([_*step_size for _ in range(len(loss_list))], loss_list)
    plt.savefig(os.path.join(log_path, time.strftime("%Y-%m-%d%H:%M:%S",time.localtime())+'loss.png'))
        
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))



"""
pandas 处理大文件
修改df中各特征的存储格式，显著降低内存占用

"""
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)


    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df



if __name__=='__main__':
   # process_behaviors(args)
    #get_News_Embeds('../MIND/train/','../MIND/dev/')
    get_entity_embeds('../MIND/train/','../MIND/dev/')

