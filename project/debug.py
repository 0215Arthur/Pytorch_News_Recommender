import pickle
from data_handler import MyDataset
from config import Config


config = Config()
config.__MIND__()
MyDataset(config,None, "MIND")
 
with open("./dataset_processed/MIND/train_datas.pkl", 'rb') as f:
    train_data=pickle.load(f)

print(len(train_data))
# for _ in train_data:
#     print(_)
# val 样本量 278093
# test 样本量 279013
# train 样本量 994540