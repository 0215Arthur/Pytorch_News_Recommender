import pickle
from data_handler import MyDataset
from config import Config
import torch
import torch.nn.functional as F

a = torch.tensor([1,0.3,0.05])
print(F.gumbel_softmax(a))
print(F.softmax(a))


config = Config()
config.__MIND__()

 
with open("./dataset_processed/MIND/test_datas.pkl", 'rb') as f:
    train_data=pickle.load(f)

print(len(train_data))
nums = 0
for i,_ in enumerate(train_data):
    # print(len(_[1]))
    if (len(_[1]) < 2):
        print(_[1])
    nums += (len(_[1]))
    # if i > 5:
    #     break
print(nums/len(train_data))
# val 样本量 278093  260727
# test 样本量 279013 259552
# train 样本量 994540

with open("./dataset_processed/MIND/news.pkl", 'rb') as f:
    data=pickle.load(f)
print(list(data.keys())[:5])
print(data['N88753'])
print(data['N86255'])