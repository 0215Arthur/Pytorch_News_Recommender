import torch
import os
from config import Config
from data_handler import *
from model import NRMS_V0,NRMS_V1
import argparse
from train_eval import train_demo
from torch.utils.data import Dataset,DataLoader


parser = argparse.ArgumentParser(description='MIND')
#parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--target', default='gender', type=str, help='gender or age')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--test', default=0, type=int, help='gender or age')
args = parser.parse_args()


if __name__=='__main__':
    print('current: uid',os.getpid())
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    #build_dataset(config,_type=0)
    torch.backends.cudnn.deterministic = True
    model_name='NRMS_V0_DEMO'
    config=Config(model_name)
    config.batch_size=256
    print(model_name,config.batch_size)
    config.mode='demo'
    #config.learning_rate=0.01
    config.word_embedding_pretrained='demo_word_embedding.npz'

    dataset=load_dataset(config,'small_train.pkl',config.data_path,_type=0)
    train_data=MyDataset(config,dataset ,type=0)
    train_iter = DataLoader(dataset=train_data, 
                              batch_size=config.batch_size, 
                              num_workers=6,
                              drop_last=False,
                              shuffle=True,
                              pin_memory=False)

     
    
    dev_data=load_dataset(config,'small_dev.pkl',config.data_path,_type=1)
    
    dev_data=MyDataset(config,dev_data ,type=1)
    print("dev_data nums:::",len(dev_data))

    dev_iter = DataLoader(dataset=dev_data, 
                              batch_size=512, 
                              num_workers=6,
                              drop_last=False,
                              shuffle=False,
                              pin_memory=False)


    model=NRMS_V0(config).to(config.device)
    print(model.parameters)
    #init_network(model)
    train_demo(config,model,train_iter,dev_iter)