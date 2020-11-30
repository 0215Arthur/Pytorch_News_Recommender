"""
xxx
"""
import torch
import os
from config import Config
from data_handler import *

import argparse
from train_eval import train,test
from torch.utils.data import Dataset,DataLoader
import time
from model import Model

parser = argparse.ArgumentParser(description='MIND')

parser.add_argument('--model', type=str, required=True, help='choose the proper model')
parser.add_argument('--dataset', default='large', type=str, required=True,help='large dataset or demo dataset')
parser.add_argument('--test', default=False, type=bool, help='run the test dataset')
#parser.add_argument('--test', default=True, type=bool, help='run the test dataset')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--load', type=str, default=None,
                    help='load the pretrained model ckpt file')
parser.add_argument('--description', type=str, default=None,
                    help='the basic description of current experiment')             

args = parser.parse_args()


if __name__=='__main__':
    print('current: uid',os.getpid())
    torch.manual_seed(422)
    if args.description is None:
        model_name=args.model+'_'+time.strftime('%m-%d_%H')
    else:
        model_name=args.model+'_'+args.description

    torch.cuda.manual_seed_all(422)
    #build_dataset(config,_type=0)
    torch.backends.cudnn.deterministic = True

    config=Config(model_name)
    if args.model=='list_rank':
        config.sample_size=15
    config.batch_size=512
    config.num_epochs=6
    #config.word_embedding_pretrained='all_word_embedding_v3.npz'
    config.mode=args.dataset
    config.__nrms__()
    recommender=Model(config,args)
    print(model_name)

    print(config.device)
    #print(config.__nrms__())
    
  
    #model=NRMS_V1(config).to(config.device)
    print(recommender.parameters)
    #init_network(model)
    if not args.test:
        if args.model=='list_rank':
            train_data='list_train_datas.pkl'
        else:
            train_data='train_datas.pkl'


        dataset=load_dataset(config,train_data,config.data_path,_type=0)
        train_data=MyDataset(config,dataset ,type=0)
        train_iter = DataLoader(dataset=train_data, 
                                batch_size=config.batch_size, 
                                num_workers=6,
                                drop_last=False,
                                shuffle=True,
                                pin_memory=False)
        
        dev_data=load_dataset(config,'dev_datas.pkl',config.data_path,_type=1)
        
        dev_data=MyDataset(config,dev_data[:100000] ,type=1)

        dev_iter = DataLoader(dataset=dev_data, 
                                batch_size=config.batch_size, 
                                num_workers=6,
                                drop_last=False,
                                shuffle=False,
                                pin_memory=False)



        train(config,recommender,train_iter,dev_iter)
    

    if args.test:
        # print('In this mode, we will train and evaluate the model........')
        # print('begin training the recommender...')
        # train(config,recommender,train_iter,dev_iter)
        print('evaluating....')
        recommender.load_state_dict(torch.load(config.save_path+args.load))
        test_data=load_dataset(config,'test_datas.pkl',config.data_path,_type=1)
    
        test_data=MyDataset(config,test_data ,type=1)

        test_iter = DataLoader(dataset=test_data, 
                                batch_size=config.batch_size, 
                                num_workers=6,
                                drop_last=False,
                                shuffle=False,
                                pin_memory=False)
        test(config,
             recommender,
             test_iter,
             ckpt_file=args.load)