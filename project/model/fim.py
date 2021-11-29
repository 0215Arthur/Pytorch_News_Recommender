'''
Description: Implementation of Finegrained Interest Matching method for neural news recommendation 
'''

import torch
import math
import torch.nn as nn
import numpy as np
import torchsnooper


class Model(nn.Module):
    def __init__(self,config):
        super().__init__()

        if config.dataset == "GLOBO":
            self.pretrained = True
            try:
                self.article_embedding = nn.Embedding.from_pretrained(torch.tensor(
                    np.load(config.data_path + config.article_embedding_pretrained)["embeddings"].astype('float32')), 
                    freeze=True,padding_idx=0).to(config.device)
                print(self.article_embedding.weight.size())
            except:
                print("load article embedding error.")
        #self.category_embedding =nn.Embedding(config.category_nums, config.cate_embed_size, padding_idx=0)
        #self.subcategory_embedding =nn.Embedding(config.subcategory_nums, config.cate_embed_size, padding_idx=0)
        else:
            if config.word_embedding_pretrained is not None:
                self.embedding = nn.Embedding.from_pretrained(torch.tensor(
                    np.load(config.data_path +config.word_embedding_pretrained)["embeddings"].astype('float32')), 
                    freeze=False,padding_idx=0).to(config.device)
            else:
                self.embedding = nn.Embedding(config.n_words, config.word_embed_size, padding_idx=0)

        self.cdd_size = (config.npratio + 1) if config.npratio > 0 else 1
        self.batch_size = config.batch_size
        self.level = config.dilation_level
        self.dropout_p = config.dropout
        
        # concatenate category embedding and subcategory embedding
        self.signal_length = config.n_words_title# + 1 + 1
        self.his_size = config.history_len

        self.kernel_size = config.kernel_size
        self.filter_num = config.filter_num
        self.embedding_dim = config.word_embed_size

        self.device = config.device

        # pretrained embedding
        # self.embedding = vocab.vectors.to(self.device)
        # elements in the slice along dim will sum up to 1 
        self.softmax = nn.functional.softmax
        
        self.CNN_d1 = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=1,padding=1)
        self.CNN_d2 = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=2,padding=2)
        self.CNN_d3 = nn.Conv1d(in_channels=self.embedding_dim,out_channels=self.filter_num,kernel_size = self.kernel_size,dilation=3,padding=3)

        self.ReLU = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(self.filter_num)
        self.DropOut = nn.Dropout(p=self.dropout_p)
        self.SeqCNN3D = nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=32,kernel_size=[3,3,3],padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3,3,3],stride=[3,3,3]),
            nn.Conv3d(in_channels=32,out_channels=16,kernel_size=[3,3,3],padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=[3,3,3],stride=[3,3,3])
        )
        
        self.predictor = nn.Linear(320,1)
    
    
    # @torchsnooper.snoop()
    def _HDC(self,news_embedding_set):
        """ stack 1d CNN with dilation rate expanding from 1 to 3
        
        Args:
            news_embedding_set: tensor of [set_size, signal_length, embedding_dim]
        Returns:
            news_embedding_dilations: tensor of [set_size, levels(3), signal_length, filter_num]
        """

        # don't know what d_0 meant in the original paper
        # news_embedding_dilations = torch.zeros((news_embedding_set.shape[0],self.level,self.signal_length,self.filter_num),device=self.device)
        news_embedding_dilations = []
        news_embedding_set = news_embedding_set.permute(0,2,1)

        news_embedding_d1 = self.CNN_d1(news_embedding_set)
        # news_embedding_dilations[:,0,:,:] = self.ReLU(news_embedding_d1)
        news_embedding_dilations.append(self.ReLU(news_embedding_d1).unsqueeze(dim=1))

        news_embedding_d2 = self.CNN_d2(news_embedding_set)
        news_embedding_dilations.append(self.ReLU(news_embedding_d2).unsqueeze(dim=1))
        
        news_embedding_d3 = self.CNN_d3(news_embedding_set)
        news_embedding_dilations.append(self.ReLU(news_embedding_d3).unsqueeze(dim=1))
        #news_embedding_d2 = self.LayerNorm(news_embedding_d2.permute(0,2,1))
        #news_embedding_dilations[:,1,:,:] = self.ReLU(news_embedding_d2)        

        # news_embedding_d3 = self.CNN_d3(news_embedding_set)
        # news_embedding_dilations[:,2,:,:] = self.ReLU(news_embedding_d3)
        news_embedding_dilations = self.LayerNorm(torch.cat(news_embedding_dilations, dim=1).permute(0,1,3,2))
        return news_embedding_dilations
        
    def _news_encoder(self,news_embedding):
        """ encode set of news to news representation
        
        Args:
            news_set: tensor of [set_size, signal_length]
        
        Returns:
            news_embedding_dilations: tensor of [set_size, level, signal_length, filter_num]
        """
      
        news_embedding_dilations = self._HDC(self.DropOut(news_embedding))
        return news_embedding_dilations
    
    def _fusion(self,cdd_news_reprs,his_news_reprs):
        """ construct fusion tensor between candidate news repr and history news repr at each dilation level
        Args:
            cdd_news_reprs: tensor of [batch_size, 1, level, signal_length, filter_num]
            his_news_reprs: tensor of [batch_size, his_size, level, signal_length, filter_num]
        Returns:
            fusion_tensor: tensor of [batch_size, 320], where 320 is derived from MaxPooling with no padding
        """

        # cdd_news_reprs = torch.repeat_interleave(cdd_news_reprs,repeats=self.his_size,dim=0).view(-1,self.filter_num,self.signal_length)
        # fusion_tensor = torch.bmm(his_news_reprs.view(-1,self.filter_num,self.signal_length).permute(0,2,1),cdd_news_reprs) / math.sqrt(self.filter_num)

        fusion_tensor = torch.matmul(cdd_news_reprs, his_news_reprs.permute(0,1,2,4,3)) / math.sqrt(self.filter_num)
        # reshape the tensor in order to feed into 3D CNN pipeline
        fusion_tensor = fusion_tensor.permute(0,2,1,3,4)

        fusion_tensor = self.SeqCNN3D(fusion_tensor).view(self.batch_size,-1)
        return fusion_tensor
    
    def _click_predictor(self,fusion_tensors):
        """ calculate batch of click probabolity
        Args:
            fusion_tensors: tensor of [batch_size, cdd_size, 320]
        
        Returns:
            score: tensor of [batch_size, npratio+1], which is normalized click probabilty
        """
        score = self.predictor(fusion_tensors)
        if self.cdd_size > 1:
            score = nn.functional.log_softmax(score,dim=1)
        else:
            score = torch.sigmoid(score)
        return score
    
    # @torchsnooper.snoop()
    def forward(self,x):
        # compress batch_size and cdd_size into dim0
        #cdd_news_set = torch.cat([x['candidate_title'].long().to(self.device),x['candidate_category'].long().to(self.device),x['candidate_subcategory'].long().to(self.device)],dim=2).view(-1,self.signal_length)
        
        self.cdd_size = x['candidate_titles'].size()[1]

        title_embedded = self.embedding(x['candidate_titles'].long().to(self.device).view(-1, self.signal_length))

        cdd_news_reprs = self._news_encoder(title_embedded).view(self.batch_size,-1,self.level,self.signal_length, self.filter_num)
        
        

        assert cdd_news_reprs.shape[1] == self.cdd_size
        title_embedded = self.embedding(x['browsed_titles'].long().to(self.device).view(-1, self.signal_length))

        # compress batch_size and his_size into dim0
        # his_news_set = torch.cat([x['clicked_title'].long().to(self.device),x['clicked_category'].long().to(self.device),x['clicked_subcategory'].long().to(self.device)],dim=2).view(-1,self.signal_length)
        his_news_reprs = self._news_encoder(title_embedded).view(self.batch_size, -1, self.level, self.signal_length, self.filter_num)

        assert his_news_reprs.shape[1] == self.his_size
        
        if self.cdd_size > 1:
            fusion_vectors=[]
            for cdd_idx in range(self.cdd_size):
                fusion_vectors.append(self._fusion(cdd_news_reprs[:,cdd_idx,:,:,:].unsqueeze(dim=1),his_news_reprs).unsqueeze(dim=1))
            fusion_tensors = torch.cat(fusion_vectors, dim=1)
        
        else:
            fusion_tensors = self._fusion(cdd_news_reprs[:,0,:,:,:].unsqueeze(dim=1),his_news_reprs)
            
        score = self._click_predictor(fusion_tensors).squeeze()
        return score