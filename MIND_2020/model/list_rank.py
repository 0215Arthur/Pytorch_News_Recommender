# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
import torchsnooper
import torch.nn.functional as F

import math
from config import Config

 
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, head_nums,mask=None, dropout=None):
        # 
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        
        # 设置掩码
       # print(query.size(),scores.size())

        if mask is not None:
            #print(scores[0])
            _mask=(mask.unsqueeze(1)*mask.unsqueeze(2)).unsqueeze(1)
            #print(_mask.size())
            _mask=torch.cat([_mask for _ in range(head_nums)],1)
            scores = scores.masked_fill(_mask == 0, -1e9)
            #print(scores[0])

        p_attn = F.softmax(scores, dim=-1)
        #print(p_attn)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadSelfAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)
  #  @torchsnooper.snoop()
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, attn = self.attention(query, key, value, head_nums=self.h, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class AdditiveAttention(torch.nn.Module):
    def __init__(self, query_vector_dim, input_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(input_vector_dim, query_vector_dim)
        self.query_vector = nn.Parameter(torch.empty(query_vector_dim).uniform_(-1, 1))

    def forward(self, input,mask=None):
        '''
        Args:
            input: batch_size, n_input_vector, input_vector_dim
        Returns:
            result: batch_size, input_vector_dim
        '''
        # batch_size, n_input_vector, query_vector_dim -> 512*25*800= > 512*25*512
        tmp = torch.tanh(self.linear(input))
        scores=torch.matmul(tmp, self.query_vector)
       # print('Add_Attn: scores',scores.size())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # batch_size, n_input_vector
        weight = F.softmax(scores, dim=1)
        result = torch.bmm(weight.unsqueeze(dim=1), input).squeeze(dim=1)
        return result


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):

        residual = x
        x = self.w_2(self.activation(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x


class UserEncoder(torch.nn.Module):

    def __init__(self, config):
        super(UserEncoder, self).__init__()
        #self.multihead_attn = nn.MultiheadAttention(config.embed_size, config.num_heads)
        self.multihead_attn= MultiHeadSelfAttention(config.user_heads_num,config.title_size,config.dropout)
        self.ffn=PositionwiseFeedForward(config.title_size,config.title_size)

        self.additive_attention = AdditiveAttention(config.query_vector_dim_large, config.title_size)
        self.config=config
     
    #@torchsnooper.snoop()
    def forward(self, user_feats,attn_masks):

        #user_feats=torch.transpose(user_feats, 0, 1)  # 调整特征维度，以符合multihead_attn接口
        # 输出 attn_output (Length*Batch*Embeds), attn_output_weights   
        attn_output = self.multihead_attn(user_feats ,
                                        user_feats,user_feats,mask=attn_masks)
        # print(user_feats)
        # print('User_Encoder MultiheadSA output:',attn_output.size(),attn_output)
        attn_output=self.ffn(attn_output)

        title_vector = self.additive_attention(attn_output,mask=attn_masks) # Batch_size * embed_size

        return title_vector



class NewsEncoder(torch.nn.Module):
    def __init__(self, config):
        super(NewsEncoder, self).__init__()
        
        # title
        
        self.category_embedding =nn.Embedding(config.category_nums, config.cate_embed_size, padding_idx=0)
        self.subcategory_embedding =nn.Embedding(config.subcategory_nums, config.cate_embed_size, padding_idx=0)
        self.news_embedding = nn.Embedding.from_pretrained(torch.tensor(
            np.load( config.data_path +config.bert_embedding_pretrained)["embeddings"].astype('float32')), 
             freeze=True).to(config.device)
       
        news_dense = [
            nn.Linear(config.feature_size, config.title_size),#,
            GELU()#,#nn.GeLU(inplace=False)
        ]
        self.news_dense = nn.Sequential(*news_dense)
        self.dropout = nn.Dropout(p=config.dropout)

        
  #  @torchsnooper.snoop()
    def forward(self, _input):
        '''
        Args:
            news:
                {
                    'title': Tensor(batch_size) * n_words_title
                }
        Returns:
            news_vector: Tensor(batch_size, word_embedding_dim)
        '''

        # Title
        # torch.stack(news['title'], dim=1) size: (batch_size, n_words_title)
        # batch_size, n_words_title, word_embedding_dim
        news_ids,news_categ,news_subcateg=_input
        categ_embeds = self.category_embedding(news_categ)
        subcateg_embeds = self.subcategory_embedding(news_subcateg)
        news_embeds = self.news_embedding(news_ids) 
        #print('news_embeds',news_embeds)
        
        news_vector=torch.cat([news_embeds,categ_embeds,subcateg_embeds],-1)
        news_vector = self.news_dense(news_vector) 
        news_vector=self.dropout(news_vector)

        # news_embeds = self.news_embedding(news_ids) 
        # news_vector = self.news_dense(news_embeds) 
        # Batch_size * embed_size
        #print("news_vector:",news_vector.size())
        return news_vector

class Transformer_Encoder(nn.Module):
    def __init__(self, config):
        super(Transformer_Encoder, self).__init__()
        self.multihead_attn= MultiHeadSelfAttention(config.list_num_heads,config.title_size,config.dropout)
        self.ffn=PositionwiseFeedForward(config.title_size,config.title_size)

    def forward(self,ui_vectors,sample_masks=None):
        #ui_vectors,sample_masks=x
        attn_output = self.multihead_attn(ui_vectors, ui_vectors,ui_vectors,mask=sample_masks) 
        attn_output = self.ffn(attn_output)
        return attn_output

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.user_encoder = UserEncoder(config)
        self.news_encoder= NewsEncoder(config)
   
        self.encoder= Transformer_Encoder(config)
        
        self.title_size=config.title_size
        self.device=config.device


        news_dense = [
            nn.Linear(config.title_size*2, config.title_size),
            GELU()#,
            #nn.Linear(config.feature_size, config.feature_size)
            #nn.ReLU( )
        ]
        self.Iter_Linear=nn.Sequential(*news_dense)
    
        #self.fc = nn.Linear(config.feature_size, config.feature_size)
        #self.fc_1 = nn.Linear(config.title_size*2,  1)
        self.fc = nn.Linear(config.title_size,  1)
        self.norm = nn.LayerNorm(config.title_size*2)
        
        
    
    #@torchsnooper.snoop()
    def forward(self,x ):
         
        _input=(x['browsed_ids'].to(self.device),x['browsed_categ_ids'].to(self.device),x['browsed_subcateg_ids'].to(self.device))#history_categ,history_subcateg)
        
        user_feats = self.news_encoder(_input) # [batch_size, seq_len, embeding]=[128, 32, 300]
        #user_feats=user_feats.permute(1,0,2)
        #attn_masks=x['browsed_mask'].to(self.device)
        user_vector=self.user_encoder(user_feats,attn_masks=None)

        # user_vector: Batch_size * feature_size
        user_vector = user_vector.unsqueeze(1)

        candidate_input=(x['candidate_ids'].to(self.device),x['candidate_categ_ids'].to(self.device),x['candidate_subcateg_ids'].to(self.device))
        # news_vector: Batch_size * sample_size * feature_size
        news_vectors = self.news_encoder(candidate_input)
        #print(news_vectors)
        #print(user_vector)
        # print(x['candidate_mask'].size())
        # print(x['candidate_mask'])
        sample_masks=x['candidate_mask'].to(self.device)
        sample_size=sample_masks.size(1)
        user_vectors=user_vector.repeat(1,sample_size,1)
        #ui_vectors=user_vector*news_vectors # B * S * F  512*300*512

        #user_vectors=user_vector.repeat(1,sample_masks.size(1),1)
        ui_vectors=torch.cat([user_vectors,news_vectors],2)
        ui_vectors=self.norm(ui_vectors)
        ui_vectors=self.Iter_Linear(ui_vectors)
        
        if sample_masks is not None:
            _sample_masks=sample_masks.unsqueeze(2).repeat(1,1,self.title_size)
            #print(_sample_masks)

            ui_vectors=ui_vectors.masked_fill(_sample_masks == 0, 0)
        
        attn_output=self.encoder(ui_vectors,sample_masks)
       # attn_output=self.encoder_1(attn_output,sample_masks)
         
        pred=self.fc(attn_output).squeeze(-1)
 
        if sample_masks is not None:
            pred = pred.masked_fill(sample_masks == 0, -1e9)
            #pred_add= pred.masked_fill(sample_masks == 0, -1e9)


        return pred#,pred_add




        
class A(nn.Module):
    def __init__(self, config):
        super(A, self).__init__()
         
       

        self.user_encoder = UserEncoder(config)
        self.news_encoder= NewsEncoder(config)
        # self.multihead_attn= MultiHeadSelfAttention(config.num_heads_2,config.feature_size*2,config.dropout)
        # self.ffn=PositionwiseFeedForward(config.feature_size*2,config.feature_size*2)
        
        self.encoder= Transformer_Encoder(config)
        # self.encoder_1= Transformer_Encoder(config)
        # self.encoder_2= Transformer_Encoder(config)

        
        self.feature_size=config.feature_size
        self.device=config.device


        news_dense = [
            nn.Linear(config.final_embed_size, config.feature_size),
            GELU()#,
            #nn.Linear(config.feature_size, config.feature_size)
            #nn.ReLU( )
        ]
        
        iter_dense = [
            nn.Linear(config.feature_size*2, config.feature_size),
            GELU(),
           # nn.Linear(config.feature_size, config.feature_size)
            #nn.ReLU( )
        ]
        self.Iter_Linear=nn.Sequential(*iter_dense)

        # self.news_dense = nn.Sequential(*news_dense)
        #self.Linear= nn.Linear(config.final_embed_size, config.final_embed_size)
        self.Linear=nn.Sequential(*news_dense)
        #self.fc = nn.Linear(config.feature_size, config.feature_size)
        self.fc2 = nn.Linear(config.feature_size*2,  1)
        self.fc_add = nn.Linear(config.feature_size,  1)
        self.norm = nn.LayerNorm(config.feature_size*2)
        
        
    
    #@torchsnooper.snoop()
    def forward(self,x ):
         
        _input=(x['browsed_ids'].to(self.device),x['browsed_categ_ids'].to(self.device),x['browsed_subcateg_ids'].to(self.device))#history_categ,history_subcateg)
        user_feats = self.news_encoder(_input) # [batch_size, seq_len, embeding]=[128, 32, 300]
        #user_feats=user_feats.permute(1,0,2)
        attn_masks=x['browsed_mask'].to(self.device)
        user_vector=self.user_encoder(user_feats,attn_masks)
        # user_vector: Batch_size * feature_size
        user_vector = user_vector.unsqueeze(1)

        candidate_input=(x['candidate_ids'].to(self.device),x['candidate_categ_ids'].to(self.device),x['candidate_subcateg_ids'].to(self.device))
        # news_vector: Batch_size * sample_size * feature_size
        news_vectors = self.Linear(self.news_encoder(candidate_input))   
        #print(news_vectors)
        #print(user_vector)
        # print(x['candidate_mask'].size())
        # print(x['candidate_mask'])
        sample_masks=x['candidate_mask'].to(self.device)
        sample_size=sample_masks.size(1)
        user_vectors=user_vector.repeat(1,sample_size,1)
        #ui_vectors=user_vector*news_vectors # B * S * F  512*300*512

        #user_vectors=user_vector.repeat(1,sample_masks.size(1),1)
        ui_vectors=torch.cat([user_vectors,news_vectors],2)
        ui_vectors=self.norm(ui_vectors)
        ui_vectors=self.Iter_Linear(ui_vectors)
        pred_add=self.fc_add(ui_vectors)
        
        if sample_masks is not None:
            _sample_masks=sample_masks.unsqueeze(2).repeat(1,1,self.feature_size)

            ui_vectors=ui_vectors.masked_fill(_sample_masks == 0, 0)
        ui_vectors=torch.cat([ui_vectors,news_vectors],2)
        
        # attn_output = self.multihead_attn(ui_vectors, ui_vectors,ui_vectors,mask=sample_masks)
        # # print(attn_output.size())
        # # print(attn_output)
        # self.attn_output=attn_output
        # attn_output = self.ffn(attn_output)
         #ui_vectors,sample_mask
        attn_output=self.encoder(ui_vectors,sample_masks)
       # attn_output=self.encoder_1(attn_output,sample_masks)
         
        pred=self.fc2( attn_output).squeeze(-1)
 
          # mask操作
        if sample_masks is not None:
            pred = pred.masked_fill(sample_masks == 0, -1e9)
            pred_add= pred.masked_fill(sample_masks == 0, -1e9)
  

        return pred,pred_add
