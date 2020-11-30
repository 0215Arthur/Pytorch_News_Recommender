# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
import torchsnooper
import torch.nn.functional as F

import math

from tools import log_exec_time

 
 

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
        # change: 
        self.query_vector = nn.Parameter(torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, input,mask=None):
        '''
        config:
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


 
class NewsEncoder(torch.nn.Module):
    def __init__(self, config):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.category_embedding =nn.Embedding(config.category_nums, config.cate_embed_size, padding_idx=0)
        self.subcategory_embedding =nn.Embedding(config.subcategory_nums, config.cate_embed_size, padding_idx=0)
        self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(
            np.load( config.data_path +config.word_embedding_pretrained)["embeddings"].astype('float32')), 
             freeze=False,padding_idx=0).to(config.device)
        # self.entity_embedding = nn.Embedding.from_pretrained(torch.tensor(
        #     np.load( config.data_path +config.entity_embedding_pretrained)["embeddings"].astype('float32')), 
        #      freeze=False).to(config.device)

        self.multi_head_self_attention = MultiHeadSelfAttention(config.title_heads_num, config.word_embed_size,config.dropout)
        self.additive_attention = AdditiveAttention(config.query_vector_dim, config.word_embed_size)
       
        self.dropout=nn.Dropout(config.dropout)
 
    #@torchsnooper.snoop()
    def forward(self, data,attn_masks=None):
        '''
        config:
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
        title_ids,abst_ids,news_categ,news_subcateg =data

        abst_embeds_list=[]
        title_embeds_list=[]
        # news_entity_ids=news_entity_ids.permute(1,0,2)
        # for _ids in news_entity_ids:
        #     #print(_ids,type(_ids))
        #     entity_embeds=self.entity_embedding(_ids)
        #     entity_embeds = entity_embeds.permute(0, 2, 1)
        #     out=[conv(entity_embeds) for conv in self.conv_blocks]
        #     out = torch.cat(out, dim=1) # Batch * filter_size*1
        #     entity_embeds = out.view(-1, out.size(1)) # Batch * filter_size
        #     entity_embeds_list.append(entity_embeds)
        # entity_vector=torch.stack(entity_embeds_list).permute(1,0,2)
        #print(title_ids.size())

        title_ids=title_ids.permute(1,0,2)
         
        for _ids in title_ids:
            title_embedded = self.word_embedding(_ids) 
            attn_output = self.multi_head_self_attention(title_embedded,title_embedded,title_embedded)
           # attn_output=self.dropout(attn_output)
            title_vector = self.additive_attention(attn_output)
            title_embeds_list.append(title_vector)
         
        title_vector=torch.stack(title_embeds_list).permute(1,0,2)

        abst_ids=abst_ids.permute(1,0,2)
         
        for _ids in abst_ids:
            abst_embedded = self.word_embedding(_ids) 
            attn_output = self.multi_head_self_attention(abst_embedded,abst_embedded,abst_embedded)
            abst_vector = self.additive_attention(attn_output)
            abst_embeds_list.append(abst_vector)
         
        abst_vector=torch.stack(abst_embeds_list).permute(1,0,2)
        categ_embeds = self.category_embedding(news_categ)
        subcateg_embeds = self.subcategory_embedding(news_subcateg)
        news_vector=torch.cat([title_vector,abst_vector,categ_embeds,subcateg_embeds],-1)
        news_vector=self.dropout(news_vector)

        return news_vector

class UserEncoder(torch.nn.Module):

    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.multi_head_self_attention = MultiHeadSelfAttention(config.user_heads_num, 
                                                     config.news_feature_size,
                                                     config.dropout)
        self.additive_attention = AdditiveAttention(config.query_vector_dim_large, 
                                                    config.news_feature_size)

    def forward(self, news_vectors,attn_masks=None):
        attn_output = self.multi_head_self_attention(news_vectors,news_vectors,news_vectors,mask=attn_masks)
        user_vector = self.additive_attention(attn_output,attn_masks)
        return user_vector



class Model(torch.nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = UserEncoder(config)
    
        #self.encoder= Transformer_Encoder(config)
        #self.fc = nn.Linear(config.feature_size*2,  1)
        self.config=config
        self.norm = nn.LayerNorm(config.news_feature_size)
        # iter_dense = [
        #     nn.Linear(config.feature_size*2, config.feature_size),
        #     GELU(),
        #     nn.Linear(config.feature_size, 1)
        #     #nn.ReLU( )
        # ]
        self.device=config.device
        #self.Iter_Linear=nn.Sequential(*iter_dense)
    #@torchsnooper.snoop()
    def forward(self, batch):
        _input=(batch['browsed_titles'].to(self.device), 
                batch['browsed_absts'].to(self.device),  
                batch['browsed_categ_ids'].to(self.device), 
                batch['browsed_subcateg_ids'].to(self.device), 
                #batch['browsed_entity_ids'].to(self.device)
                )
        candidate_input=(batch['candidate_titles'].to(self.device), 
                         batch['candidate_absts'].to(self.device), 
                         batch['candidate_categ_ids'].to(self.device),
                         batch['candidate_subcateg_ids'].to(self.device),
                         #batch['candidate_entity_ids'].to(self.device)
                         )


        # b*sample_size*E
       # candidate_title_ids=batch['candidate_titles'].to(self.config.device).permute(1,0,2)
        #candidate_abst_ids=batch['candidate_absts'].to(self.config.device).permute(1,0,2)
        candidate_vector =self.news_encoder(candidate_input)# torch.stack([self.news_encoder(x) for x in  candidate_title_ids  ])
        #candidate_vector = torch.stack([self.news_encoder(x) for x in zip(candidate_title_ids,candidate_abst_ids) ])
        #candidate_vector=candidate_vector.permute(1,0,2)
        # b*history_len*E
        browsed_vector =self.news_encoder(_input)
         
       # browsed_abst_ids=batch['browsed_absts'].to(self.config.device).permute(1,0,2)
        #browsed_vector = torch.stack([self.news_encoder(x) for x in  browsed_title_ids ])
        #browsed_vector=browsed_vector.permute(1,0,2)
        sample_masks=batch['candidate_mask'].to(self.config.device)

        #browsed_mask=batch['browsed_mask'].to(self.config.device)
        browsed_vector=self.norm(browsed_vector)

        # b*E ,
        user_vector = self.user_encoder(browsed_vector,attn_masks=None).unsqueeze(1)
 


        pred =  torch.sum(user_vector*candidate_vector,2)
        if batch['candidate_mask'] is not None:
            pred = pred.masked_fill(sample_masks == 0, -1e9)

        return pred