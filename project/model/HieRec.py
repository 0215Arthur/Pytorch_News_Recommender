# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
import torchsnooper
import torch.nn.functional as F
from model.layers import MultiHeadSelfAttention, AdditiveAttention


class NewsEncoder(torch.nn.Module):
    def __init__(self, config):
        super(NewsEncoder, self).__init__()
        self.config = config
        #self.category_embedding =nn.Embedding(config.category_nums, config.cate_embed_size, padding_idx=0)
        #self.subcategory_embedding =nn.Embedding(config.subcategory_nums, config.cate_embed_size, padding_idx=0)
        if config.word_embedding_pretrained is not None:
            self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(
                np.load(config.data_path +config.word_embedding_pretrained)["embeddings"].astype('float32')), 
                freeze=False,padding_idx=0).to(config.device)
        else:
            self.embedding = nn.Embedding(config.n_words, config.word_embed_size, padding_idx=0)

        self.multi_head_self_attention = MultiHeadSelfAttention(config.title_heads_num, config.word_embed_size,config.dropout)
        self.additive_attention = AdditiveAttention(config.query_vector_dim, config.word_embed_size)

        self.dropout=nn.Dropout(config.dropout)

    def forward(self, data, attn_masks=None):
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
        title_ids,abst_ids =data

        abst_embeds_list=[]
        title_embeds_list=[]

        title_ids=title_ids.permute(1,0,2)
         
        for _ids in title_ids:
            title_embedded = self.word_embedding(_ids) 
            attn_output = self.multi_head_self_attention(title_embedded,title_embedded,title_embedded)
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

        # categ_embeds = self.category_embedding(news_categ)
        # subcateg_embeds = self.subcategory_embedding(news_subcateg)
        news_vector=torch.cat([title_vector,abst_vector],-1)
        news_vector=self.dropout(news_vector)


        # abst_embedded = self.word_embedding(abst_ids) 
        # attn_output = self.multi_head_self_attention(abst_embedded,abst_embedded,abst_embedded,mask=attn_masks)
        # abst_vector = self.additive_attention(attn_output)

        return news_vector

class HirUserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(HirUserEncoder, self).__init__()

        self.topic_embedding = nn.Embedding(config.batch_size,config.cate_embed_size)
        self.multi_head_self_attention = MultiHeadSelfAttention(config.user_heads_num, 
                                                     config.news_feature_size,
                                                     config.dropout)
                                                     
        self.additive_attention = AdditiveAttention(config.query_vector_dim_large, 
                                                    config.news_feature_size)
        
        


    # @torchsnooper.snoop()
    # clicked_title_input,clicked_vert_input,clicked_vert_mask_input,clicked_subvert_input,clicked_subvert_mask_input,vert_subvert_mask_input,vert_num_input,subvert_num_input
    def forward(self, news_vectors,attn_masks,categorys,subcategorys):
        '''
        config:
            news:
                {
                    'title': Tensor(batch_size) * n_words_title
                }
        Returns:
            news_vector: Tensor(batch_size, word_embedding_dim)
        news_vector (256,50,600) bs 256 history_len 50 ebs 600  
        subcategorys (256,50)
        categorys (256,50)
        TODO: 
        1.group news that has the same subcategory, and compute the sub_topic_rep
        2.group news that has the same topic, and compute the topic_rep
        3.subcategory和category有对应关系吗？怎么取
        '''
        subcategory_group = [] #save the unique subcategory_dict of every sequence in this batch
        category_group = [] #save the unique subcategory_dict of every sequence in this batch
        for subcates in subcategorys.tolist():
            usubcates=np.unique(subcates) #某一个sequence中的所有不重复subcates
            subcategory_dict = {}         #{subcate:[indexes]}
            for subcate in usubcates:
                indexes = [i for i, x in enumerate(subcates) if x==subcate]
                subcategory_dict[subcate] = indexes
            subcategory_group.append(subcategory_dict) #[{subcate1:[indexes1]},{subcate2:[indexes2]},...]
        for cates in categorys.tolist():
            ucates=np.unique(cates) #某一个sequence中的所有不重复catess
            category_dict = {}         #{cate:[indexes]}
            for cate in ucates:
                indexes = [i for i, x in enumerate(cates) if x==cate]
                category_dict[cate] = indexes
            category_group.append(category_dict) #[{cate1:[indexes1]},{cate2:[indexes2]},...]
        # subcategory_group[i] 即第i个sequence的subcate对应news的index的dict
        # subtopic-level attention
        user_subtopic_att = self.multi_head_self_attention(news_vectors,news_vectors,news_vectors,mask=attn_masks) #256,50,600
        for subcategory_dict in subcategory_group:
            user_subtopic_att = self.multi_head_self_attention(news_vectors,news_vectors,news_vectors,mask=attn_masks) #256,50,600
            user_subtopic_c = self.additive_attention(user_subtopic_att,attn_masks)#256,600    
                                    
       # subtopic embedding
        subtopic_ebd_s = self.topic_embedding(subcategorys)
        # subtopic-level aggregation
        user_subtopic_u = torch.mm(user_subtopic_c,torch.mean(subtopic_ebd_s,dim=1))
        
        return user_subtopic_u

class ClickPredictor(nn.Module):
    def __init__(self):
        super(ClickPredictor, self).__init__()

    def forward(self, news_vector, user_vector):
        predict = torch.bmm(user_vector.unsqueeze(dim=1), news_vector.unsqueeze(dim=2)).flatten()
        return predict


class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = HirUserEncoder(config)
    
        #self.encoder= Transformer_Encoder(config)
        #self.fc = nn.Linear(config.feature_size*2,  1)
        self.config=config

        self.device=config.device
        #self.Iter_Linear=nn.Sequential(*iter_dense)
    #@torchsnooper.snoop()
    ##################################################################
    # title_inputs,vert_inputs,subvert_inputs,                       #
    # clicked_title_input,clicked_vert_input,clicked_vert_mask_input,#
    # clicked_subvert_input,clicked_subvert_mask_input,              #
    # vert_subvert_mask_input,vert_num_input,subvert_num_input,      #
    # rw_vert_input,rw_subvert_input                                 #
    ##################################################################
    def forward(self, batch):
        _input=(batch['browsed_titles'].to(self.device), 
                batch['browsed_absts'].to(self.device), 
                # batch['browsed_categ_ids'].to(self.device), 
                # batch['browsed_subcateg_ids'].to(self.device), 
                #batch['browsed_entity_ids'].to(self.device)
                )
        candidate_input=(batch['candidate_titles'].to(self.device), 
                         batch['candidate_absts'].to(self.device), 
                        #  batch['candidate_categ_ids'].to(self.device),
                        #  batch['candidate_subcateg_ids'].to(self.device),
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
         
        sample_masks=batch['candidate_mask'].to(self.config.device)

        # b*E
        user_vector = self.user_encoder(browsed_vector,
                                        batch['browsed_mask'].to(self.config.device),
                                        batch['browsed_categ_ids'].to(self.config.device),
                                        batch['browsed_subcateg_ids'].to(self.config.device)).unsqueeze(1)
         
        pred =  torch.sum(user_vector*candidate_vector,2)
        if batch['candidate_mask'] is not None:
            pred = pred.masked_fill(sample_masks == 0, -1e9)

        return pred
    
    def update_rep(self, news_iter, batch_size=1280):
        """
        更新得到新闻标注
        """
        self.news_embeds = torch.Tensor(self.config.news_nums + 1, self.config.news_feature_size).cuda()

        for i,batch in enumerate(news_iter):
            if len(batch['titles']) == batch_size:
                news_vector = self.news_encoder((batch['titles'].to(self.device).view(256, batch_size // 256, -1 ),\
                                                batch['absts'].to(self.device).view(256, batch_size // 256, -1 ))).view(-1, self.config.news_feature_size)
                self.news_embeds[i*batch_size + 1: (i+1)*batch_size + 1].copy_(news_vector)
            else:
                sz = len(batch['titles'])
                news_vector = self.news_encoder((batch['titles'].to(self.device).view(sz, 1, -1),\
                                                batch['absts'].to(self.device).view(sz, 1, -1 ))).view(-1, self.config.news_feature_size)
                self.news_embeds[i*batch_size + 1:].copy_(news_vector)
        return 

    
    def predict(self, batch):
        """
        快速进行评估测试
        """
        browsed_vector = self.news_embeds[batch['browsed_ids'].to(self.config.device)]
        user_vector = self.user_encoder(browsed_vector,batch['browsed_mask'].to(self.config.device)).unsqueeze(1)
        candidate_vector =self.news_embeds[batch['candidate_ids'].to(self.config.device)]
        # print(user_vector.size())
        # print(candidate_vector.size())
        pred =  torch.sum(user_vector*candidate_vector,2)
        if batch['candidate_mask'] is not None:
            pred = pred.masked_fill(batch['candidate_mask'].to(self.config.device) == 0, -1e9)

        return pred