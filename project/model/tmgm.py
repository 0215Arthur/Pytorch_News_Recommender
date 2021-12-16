# coding: UTF-8
"""
Time-aware Multi-Granularity User Preference Modeling, TMGM
"""
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
        self.pretrained = False
        if config.dataset == "GLOBO":
            self.pretrained = True
            try:
                self.article_embedding = nn.Embedding.from_pretrained(torch.tensor(
                    np.load(config.data_path + config.article_embedding_pretrained)["embeddings"].astype('float32')), 
                    freeze=True, padding_idx=0).to(config.device)
                print(self.article_embedding.weight.size())
            except:
                print("load article embedding error.")
        else:
            if config.word_embedding_pretrained is not None:
                self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(
                    np.load(config.data_path +config.word_embedding_pretrained)["embeddings"].astype('float32')), 
                    freeze=False,padding_idx=0).to(config.device)
            else:
                self.embedding = nn.Embedding(config.n_words, config.word_embed_size, padding_idx=0)
            # self.entity_embedding = nn.Embedding.from_pretrained(torch.tensor(
            #     np.load( config.data_path +config.entity_embedding_pretrained)["embeddings"].astype('float32')), 
            #      freeze=False).to(config.device)
            self.category_embedding = nn.Embedding(config.category_nums, config.cate_embed_size, padding_idx=0)
            self.subcategory_embedding = nn.Embedding(config.subcategory_nums, config.cate_embed_size, padding_idx=0)
            self.multi_head_self_attention = MultiHeadSelfAttention(config.title_heads_num, config.word_embed_size,config.dropout)
            self.additive_attention = AdditiveAttention(config.query_vector_dim, config.word_embed_size)
        self.dropout=nn.Dropout(config.dropout)
        # conv_blocks = []
        # for filter_size in config.kernel_sizes_2:
        #     maxpool_kernel_size = config.entity_nums - filter_size + 1
        #     conv = nn.Conv1d(in_channels=config.cate_embed_size, out_channels=config.filter_nums, kernel_size=filter_size)
             
        #     component = nn.Sequential(
        #         conv,
        #       #  nn.ReLU(),
        #         nn.MaxPool1d(kernel_size=maxpool_kernel_size))
        #     conv_blocks.append(component)
        
        # self.conv_blocks = nn.ModuleList(conv_blocks)

        # news_dense = [
        #     nn.Linear(config.final_embed_size, config.feature_size),
        #     GELU()#,
        #     #nn.Linear(config.feature_size, config.feature_size)
        #     #nn.ReLU( )
        # ]
        # # self.news_dense = nn.Sequential(*news_dense)
        # #self.Linear= nn.Linear(config.final_embed_size, config.final_embed_size)
        # self.Linear=nn.Sequential(*news_dense)
    

    def initialize(self):
        if self.config.dataset != "GLOBO":
            self.multi_head_self_attention.initialize()
            self.additive_attention.initialize()
            nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.category_embedding.weight[0])
            nn.init.uniform_(self.subcategory_embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.subcategory_embedding.weight[0])


    # @torchsnooper.snoop()
    def forward(self, data, categ, attn_masks=None):
        '''
        config:
            news:
                {
                    'title': Tensor(batch_size) * n_words_title
                }
        Returns:
            news_vector: Tensor(batch_size, word_embedding_dim)
        '''
        if self.pretrained:
            return self.article_embedding(data)
        else:   
            news_categ, news_subcateg = categ
            if isinstance(data, tuple):
                title_ids,abst_ids = data
            else:
                title_ids = data
                abst_ids = None
            abst_embeds_list=[]
            title_embeds_list=[]

            title_ids=title_ids.permute(1,0,2)
            
            for _ids in title_ids:
                title_embedded = self.word_embedding(_ids) 
                attn_output = self.multi_head_self_attention(title_embedded,title_embedded,title_embedded)
                title_vector = self.additive_attention(attn_output)
                title_embeds_list.append(title_vector)
            
            title_vector=torch.stack(title_embeds_list).permute(1,0,2)
            category_vector = self.category_embedding(news_categ)
            subcategory_vector = self.subcategory_embedding(news_subcateg)
            if abst_ids is None:
                news_vector = torch.cat([category_vector, subcategory_vector, title_vector],dim=-1)
                return self.dropout(news_vector)

            abst_ids=abst_ids.permute(1,0,2)
            
            for _ids in abst_ids:
                abst_embedded = self.word_embedding(_ids) 
                attn_output = self.multi_head_self_attention(abst_embedded,abst_embedded,abst_embedded)
                abst_vector = self.additive_attention(attn_output)
                abst_embeds_list.append(abst_vector)
            
            abst_vector=torch.stack(abst_embeds_list).permute(1,0,2)


            news_vector=torch.cat([category_vector, subcategory_vector,title_vector,abst_vector],-1)
            news_vector=self.dropout(news_vector)

        return news_vector


class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        # self.encoder = nn.Sequential(
        #             MultiHeadSelfAttention(config.user_heads_num, 
        #                                              config.news_feature_size,
        #                                              config.dropout),
        #             AdditiveAttention(config.query_vector_dim_large, 
        #                                             config.news_feature_size)
        #             )
        # self.encoder_2 = nn.Sequential(
        #             MultiHeadSelfAttention(config.user_heads_num, 
        #                                              config.news_feature_size,
        #                                              config.dropout),
        #             AdditiveAttention(config.query_vector_dim_large, 
        #                                             config.news_feature_size)
        #             )
        self.multi_head_self_attention = MultiHeadSelfAttention(config.user_heads_num, 
                                                     config.news_feature_size,
                                                     config.dropout)
                                                     
        self.additive_attention = AdditiveAttention(config.query_vector_dim_large, 
                                                    config.news_feature_size)
    
    def initialize(self):
        # for module in self.encoder:
        #     module.initialize()

        # for module in self.encoder_2:
        #     module.initialize()        

        self.multi_head_self_attention.initialize()
        self.additive_attention.initialize()


    # @torchsnooper.snoop()
    def forward(self, news_vectors,attn_masks,time_mat):
        k = time_mat.size()[1]
        vectors = []
        #print(time_mat.size())
        for i in range(k):
            
            mat = time_mat[:,i,:,:]
            # print(mat.size())
            attn_output = self.multi_head_self_attention(news_vectors,news_vectors,news_vectors,mask=attn_masks, time_mat=mat)
            user_vector = self.additive_attention(attn_output, attn_masks)
            vectors.append(user_vector)
        # attn_output = self.multi_head_self_attention(news_vectors,news_vectors,news_vectors,mask=attn_masks, time_mat=time_mat2)
        # user_vector = self.additive_attention(attn_output, attn_masks)
        # self.encoder(news_vectors)
        user_vector = None
        for vector in vectors:
            if user_vector is None:
                user_vector = vector
                continue
            user_vector += vector

        return user_vector


class Model(torch.nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = UserEncoder(config)
        self.news_encoder.initialize()
        self.user_encoder.initialize()
        #self.encoder= Transformer_Encoder(config)
        #self.fc = nn.Linear(config.feature_size*2,  1)
        self.config=config
        # self.norm = nn.LayerNorm(config.feature_size*2)
        # iter_dense = [
        #     nn.Linear(config.feature_size*2, config.feature_size),
        #     GELU(),
        #     nn.Linear(config.feature_size, 1)
        #     #nn.ReLU( )
        # ]
        self.device=config.device
        #self.Iter_Linear=nn.Sequential(*iter_dense)
    
    # @torchsnooper.snoop()
    def forward(self, batch):
        if self.config.dataset == "GLOBO":
            candidate_vector =self.news_encoder(batch['candidate_ids'].to(self.device), None)
            browsed_vector =self.news_encoder(batch['browsed_ids'].to(self.device), None)
        elif self.config.dataset == "MIND":
            _input=(batch['browsed_titles'].to(self.device), 
                    batch['browsed_absts'].to(self.device), 
                    #batch['browsed_entity_ids'].to(self.device)
                    )
            browsed_categ = ( batch['browsed_categ_ids'].to(self.device), 
                            batch['browsed_subcateg_ids'].to(self.device), 
                    )
            candidate_input=(batch['candidate_titles'].to(self.device), 
                            batch['candidate_absts'].to(self.device), 

                            #batch['candidate_entity_ids'].to(self.device)
                            )
            cand_categ = (batch['candidate_categ_ids'].to(self.device),
                            batch['candidate_subcateg_ids'].to(self.device)
                    )
            candidate_vector =self.news_encoder(candidate_input, cand_categ)
        
            browsed_vector =self.news_encoder(_input, browsed_categ)
        elif self.config.dataset == "ADR":
            _input=batch['browsed_titles'].to(self.device)
            browsed_categ = ( batch['browsed_categ_ids'].to(self.device), 
                            batch['browsed_subcateg_ids'].to(self.device), 
                    )
            cand_categ = (batch['candidate_categ_ids'].to(self.device),
                            batch['candidate_subcateg_ids'].to(self.device)
                    )
            candidate_input=batch['candidate_titles'].to(self.device)
            candidate_vector =self.news_encoder(candidate_input, cand_categ)
            browsed_vector =self.news_encoder(_input, browsed_categ)
         
       # browsed_abst_ids=batch['browsed_absts'].to(self.config.device).permute(1,0,2)
        #browsed_vector = torch.stack([self.news_encoder(x) for x in  browsed_title_ids ])
        #browsed_vector=browsed_vector.permute(1,0,2)
        sample_masks=batch['candidate_mask'].to(self.config.device)

        # b*E
        user_vector = self.user_encoder(
                                        browsed_vector,
                                        batch['browsed_mask'].to(self.config.device),
                                        batch['time_mat'].to(self.config.device)
                                        ).unsqueeze(1)
        # ui_vectors=user_vector*candidate_vector
        # ui_vectors=torch.cat([ui_vectors,candidate_vector],2)
        # attn_output=self.encoder(ui_vectors,sample_masks)
        # pred=self.fc( attn_output).squeeze(-1)
        # sample_size=sample_masks.size(1)
        # user_vectors=user_vector.repeat(1,sample_size,1)
        # ui_vectors=torch.cat([user_vectors,candidate_vector],2)
        # ui_vectors=self.norm(ui_vectors)
        # pred=self.Iter_Linear(ui_vectors).squeeze(-1)
         
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
                if "absts" in batch:
                    news_vector = self.news_encoder((
                                                        batch['titles'].to(self.device).view(256, batch_size // 256, -1 ),\
                                                        batch['absts'].to(self.device).view(256, batch_size // 256, -1 )
                                                    ),
                                                    (
                                                        batch['categ_ids'].to(self.device).view(256, batch_size // 256, -1 ),
                                                        batch['subcateg_ids'].to(self.device).view(256, batch_size // 256, -1 )
                                                    )
                                                    ).view(-1, self.config.news_feature_size)
                else:
                    news_vector = self.news_encoder(
                                                    batch['titles'].to(self.device).view(256, batch_size // 256, -1 ),
                                                    (
                                                        batch['categ_ids'].to(self.device).view(256, batch_size // 256, -1 ),
                                                        batch['subcateg_ids'].to(self.device).view(256, batch_size // 256, -1 )
                                                    )
                                                    ).view(-1, self.config.news_feature_size)
                self.news_embeds[i*batch_size + 1: (i+1)*batch_size + 1].copy_(news_vector)
            else:
                sz = len(batch['titles'])
                if "absts" in batch:
                    news_vector = self.news_encoder((
                                                        batch['titles'].to(self.device).view(sz, 1, -1),\
                                                        batch['absts'].to(self.device).view(sz, 1, -1 )
                                                    ),
                                                    (
                                                        batch['categ_ids'].to(self.device).view(sz, 1, -1),
                                                        batch['subcateg_ids'].to(self.device).view(sz, 1, -1)
                                                    )
                                                    ).view(-1, self.config.news_feature_size)
                else:
                    news_vector = self.news_encoder(
                                                        batch['titles'].to(self.device).view(sz, 1, -1),
                                                    (
                                                        batch['categ_ids'].to(self.device).view(sz, 1, -1),
                                                        batch['subcateg_ids'].to(self.device).view(sz, 1, -1)
                                                    )
                                                    ).view(-1, self.config.news_feature_size)                    
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