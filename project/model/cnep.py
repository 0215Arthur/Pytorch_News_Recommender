"""
基于CNE新闻编码器改造的模型
"""
"""
Neural News Recommendation with Collaborative News Encoding and Structural User Encoding
(EMNLP-2021 Finding).
https://arxiv.org/pdf/2109.00750.pdf
https://github.com/Veason-silverbullet/NNR/blob/main/newsEncoders.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchsnooper
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from model.layers import CNEAttention, ScaledDotProduct_CandidateAttention, MultiHeadSelfAttention, AdditiveAttention



class baseEncoder(nn.Module):
    def __init__(self, config):
        super(baseEncoder, self).__init__()
        self.word_embedding_dim = config.word_embedding_dim
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        with open('word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_abstract_length) + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        self.category_embedding = nn.Embedding(num_embeddings=config.category_num, embedding_dim=config.category_embedding_dim)
        self.subCategory_embedding = nn.Embedding(num_embeddings=config.subCategory_num, embedding_dim=config.subCategory_embedding_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.auxiliary_loss = None

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subCategory_embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.subCategory_embedding.weight[0])

    # Input
    # title_text          : [batch_size, news_num, max_title_length]
    # title_mask          : [batch_size, news_num, max_title_length]
    # title_entity        : [batch_size, news_num, max_title_length]
    # content_text        : [batch_size, news_num, max_content_length]
    # content_mask        : [batch_size, news_num, max_content_length]
    # content_entity      : [batch_size, news_num, max_content_length]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # user_embedding      : [batch_size, user_embedding_dim]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]
    def forward(self, title_text, title_mask, title_entity, content_text, content_mask, content_entity, category, subCategory, user_embedding):
        raise Exception('Function forward must be implemented at sub-class')

    # Input
    # news_representation : [batch_size, news_num, unfused_news_embedding_dim]
    # category            : [batch_size, news_num]
    # subCategory         : [batch_size, news_num]
    # Output
    # news_representation : [batch_size, news_num, news_embedding_dim]



class NewsEncoder(nn.Module):
    def __init__(self, config):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.pretrained = False
        self.dropout = nn.Dropout(p=config.dropout, inplace=True)
        if config.dataset == "GLOBO":
            self.pretrained = True
            try:
                self.article_embedding = nn.Embedding.from_pretrained(torch.tensor(
                    np.load(config.data_path + config.article_embedding_pretrained)["embeddings"].astype('float32')), 
                    freeze=True,padding_idx=0).to(config.device)
                print(self.article_embedding.weight.size())
            except:
                print("load article embedding error.")
        else:
            self.word_embedding = [nn.Embedding.from_pretrained(torch.tensor(
                np.load( config.data_path +config.word_embedding_pretrained)["embeddings"].astype('float32')), 
                freeze=True,padding_idx=0).to(config.device),
                nn.Dropout(p=config.dropout, inplace=False)]
            
            self.category_embedding =nn.Embedding(config.category_nums, config.cate_embed_size, padding_idx=0)
            self.subcategory_embedding =nn.Embedding(config.subcategory_nums, config.cate_embed_size, padding_idx=0)
            
            self.word_embedding = nn.Sequential(*self.word_embedding)

            self.max_title_length = config.n_words_title
            self.max_content_length = config.n_words_abst
            self.word_embedding_dim = config.word_embed_size
            
            # title / content encoding
            self.title_base_attention = MultiHeadSelfAttention(config.title_heads_num, config.word_embed_size,config.dropout)
            self.content_base_attention = MultiHeadSelfAttention(config.title_heads_num, config.word_embed_size,config.dropout)
            self.title_add_attention = CNEAttention(self.word_embedding_dim, config.attention_dim)
            self.content_add_attention = CNEAttention(self.word_embedding_dim, config.attention_dim)

            self.title_H = nn.Linear(in_features=self.word_embedding_dim, out_features=self.word_embedding_dim, bias=False)
            self.title_M = nn.Linear(in_features=self.word_embedding_dim, out_features=self.word_embedding_dim, bias=True)
            self.content_H = nn.Linear(in_features=self.word_embedding_dim, out_features=self.word_embedding_dim, bias=False)
            self.content_M = nn.Linear(in_features=self.word_embedding_dim, out_features=self.word_embedding_dim, bias=True)
            
            # self-attention
            self.title_self_attention = CNEAttention(self.word_embedding_dim, config.attention_dim)
            self.content_self_attention = CNEAttention(self.word_embedding_dim, config.attention_dim)
            # cross-attention

            
            self.cate_cross_attention = ScaledDotProduct_CandidateAttention(self.word_embedding_dim, config.cate_embed_size, config.attention_dim)
            self.subcate_cross_attention = ScaledDotProduct_CandidateAttention(self.word_embedding_dim, config.cate_embed_size, config.attention_dim)
            self.cate_fc = nn.Linear(in_features=self.cate_embed_size * 2, out_features=self.word_embedding_dim, bias=False)

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.subcategory_embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.subcategory_embedding.weight[0])

        nn.init.xavier_uniform_(self.title_H.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.title_M.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.title_M.bias)
        nn.init.xavier_uniform_(self.content_H.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.content_M.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.zeros_(self.content_M.bias)
        self.title_self_attention.initialize()
        self.content_self_attention.initialize()
        self.cate_cross_attention.initialize()
        self.subcate_cross_attention.initialize()
        self.title_base_attention.initialize()
        self.content_base_attention.initialize()
        self.title_add_attention.initialize()
        self.content_add_attention.initialize()
    
    # @torchsnooper.snoop()
    def forward(self, data):
        title_text, content_text, title_mask, content_mask, category, subCategory = data
        batch_size = category.size(0)
        news_num = category.size(1)
        # print(batch_size, news_num)
        # cate_mask = torch.Tensor(np.ones((batch_size * news_num, 2))).cuda()
        title_mask = title_mask.view([batch_size * news_num, self.max_title_length])                                                                       # [batch_size * news_num, max_title_length]
        content_mask = content_mask.view([batch_size * news_num, self.max_content_length])                                                                 # [batch_size * news_num, max_content_length]
        # title_mask = torch.cat([cate_mask, title_mask], dim=1)
        # content_mask = torch.cat([cate_mask, content_mask], dim=1)

        # 1. word embedding
        title = self.word_embedding(title_text).view([batch_size * news_num, self.max_title_length, self.word_embedding_dim])                # [batch_size * news_num, max_title_length, word_embedding_dim]
        content = self.word_embedding(content_text).view([batch_size * news_num, self.max_content_length, self.word_embedding_dim])          # [batch_size * news_num, max_content_length, word_embedding_dim]
        
        # title = torch.cat([category_representation, subCategory_representation, title], dim=1)
        # content = torch.cat([category_representation, subCategory_representation, content], dim=1)
        title_vector_h, _ = self.title_base_attention(title,title,title, title_mask)
        title_vector_m = self.title_add_attention(title_vector_h, title_mask)
        
        content_vector_h, _ = self.content_base_attention(content,content,content, content_mask)
        content_vector_m = self.content_add_attention(content_vector_h, content_mask)

        title_gate = torch.sigmoid(self.title_H(title_vector_h) + self.title_M(content_vector_m).unsqueeze(dim=1))                                  # [batch_size * news_num, max_title_length, hidden_dim * 2]
        content_gate = torch.sigmoid(self.content_H(content_vector_h) + self.content_M(title_vector_m).unsqueeze(dim=1))                            # [batch_size * news_num, max_content_length, hidden_dim * 2]
        title_h = title_vector_h * title_gate                                                            # [batch_size * news_num, max_title_length, hidden_dim * 2]
        content_h = content_vector_h * content_gate                                                    # [batch_size * news_num, max_content_length, hidden_dim * 2]
        # 3. self-attention
        title_self = self.title_self_attention(title_h, title_mask)                                                                                        # [batch_size * news_num, hidden_dim * 2]
        content_self = self.content_self_attention(content_h, content_mask)                                                                                # [batch_size * news_num, hidden_dim * 2]
        # 4. cross-attention
                
        category_rep = self.category_embedding(category).view([batch_size* news_num, -1])                                                                                    # [batch_size, news_num, category_embedding_dim]
        subCategory_rep = self.subcategory_embedding(subCategory).view([batch_size* news_num, -1])  
        title_cate_cross = self.cate_cross_attention(title_h, category_rep, title_mask)      
        title_subcate_cross = self.subcate_cross_attention(title_h, subCategory_rep, title_mask)                                                                        # [batch_size * news_num, hidden_dim * 2]
        content_cate_cross = self.cate_cross_attention(content_h, category_rep, content_mask)    
        content_subcate_cross = self.subcate_cross_attention(content_h, subCategory_rep, content_mask)                                                                  # [batch_size * news_num, hidden_dim * 2]
        news_representation = torch.cat([title_self + title_cate_cross + title_subcate_cross,\
                                        content_self + content_cate_cross + content_subcate_cross],\
                                        dim=1
                                        ).view([batch_size, news_num, self.word_embedding_dim * 2]) # [batch_size, news_num, hidden_dim * 4]
        # 5. feature fusion
        # news_representation = self.feature_fusion(news_representation, category, subCategory)                                                              # [batch_size, news_num, news_embedding_dim]
        return news_representation
    
    def feature_fusion(self, news_representation, category, subCategory):
        category_representation = self.category_embedding(category)                                                                                    # [batch_size, news_num, category_embedding_dim]
        subCategory_representation = self.subcategory_embedding(subCategory)   
        topic = torch.cat([self.dropout(category_representation), self.dropout(subCategory_representation)], dim=-1 )                                                                      # [batch_size, news_num, subCategory_embedding_dim]
        news_representation = torch.cat([news_representation, topic], dim=2) # [batch_size, news_num, news_embedding_dim]
        return news_representation

class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.multi_head_self_attention = MultiHeadSelfAttention(config.user_heads_num, 
                                                     config.news_feature_size,
                                                     config.dropout)
                                                     
        self.additive_attention = AdditiveAttention(config.query_vector_dim, 
                                                    config.news_feature_size)
    # @torchsnooper.snoop()
    def forward(self, news_vectors,attn_masks):
        attn_output, _ = self.multi_head_self_attention(news_vectors,news_vectors,news_vectors,mask=attn_masks)
        user_vector = self.additive_attention(attn_output,attn_masks)
        return user_vector



class Model(torch.nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = UserEncoder(config)
        self.news_encoder.initialize()
        #self.encoder= Transformer_Encoder(config)
        #self.fc = nn.Linear(config.feature_size*2,  1)
        self.config=config
        self.device=config.device

    # @torchsnooper.snoop()
    def forward(self, batch):
        if self.config.dataset == "GLOBO":
            candidate_vector =self.news_encoder(batch['candidate_ids'].to(self.device))
            browsed_vector =self.news_encoder(batch['browsed_ids'].to(self.device))
        else:
            _input=(batch['browsed_titles'].to(self.device), 
                    batch['browsed_absts'].to(self.device), 
                    batch['browsed_title_mask'].to(self.device), 
                    batch['browsed_abst_mask'].to(self.device), 
                    batch['browsed_categ_ids'].to(self.device), 
                    batch['browsed_subcateg_ids'].to(self.device), 
                    #batch['browsed_entity_ids'].to(self.device)
                    )
            candidate_input=(batch['candidate_titles'].to(self.device), 
                            batch['candidate_absts'].to(self.device), 
                            batch['candidate_title_mask'].to(self.device), 
                            batch['candidate_abst_mask'].to(self.device), 
                            batch['candidate_categ_ids'].to(self.device),
                            batch['candidate_subcateg_ids'].to(self.device)
                            )
            candidate_vector =self.news_encoder(candidate_input)
        
            browsed_vector =self.news_encoder(_input)
         
       # browsed_abst_ids=batch['browsed_absts'].to(self.config.device).permute(1,0,2)
        #browsed_vector = torch.stack([self.news_encoder(x) for x in  browsed_title_ids ])
        #browsed_vector=browsed_vector.permute(1,0,2)
        sample_masks=batch['candidate_mask'].to(self.config.device)

        # b*E
        user_vector = self.user_encoder(browsed_vector,batch['browsed_mask'].to(self.config.device)).unsqueeze(1)
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
                news_vector = self.news_encoder((batch['titles'].to(self.device).view(256, batch_size // 256, -1 ),\
                                                batch['absts'].to(self.device).view(256, batch_size // 256, -1 ),
                                                batch['title_mask'].to(self.device).view(256, batch_size // 256, -1 ),\
                                                batch['abst_mask'].to(self.device).view(256, batch_size // 256, -1 ),\
                                                batch['categ_ids'].to(self.device).view(256, -1 ),\
                                                batch['subcateg_ids'].to(self.device).view(256, -1))).view(-1, self.config.news_feature_size)
                self.news_embeds[i*batch_size + 1: (i+1)*batch_size + 1].copy_(news_vector)
            else:
                sz = len(batch['titles'])
                news_vector = self.news_encoder((batch['titles'].to(self.device).view(sz, 1, -1),\
                                                batch['absts'].to(self.device).view(sz, 1, -1 ),
                                                batch['title_mask'].to(self.device).view(sz, 1, -1 ),\
                                                batch['abst_mask'].to(self.device).view(sz, 1, -1 ),\
                                                batch['categ_ids'].to(self.device),\
                                                batch['subcateg_ids'].to(self.device))).view(-1, self.config.news_feature_size)
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