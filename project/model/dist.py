# coding: UTF-8
"""
Graph Neural News Recommendation with Unsupervised Preference Disentanglement
https://aclanthology.org/2020.acl-main.392.pdf
"""
import torch
import torch.nn as nn
import numpy as np
import torchsnooper
import torch.nn.functional as F
from torch.autograd import Variable
from pyinstrument import Profiler
import time 
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
        # self.entity_embedding = nn.Embedding.from_pretrained(torch.tensor(
        #     np.load( config.data_path +config.entity_embedding_pretrained)["embeddings"].astype('float32')), 
        #      freeze=False).to(config.device)

        self.multi_head_self_attention = MultiHeadSelfAttention(config.title_heads_num, config.word_embed_size,config.dropout)
        self.additive_attention = AdditiveAttention(config.query_vector_dim, config.word_embed_size)
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
        self.dropout=nn.Dropout(config.dropout)
        # news_dense = [
        #     nn.Linear(config.final_embed_size, config.feature_size),
        #     GELU()#,
        #     #nn.Linear(config.feature_size, config.feature_size)
        #     #nn.ReLU( )
        # ]
        # # self.news_dense = nn.Sequential(*news_dense)
        # #self.Linear= nn.Linear(config.final_embed_size, config.final_embed_size)
        # self.Linear=nn.Sequential(*news_dense)
    # @torchsnooper.snoop()
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
            attn_output, _ = self.multi_head_self_attention(title_embedded,title_embedded,title_embedded)
            title_vector = self.additive_attention(attn_output)
            title_embeds_list.append(title_vector)
         
        title_vector=torch.stack(title_embeds_list).permute(1,0,2)

        # abst_ids=abst_ids.permute(1,0,2)
         
        # for _ids in abst_ids:
        #     abst_embedded = self.word_embedding(_ids) 
        #     attn_output = self.multi_head_self_attention(abst_embedded,abst_embedded,abst_embedded)
        #     abst_vector = self.additive_attention(attn_output)
        #     abst_embeds_list.append(abst_vector)
         
        # abst_vector=torch.stack(abst_embeds_list).permute(1,0,2)

        # categ_embeds = self.category_embedding(news_categ)
        # subcateg_embeds = self.subcategory_embedding(news_subcateg)
        news_vector=title_vector#torch.cat([title_vector,abst_vector],-1)
        news_vector=self.dropout(news_vector)


        # abst_embedded = self.word_embedding(abst_ids) 
        # attn_output, _ = self.multi_head_self_attention(abst_embedded,abst_embedded,abst_embedded,mask=attn_masks)
        # abst_vector = self.additive_attention(attn_output)

        return news_vector


class RoutingLayer(nn.Module):
    def __init__(self, feature_size, K, num_iterations=7):
        super(RoutingLayer, self).__init__()

        self.num_iterations = num_iterations

        projs=[]
        for i in range(K):
            proj = nn.Sequential(
                nn.Linear(feature_size, feature_size // K),
                nn.ReLU()
            )
            projs.append(proj)
        self.projs = nn.ModuleList(projs)

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    #@torchsnooper.snoop()
    def l2(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        # scale = squared_norm / (1 + squared_norm)
        return tensor / torch.sqrt(squared_norm)
    
    #@torchsnooper.snoop()
    def forward(self, x):
        """
        x: batch * dim
        """

        # N * L * dim///K 得到子空间特征  => N * 1 * L * dim//K
        x = [self.l2(proj(x)).unsqueeze(1) for proj in self.projs]
        x = torch.cat(x, axis=1) # N * K * L * dim格式
        # print("x:", x.size())
        target = x[:,:,0,:].unsqueeze(-2)  # N * k * dim
        news = x[:, :, 1:, :]
        #x = x.permute(0, 1, 2, 3)
        # print("target:", target.size())
        # print("news:", news.size())
        alpha = torch.sum(target * news, axis=-1)
        # print("alpha:", alpha.size())
        alpha = F.softmax(alpha, dim = -1).unsqueeze(-1).repeat(1, 1, 1, news.size(-1)) # N * k * 1 * L
        output = target.squeeze() + torch.sum(alpha * news, axis = 2) # k * N

        return self.l2(output)

class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.topic_embeddings=nn.Embedding(config.subcategory_nums, config.word_embed_size)
        self.multi_head_self_attention = MultiHeadSelfAttention(config.title_heads_num, config.word_embed_size,config.dropout, config.word_embed_size)
        self.additive_attention = AdditiveAttention(config.query_vector_dim, config.word_embed_size)

    #@torchsnooper.snoop()
    def forward(self, news_vectors, topic_ids,attn_masks, i=0):
        """
        batch * N(历史长度50) * dim
        """
        # print(topic_ids)
        topic_embeds = self.topic_embeddings(topic_ids) # batch * 1 * dim
        # print("topic size:",topic_embeds.size())
        attn_output, _ = self.multi_head_self_attention(news_vectors,news_vectors,news_vectors,mask=attn_masks, topic=topic_embeds)
        if not self.training:
            torch.save(_, "./tmp/{}_topic.pt".format(i))
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
        # self.norm = nn.LayerNorm(config.feature_size*2)
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
        # profiler = Profiler()
        # profiler.start()
        _input=(batch['browsed_titles'].to(self.device), 
                batch['browsed_absts'].to(self.device), 
               # batch['browsed_categ_ids'].to(self.device), 
               # batch['browsed_subcateg_ids'].to(self.device), 
                #batch['browsed_entity_ids'].to(self.device)
                )
        candidate_input=(batch['candidate_titles'].to(self.device), 
                         batch['candidate_absts'].to(self.device), 
                        # batch['candidate_categ_ids'].to(self.device),
                         #batch['candidate_subcateg_ids'].to(self.device),
                         #batch['candidate_entity_ids'].to(self.device)
                         )

        start = time.time()
        # b*sample_size*E
       # candidate_title_ids=batch['candidate_titles'].to(self.config.device).permute(1,0,2)
        #candidate_abst_ids=batch['candidate_absts'].to(self.config.device).permute(1,0,2)
        candidate_vector =self.news_encoder(candidate_input)# torch.stack([self.news_encoder(x) for x in  candidate_title_ids  ])
        #candidate_vector = torch.stack([self.news_encoder(x) for x in zip(candidate_title_ids,candidate_abst_ids) ])
        #candidate_vector=candidate_vector.permute(1,0,2)
        # b*history_len*E
        #print(time.time() - start)
        start = time.time() 
        browsed_vector =self.news_encoder(_input)
        #print(time.time() - start)
       # browsed_abst_ids=batch['browsed_absts'].to(self.config.device).permute(1,0,2)
        #browsed_vector = torch.stack([self.news_encoder(x) for x in  browsed_title_ids ])
        #browsed_vector=browsed_vector.permute(1,0,2)
        sample_masks=batch['candidate_mask'].to(self.config.device)

        start = time.time() 
        # b*E
        user_vector = self.user_encoder(browsed_vector,\
                                        batch['browsed_subcateg_ids'].to(self.config.device),\
                                        batch['browsed_mask'].to(self.config.device)
                                        ).unsqueeze(1)
        #print(time.time() - start)
        pred =  torch.sum(user_vector*candidate_vector,2)
        if batch['candidate_mask'] is not None:
            pred = pred.masked_fill(sample_masks == 0, -1e9)
        # profiler.stop()

        # profiler.print()
        return pred
    
    def update_rep(self, news_iter, batch_size=1280):
        """
        更新得到新闻标注
        """
        self.news_embeds = torch.Tensor(self.config.news_nums + 1, self.config.news_feature_size).cuda()

        for i,batch in enumerate(news_iter):
            if len(batch['titles']) == batch_size:
                news_vector = self.news_encoder((batch['titles'].to(self.device).view(256, batch_size // 256, -1 ),\
                                                batch['absts'].to(self.device).view(256, batch_size // 256, -1 ))).contiguous().view(-1, self.config.news_feature_size)
                self.news_embeds[i*batch_size + 1: (i+1)*batch_size + 1].copy_(news_vector)
            else:
                sz = len(batch['titles'])
                news_vector = self.news_encoder((batch['titles'].to(self.device).view(sz, 1, -1),\
                                                batch['absts'].to(self.device).view(sz, 1, -1 ))).contiguous().view(-1, self.config.news_feature_size)
                self.news_embeds[i*batch_size + 1:].copy_(news_vector)
        return 

    
    def predict(self, batch):
        """
        快速进行评估测试
        """
        browsed_vector = self.news_embeds[batch['browsed_ids'].to(self.config.device)]
        user_vector = self.user_encoder(browsed_vector,batch['browsed_subcateg_ids'].to(self.config.device),batch['browsed_mask'].to(self.config.device)).unsqueeze(1)
        candidate_vector =self.news_embeds[batch['candidate_ids'].to(self.config.device)]
        # print(user_vector.size())
        # print(candidate_vector.size())
        pred =  torch.sum(user_vector*candidate_vector,2)
        if batch['candidate_mask'] is not None:
            pred = pred.masked_fill(batch['candidate_mask'].to(self.config.device) == 0, -1e9)

        return pred