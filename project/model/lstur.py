# coding: UTF-8
"""
Neural News Recommendation with Long- and Short-term User Representations
ACL2019 https://aclanthology.org/P19-1033.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import torchsnooper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
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
        self.pretrained = False
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
            self.category_embedding =nn.Embedding(config.category_nums, config.cate_embed_size, padding_idx=0)
            self.subcategory_embedding =nn.Embedding(config.subcategory_nums, config.cate_embed_size, padding_idx=0)
            self.word_embedding = [nn.Embedding.from_pretrained(torch.tensor(
                np.load( config.data_path +config.word_embedding_pretrained)["embeddings"].astype('float32')), 
                freeze=True,padding_idx=0).to(config.device),
                nn.Dropout(p=config.dropout, inplace=False)]

            self.word_embedding = nn.Sequential(*self.word_embedding)
        
            assert config.kernel_sizes >= 1 and config.kernel_sizes % 2 == 1
            self.title_CNN = nn.Conv2d(
                1,
                config.num_filters,
                (config.kernel_sizes, config.word_embed_size),
                padding=(int((config.kernel_sizes - 1) / 2), 0))
            self.title_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.num_filters)
    # @torchsnooper.snoop()
    def forward(self, data):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, num_filters * 3
        """
        # Part 1: calculate category_vector
        if self.pretrained:
            return self.article_embedding(data)
        else:  
            title_ids, news_categ,news_subcateg = data
        # batch_size, num_filters
            category_vector = self.category_embedding(news_categ)

            # Part 2: calculate subcategory_vector

            # batch_size, num_filters
            subcategory_vector = self.subcategory_embedding(
                news_subcateg)

            # Part 3: calculate weighted_title_vector

            # batch_size, num_words_title, word_embedding_dim
            title_vector = F.dropout(self.word_embedding(title_ids),
                                    p=self.config.dropout,
                                    training=self.training)
            # batch_size, num_filters, num_words_title
            convoluted_title_vector = self.title_CNN(
                title_vector.unsqueeze(dim=1)).squeeze(dim=3)
            # batch_size, num_filters, num_words_title
            activated_title_vector = F.dropout(F.relu(convoluted_title_vector),
                                            p=self.config.dropout,
                                            training=self.training)
            # batch_size, num_filters
            weighted_title_vector = self.title_attention(
                activated_title_vector.transpose(1, 2))

            # batch_size, num_filters * 3
            news_vector = torch.cat(
                [category_vector, subcategory_vector, weighted_title_vector],
                dim=1)
            return news_vector
    
   
class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        #assert int(config.num_filters * 1.5) == config.num_filters * 1.5
        # self.gru = nn.GRU(
        #     config.news_encoder_size,
        #     config.news_encoder_size if config.long_short_term_method == 'ini'
        #     else int(config.news_encoder_size / 2))
        self.lstm = nn.LSTM(config.news_encoder_size, 
                    config.news_encoder_size if config.long_short_term_method == 'ini' else int(config.news_encoder_size / 2), 
                    batch_first=True, bidirectional=False)
        # self.initialize()
    # @torchsnooper.snoop()
    def forward(self, user, clicked_news_length, clicked_news_vector):
        """
        Args:
            user:
                ini: batch_size, num_filters * 3
                con: batch_size, num_filters * 1.5
            clicked_news_length: batch_size,
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters * 3
        Returns:
            (shape) batch_size, num_filters * 3
        """
        clicked_news_length[clicked_news_length == 0] = 1
        # 1, batch_size, num_filters * 3
        if self.config.long_short_term_method == 'ini':
            packed_clicked_news_vector = pack_padded_sequence(
                clicked_news_vector,
                clicked_news_length,
                batch_first=True,
                enforce_sorted=False)
            # _, (last_hidden, _) = self.title_lstm(sorted_title)
            _, (last_hidden, _) = self.lstm(packed_clicked_news_vector,
                                      user.unsqueeze(dim=0))
            # gru
            # _, last_hidden = self.gru(packed_clicked_news_vector,
            #                           user.unsqueeze(dim=0))
            return last_hidden.squeeze(dim=0)
        else:
            packed_clicked_news_vector = pack_padded_sequence(
                clicked_news_vector,
                clicked_news_length,
                batch_first=True,
                enforce_sorted=False)
            # _, last_hidden = self.gru(packed_clicked_news_vector)
            _, (last_hidden, _) = self.lstm(packed_clicked_news_vector)
            return torch.cat((last_hidden.squeeze(dim=0), user.squeeze()), dim=1)

    def initialize(self):
        for parameter in self.lstm.parameters():
            if len(parameter.size()) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.zeros_(parameter.data)

class Model(torch.nn.Module):
    """
    LSTUR network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, config):
        """
        # ini
        user embedding: num_filters * 3
        news encoder: num_filters * 3
        GRU:
        input: num_filters * 3
        hidden: num_filters * 3
        # con
        user embedding: num_filter * 1.5
        news encoder: num_filters * 3
        GRU:
        input: num_fitlers * 3
        hidden: num_filter * 1.5
        """
        super(Model, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = UserEncoder(config)
        self.user_encoder.initialize()
        #self.click_predictor = DotProductClickPredictor()
        print("model name: lstur")
        self.user_embedding = nn.Embedding(
            config.user_nums,
            config.news_encoder_size if config.long_short_term_method == 'ini'
            else int(config.news_encoder_size /2),
            padding_idx=0)
        self.device=config.device
    # @torchsnooper.snoop()
    def forward(self, batch):
        """
        Args:
        Returns:
            click_probability: batch_size
        """

        user = F.dropout2d(self.user_embedding(batch['user_id'].to(self.device)).unsqueeze(dim=0),
                           p=self.config.masking_probability,
                           training=self.training).squeeze(dim=0)
        # batch_size, 1 + K, num_filters * 3
        #  
        if self.config.dataset=="GLOBO":
            clicked_news_vector=self.news_encoder(batch["browsed_ids"].to(self.device))
            candidate_news_vector=self.news_encoder(batch["candidate_ids"].to(self.device))
        else:
            clicked_news=batch['browsed_titles'].to(torch.device('cuda')).permute(1,0,2) 
            
            browsed_categ=batch['browsed_categ_ids'].to(torch.device('cuda')).permute(1,0)
            
            browsed_subcateg=batch['browsed_subcateg_ids'].to(torch.device('cuda')).permute(1,0)

            candidate_news=batch['candidate_titles'].to(torch.device('cuda')).permute(1,0,2)

            candidate_categ=batch['candidate_categ_ids'].to(torch.device('cuda')).permute(1,0)

            candidate_subcateg=batch['candidate_subcateg_ids'].to(torch.device('cuda')).permute(1,0)

            candidate_news_vector = torch.stack(
                [self.news_encoder(x) for x in zip(candidate_news,candidate_categ,candidate_subcateg)], dim=1)
            # ini: batch_size, num_filters * 3
            # con: batch_size, num_filters * 1.5


            # batch_size, num_clicked_news_a_user, num_filters * 3
            clicked_news_vector = torch.stack(
                [self.news_encoder(x) for x in zip(clicked_news,browsed_categ,browsed_subcateg)], dim=1)
        # batch_size, num_filters * 3
        sample_masks=batch['candidate_mask'].to(self.device)
        
        browsed_lens=batch['browsed_lens'].to(self.device)

        user_vector = self.user_encoder(user, browsed_lens,
                                        clicked_news_vector).unsqueeze(1)
        # batch_size, 1 + K
        
        # print(candidate_news_vector.size())
        pred =  torch.sum(user_vector*candidate_news_vector,2)
        if batch['candidate_mask'] is not None:
            pred = pred.masked_fill(sample_masks == 0, -1e9)

        return pred
    
 