
import torch
import os

from torch.utils.data import dataset
#os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

class Config(object):
    """配置参数"""
    def __init__(self, model_name='NRMS',dataset='MIND'):
        """
        通用配置
        """
        self.version = ""
        self.model_name = model_name.lower()
        self.data_path='./dataset_processed/'
        self.save_path =   './save_model/'        # 模型训练结果
        self.log_path =  './logs/' + self.model_name
        self.train_data=f'{dataset}/train_datas.pkl'
        self.val_data=f'{dataset}/val_datas.pkl'
        self.test_data=f'{dataset}/test_datas.pkl'
        self.user_dict=f'{dataset}/user_dict.pkl'

        self.save_flag=True  # 模型存储标志

        self.word_freq_threshold=3 # 字典构建中 词汇低频阈值

        self.random_seed=1024 # 随机种子
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备        

        self.eval_step=400
        self.batch_size = 128                                          # mini-batch大小
        self.learning_rate = 1e-3   
        self.dropout = 0.2                                              # 随机失活
        self.require_improvement = 10000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.warm_up_steps=500
        self.warm_up=False
        if dataset.upper() == "MIND":
            self.__MIND__()
        elif dataset.upper() == "GLOBO":
            self.__GLOBO__()
        elif dataset.upper() == "ADR":
            self.__ADR__()
        self.dataset = dataset.upper()
        if model_name.lower() == "nrms":
            self.__nrms__()
        elif model_name.lower() == "dist":
            self.__dist__()
        elif model_name.lower() == "lstur":
            self.__lstur__()
        elif model_name.lower() == 'cne':
            self.__cne__()
        elif model_name.lower() == 'cnep':
            self.__cnep__()
        elif model_name.lower() == 'fim':
            self.__fim__()

    def __MIND__(self):
        """
        配置 mind数据集构造及设置参数
        """
        self.word_embed_size=300                                      #单词嵌入维度
        self.category_nums=18 + 1  #类别数量 
        self.n_words=38284    # 单词数量
        self.num_epochs = 5                                           # epoch数
        self.subcategory_nums=285+1 # 子类别数量
        self.entity_nums=10      # 单个新闻实体数量
        self.n_words_title=20  # 标题单词长度
        self.n_words_abst=40   # 摘要单词长度
        self.history_len=50  # 历史序列长度
        self.negsample_size=4   # 负样本采样规模
        self.max_candidate_size=100  # 候选样本最大长度
        self.user_nums=458044
        self.news_nums=104151
        self.word_embedding_pretrained =  'MIND/all_word_embedding.npz'
        self.entity_embedding_pretrained=  'MIND/entitiy_embeds.npz'
    
    def __GLOBO__(self):
        self.article_embed_size=250                                      #单词嵌入维度
        self.category_nums=20 + 1  #类别数量 
        self.n_words=38284    # 单词数量
        self.num_epochs = 10                                           # epoch数
        self.subcategory_nums=285+10 # 子类别数量
        self.history_len=20  # 历史序列长度
        self.negsample_size=3   # 负样本采样规模
        self.max_candidate_size=200  # 候选样本最大长度
        self.news_nums=21482
        self.user_nums=38364
        self.article_embedding_pretrained =  'GLOBO/article_embed.npz'
    
    def __ADR__(self):
        self.word_embed_size=50                                      #单词嵌入维度
        self.category_nums=18 + 1  #类别数量 
        self.n_words=18417    # 单词数量
        self.num_epochs = 10                                           # epoch数
        self.subcategory_nums=103 # 子类别数量
        self.entity_nums=10      # 单个新闻实体数量
        self.n_words_title=10  # 标题单词长度
        # self.n_words_abst=40   # 摘要单词长度
        self.history_len=20  # 历史序列长度
        self.negsample_size=4   # 负样本采样规模
        self.max_candidate_size=100  # 候选样本最大长度
        self.user_nums=203453
        self.news_nums=15309
        self.word_embedding_pretrained =  'MIND/all_word_embedding.npz'
        self.entity_embedding_pretrained=  'MIND/entitiy_embeds.npz'

    def __nrms__(self):
        # self.title_size=512 
        # self.feature_size=712
        if self.dataset == "GLOBO":
            self.news_feature_size=250 #600
            self.query_vector_dim=200 
            self.news_encoder_size=250
            self.query_vector_dim_large= 400 
            self.user_heads_num=5 
        else: 
            self.news_feature_size=600

            self.query_vector_dim=200  # additive attn 向量维度
            self.query_vector_dim_large= 400 

            self.news_encoder_size= 600

            self.user_heads_num=6      # mind: 6
        self.num_heads_2=4         # 

        self.kernel_sizes = 3 # 
        self.kernel_sizes_2 = [2,4]

        self.filter_nums_2=50
        self.title_heads_num=6
  
    def __gnud__(self):
        self.title_size=512 
        self.feature_size=712
        self.news_feature_size=600
        self.news_nums=104151
        self.filter_nums=100
        # self.bert_embed_size=512

        self.query_vector_dim=200  # additive attn 向量维度
        self.query_vector_dim_large= 400 

        self.news_encoder_size=600
        self.long_short_term_method='ini'

        self.user_heads_num=6      # 
        self.num_heads_2=4         # 
        self.list_num_heads=8         # 

        self.kernel_sizes = [2,4,5] # 
        self.num_filters=400
        self.filter_nums_2=50
        self.title_heads_num=6
        self.num_attention_heads=10

    def __dist__(self):
        self.title_size=512 
        self.feature_size=712
        self.news_feature_size=300
        self.news_nums=104151
        self.filter_nums=100
        # self.bert_embed_size=512

        self.query_vector_dim=200  # additive attn 向量维度
        self.query_vector_dim_large= 400 

        self.news_encoder_size=600
        self.long_short_term_method='ini'

        self.user_heads_num=6      # 
        self.num_heads_2=4         # 
        self.list_num_heads=8         # 

        self.kernel_sizes = [2,4,5] # 
        self.num_filters=400
        self.filter_nums_2=50
        self.title_heads_num=6
        self.num_attention_heads=10

    def __hierec__(self):
        self.title_size=512 
        self.feature_size=712
        self.news_feature_size=600
        
        # self.bert_embed_size=512

        self.query_vector_dim=200  # additive attn 向量维度
        self.query_vector_dim_large= 400 

        self.news_encoder_size=600
        self.long_short_term_method='ini'

        self.user_heads_num=6      # 
        self.num_heads_2=4         # 
        self.list_num_heads=8         # 

        self.kernel_sizes = 3 # 
        self.kernel_sizes_2 = [2,4]
        self.num_filters=400
        self.filter_nums_2=50
        self.title_heads_num=6
        self.num_attention_heads=10

    def __lstur__(self):
        self.long_short_term_method='con'
        if self.dataset=="GLOBO":
            self.news_encoder_size= 250
        else:
            self.news_encoder_size= 500
        self.masking_probability=0.5
        self.kernel_sizes = 3 # 
        self.num_filters=300
        self.query_vector_dim=200
    
    def __cne__(self):
        self.cate_embed_size=200 # 类别嵌入大小
        self.hidden_dim=200
        self.attention_dim=200
        self.query_vector_dim=200  # additive attn 向量维度
        self.news_feature_size= 800
        self.user_heads_num=8      # mind: 6

    def __cnep__(self):
        self.cate_embed_size=100 # 类别嵌入大小
        self.attention_dim=200
        self.query_vector_dim=200  # additive attn 向量维度
        self.news_feature_size= 800
        self.user_heads_num=4      # mind: 6
        self.title_heads_num=6

    def __fim__(self):
        self.dilation_level=3
        self.npratio=4
        self.filter_num=150
        self.kernel_size=3

        