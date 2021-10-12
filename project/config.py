
import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

class Config(object):
    """配置参数"""
    def __init__(self, model_name='NRMS',dataset='MIND'):
        """
        通用配置
        """
        self.model_name = model_name
        self.data_path='./dataset_processed/'
        self.word_embedding_pretrained =  f'/{dataset}/all_word_embedding.npz'
        self.entity_embedding_pretrained=  f'/{dataset}/entitiy_embeds.npz'
        self.save_path =   './save_model/'        # 模型训练结果
        self.log_path =  './logs/' + self.model_name
        self.train_data=f'/{dataset}/train_datas.pkl'
        self.val_data=f'/{dataset}/val_datas.pkl'
        self.test_data=f'/{dataset}/test_datas.pkl'

        self.save_flag=False  # 模型存储标志

        self.word_freq_threshold=3 # 字典构建中 词汇低频阈值

        self.random_seed=1024 # 随机种子
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.cate_embed_size=100 # 类别嵌入大小
        
        self.word_embed_size=300                                      #单词嵌入维度
        self.num_epochs = 5                                           # epoch数
        self.eval_step=5000
        self.batch_size = 256                                          # mini-batch大小
        self.learning_rate = 1e-3   
        self.dropout = 0.2                                              # 随机失活
        self.require_improvement = 10000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.warm_up_steps=500
        self.warm_up=False
        if dataset == "MIND":
            self.__MIND__()
    
    def __MIND__(self):
        """
        配置 mind数据集构造及设置参数
        """
        self.category_nums=18 + 1  #类别数量 
        self.n_words=38284    # 单词数量
        self.subcategory_nums=285+1 # 子类别数量
        self.entity_nums=10      # 单个新闻实体数量
        self.n_words_title=20  # 标题单词长度
        self.n_words_abst=40   # 摘要单词长度
        self.history_len=50  # 历史序列长度
        self.negsample_size=4   # 负样本采样规模
        self.max_candidate_size=200  # 候选样本最大长度
    
    """
    NRMS
    """
    def __nrms__(self):
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


      

    