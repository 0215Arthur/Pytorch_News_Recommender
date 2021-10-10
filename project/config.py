
import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
class Config(object):

    """配置参数"""
    def __init__(self, model_name='NRMS',dataset='../MIND'):
        self.model_name = model_name
        self.data_path='./dataset_processed/'
        self.train_path = dataset + '/train/'                                # 训练集
        self.dev_path = dataset + '/dev/'                                    # 验证集
        self.test_path = dataset + '/test/'                                  # 测试集
        self.small_train_path = dataset + '/small_train/'   
        self.small_dev_path = dataset + '/small_dev/'   

        # BERT 新闻向量
        self.bert_embedding_pretrained =    'news_embeds_512.npz'
        self.word_embedding_pretrained =   'all_word_embedding.npz'
        self.entity_embedding_pretrained=  'entitiy_embeds.npz'

        self.mode='large'  # 指定数据集 large /demo 

        self.save_path =   './save_model/'        # 模型训练结果
        self.log_path =  './logs/' + self.model_name
        self.train_data='train_datas.pkl'
        self.dev_data='dev_datas.pkl'
        self.test_data='test_datas.pkl'

        self.n_words_title=20  # 标题单词长度
        self.n_words_abst=40   # 摘要单词长度
        
        self.history_len=50  # 历史序列长度
        self.sample_size=5   # 负样本采样规模
        self.max_candidate_size=300  # 候选样本最大长度

        self.save_flag=True  # 模型存储标志

        self.word_freq_threshold=3 # 字典构建中 词汇低频阈值

        self.random_seed=1024 # 随机种子
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.category_nums=18+1  #类别数量 
        self.n_words=45800    # 单词数量
        self.subcategory_nums=293+1 # 字类别数量

        self.cate_embed_size=100 # 类别嵌入大小
        self.entity_nums=10      # 单个新闻实体数量

        self.word_embed_size=300                                      #单词嵌入维度
        self.num_epochs = 5                                           # epoch数
        self.eval_step=5000
        self.batch_size = 512                                          # mini-batch大小
        self.learning_rate = 1e-3   
        self.dropout = 0.2                                              # 随机失活
        self.require_improvement = 10000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.warm_up_steps=500
        self.warm_up=False
    
    """
    NRMS
    """
    def __nrms__(self):
        self.title_size=512 
        self.feature_size=712
        self.news_feature_size=800
        self.bert_embed_size=512

        self.query_vector_dim=200  # additive attn 向量维度
        self.query_vector_dim_large= 400 

        self.news_encoder_size=600
        self.long_short_term_method='ini'

        self.user_heads_num=8      # 
        self.num_heads_2=4         # 
        self.list_num_heads=8         # 

        self.kernel_sizes = 3 # 
        self.kernel_sizes_2 = [2,4]
        self.num_filters=400
        self.filter_nums_2=50


        self.title_heads_num=6
        self.num_attention_heads=10


      

    