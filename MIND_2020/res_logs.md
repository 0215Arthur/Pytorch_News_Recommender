

模型| 验证集| 测试集 | 数据使用|实验设置 | 实验模型组别
:----|----:|----:|:----|:----|:----
NRMS-bert1024|0.672 | 0.6455 | 新闻id对应的bert向量| 6epoch 512batch 向量可微调  |T11-25_19.57_nrms_11-25_19
NRMS-glove300|0.675 |    -   | 新闻对应的标题词汇(20) |6epoch 512batch no browsed_mask |T11-25_21.36_NRMS_V1_1124
NRMS-glove300|/|/|新闻对应的标题词汇(20) |6epoch 512batch no browsed_mask/实现方式为github |

