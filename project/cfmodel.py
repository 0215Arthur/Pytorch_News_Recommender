"""
基于反事实推断进行数据增强尝试
Counterfactual Data-Augmented Sequential Recommendation

CauseRec: Counterfactual User Sequence Synthesis for Sequential Recommendation
https://github.com/gzy-rgb/CauseRec
"""

"""
反事实逻辑：
给定用户历史序列： 如果用户之前看的新闻不一样，那么他可能会浏览什么？
1. 如果没有看过某篇文章
2. 如果用户只看过某篇文章
3. 如果用户历史上看的不一样： 不同的新闻 或者 内容变化过的新闻

anchor model 与 sampler model 
"""

