# Entity_Relation_Extraction

本项目使用

- python 3.5
- pytorch 1.0.0
- transformers  2.1.1

# 第一阶段 实体抽取

第一阶段我们使用的是Bert模型，对Bert不了解的可以看看[这篇博客]( https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html )。

附上Bert的[官方论文](https://arxiv.org/abs/1810.04805)

我们新定义了5种实体类型：

1.企业（company)  2.人物(people)  3.产品(product)  4.有关部门(department)

运行截图：

![TIM图片20191118155833](https://github.com/1024642475/Entity_Relation_Extraction/blob/masr/demo_photo.png)

# 第二阶段 关系抽取

第二阶段我们采用的是BiLSTM+ATT模型，可以参考[这篇博客](https://blog.csdn.net/buppt/article/details/82961979)

附上[官方论文](https://www.aclweb.org/anthology/P16-2034.pdf)

基于上面提到的实体类型，我们定义了5种关系类型

1.合作（企业和企业之间） 

2.竞争（企业和企业之间） 

3.属于（企业和产品之间） 

4.监管（有关部门和企业之间）

5.管理（人物和企业之间） 
