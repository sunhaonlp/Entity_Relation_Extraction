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

![TIM图片20191118155833](https://github.com/1024642475/Entity_Relation_Extraction/blob/master/demo_photo.png)

# 第二阶段 关系抽取

第二阶段我们采用的是BiLSTM+ATT模型，可以参考[这篇博客](https://blog.csdn.net/buppt/article/details/82961979)

附上[官方论文](https://www.aclweb.org/anthology/P16-2034.pdf)

基于上面提到的实体类型，我们定义了5种关系类型

1.合作（企业和企业之间） 

2.竞争（企业和企业之间） 

3.属于（企业和产品之间） 

4.监管（有关部门和企业之间）

5.管理（人物和企业之间） 


# 数据集

我们使用自己编写的数据标注脚本标注了1525笔金融领域实体抽取的数据集和450笔关系抽取的数据集，数据集文件已上传至服务器，有需要者请自取

## 实体数据集示例：

```
{
    "entity-mentions": [
        {
            "text": "征信中心",
            "entity-type": "department",
            "start": 0,
            "end": 4
        },
        {
            "text": "四川新网银行股份有限公司",
            "entity-type": "company",
            "start": 30,
            "end": 42
        }
    ],
    "sentence": "征信中心回应不对数据真实性负责国家企业信用信息公示系统显示，四川新网银行股份有限公司成立于年，住所位于中国（四川）自由贸易试验区成都高新区，注册资金亿元"
}
```
链接：https://pan.baidu.com/s/1GJoAlkgogWmPxB_ZEQ_qPA 
提取码：d1b3 

## 关系数据集示例：

    上海市工商局  拼多多平台   监管  其后，国家市场监督管理总局网监司高度重视媒体反映的拼多多平台上销售侵权假冒商品等问题，已经要求上海市工商局约谈平台经营者

链接：https://pan.baidu.com/s/1qkI2WPLOllp7gUKhZ0nL_A 
提取码：zgwc 

## 预训练模型：
链接：https://pan.baidu.com/s/1LAP65tdRX6MuMzhw9nsYMg 
提取码：3lgw 
