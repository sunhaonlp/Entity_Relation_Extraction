from transformers import BertTokenizer
from transformers import BertForTokenClassification
from transformers import BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json
import os
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import  produce_length, make_label
import pandas as pd
from collections import deque
import collections
import numpy as np
import pickle
import sys
import codecs
import torch.nn as nn
import torch.optim as optim
import torch
import torch.utils.data as D
from torch.autograd import Variable
from BiLSTM_ATT import BiLSTM_ATT

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def translate(word,label):
    """
    根据开始位置和结束位置打标签
    BIOES
    """
    status = 0 # 0 表示未发现实体，1 表示正在处理实体
    begin = 0
    end = 0
    len_entity = 0
    record = []
    for i in range(len(word)):
        if(label[i]=='O'):
            status = 0
            len_entity = 0
            continue
        elif(label[i][0]=='B'):
            status = 1
            begin = i
            len_entity+=1
        elif(label[i][0]=='I'):
            len_entity+=1
        elif(label[i][0]=='E'):
            tmp = {}
            tmp['word'] = ''.join(word[begin:begin+len_entity+1])
            tmp['type'] = label[i][2:]
            record.append(tmp)
    return record

def get_input(debug=True):
    """
    得到Bert模型的输入
    """
    input_word_list = []
    input_label_list = []
    with open("tmp/input_bert.json", 'r', encoding='UTF-8') as f:
        data = json.load(f)
    bert_words = list(data["sentence"])
    label_list = ["O" for _ in bert_words]      # 首先制作全O的标签
    for entity in data["entity-mentions"]:
        en_start = entity["start"]
        en_end = entity["end"]
        en_type = entity["entity-type"]
        # 根据开始与结束位置打标签
        make_label(en_start, en_end, en_type, label_list)
    input_word_list.append(["[CLS]"]+bert_words+["[SEP]"])
    input_label_list.append(["O"]+label_list+["O"])

    return input_word_list, input_label_list

def show_args(args):
    """
    打印参数
    """
    print(args)

def eval_predict(pred_list, label_list):
    """
    评估预测的结果
    """
    #print(label_list)
    #print(pred_list)
    #input()
    f1 = f1_score(label_list, pred_list, average="micro")
    p = precision_score(label_list, pred_list, average="micro")
    r = recall_score(label_list, pred_list, average="micro")
    return f1, p, r

def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def X_padding(words):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = []
    for i in words:
        if i in word2id:
            ids.append(word2id[i])
        else:
            ids.append(word2id["UNKNOW"])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([word2id["BLANK"]] * (max_len - len(ids)))

    return ids

def pos(num):
    if num < -40:
        return 0
    if num >= -40 and num <= 40:
        return num + 40
    if num > 40:
        return 80

def position_padding(words):
    words = [pos(i) for i in words]
    if len(words) >= max_len:
        return words[:max_len]
    words.extend([81] * (max_len - len(words)))
    return words


class labelproducer:
    def __init__(self, all_labels):
        # all_labels是list中嵌套list
        self.target_to_ids = self.get_target_to_ids(all_labels)
        self.ids_to_target = {0:'O', 1: 'B-company', 2: 'I-company', 3: 'E-company', 4: 'B-department', 5: 'I-department', 6: 'E-department', 7: 'B-people', 8: 'E-people', 9: 'I-people', 10: 'B-product', 11: 'I-product', 12: 'E-product'}


    def get_target_to_ids(self, all_labels):
        """
        返回target_to_ids
        """
        target_to_ids = dict()
        for labels in all_labels:
            for label in labels:
                if label not in target_to_ids:
                    target_to_ids[label] = len(target_to_ids)
        return target_to_ids

    def convert_label_to_ids(self, labels):
        """
        将label转换成ids
        输入时label的列表
        """
        ret_ids = []
        for label in labels:
            ret_ids.append(self.target_to_ids[label])
        return ret_ids

    def convert_ids_to_label(self, all_ids):
        """
        将ids转换成label
        输入是ids的列表
        """
        ret_label = []
        for ids in all_ids:
            ret_label.append(self.ids_to_target[ids])
        return ret_label


if __name__=="__main__":

###########以下为bert部分，首先读取输入，将输入处理成了bert可以输入的json文件保存在tmp，后将预测的结果保存在tmp里的output_bert_.txt
    # 定义可见的cuda设备
    # 数据读入部分
    sentence = input("请输入你想预测的句子：")
    dict_bert = {}
    dict_bert['entity-mentions'] = []
    dict_bert['sentence'] = sentence
    final_json = json.dumps(dict_bert, indent=4, ensure_ascii=False)
    with codecs.open("tmp/input_bert.json", 'w', 'utf-8') as file:
        file.write(final_json)

    # 模型部分
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_split", default=0.5, type=float)
    parser.add_argument("--BERT_HOME", default="model/", type=str)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--seed", default=2019, type=int, help="随机模型初始化的随机种子")
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--max_len", default=256, type=int)
    parser.add_argument("--adam_epsilon", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--model_name", default="bert-base-chinese", type=str)
    args = parser.parse_args()
    BERT_HOME = args.BERT_HOME
    tokenizer = BertTokenizer.from_pretrained(os.path.join(BERT_HOME, "vocab.txt"), do_lower_case=True)

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")

    # 设置可能用到的随机变量的种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu>0:
        torch.cuda.manual_seed_all(args.seed)

    # 得到整个训练集
    input_word_list, input_label_list = get_input()
    input_word_list_original = input_word_list[0].copy()
    # 处理长度
    input_word_list, attention_mask = produce_length(input_word_list, args.max_len, "[PAD]", ret_attention_mask=True)
    input_label_list = produce_length(input_label_list, args.max_len, "O", ret_attention_mask=False)
    # 将词变成对应的编码
    input_word_ids = [tokenizer.convert_tokens_to_ids(word_list) for word_list in input_word_list]
    # 将label变成对应的编码
    lp = labelproducer(input_label_list)
    input_label_list = [lp.convert_label_to_ids(labels) for labels in input_label_list]
    # 将训练数据以及测试数据转换成tensor
    test_input_ids = torch.tensor(input_word_ids, dtype=torch.long)
    test_label = torch.tensor(input_label_list, dtype=torch.long)
    test_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    # 定义数据生成器（分批训练数据）
    test_data = TensorDataset(test_input_ids, test_attention_mask, test_label)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)
    # 定义模型
    print("load config from: {}".format(os.path.join(BERT_HOME, "bert_config.json")))
    config = BertConfig.from_pretrained(os.path.join(BERT_HOME, "bert_config.json"), num_labels=13)
    model = BertForTokenClassification.from_pretrained(BERT_HOME, config=config)
    model.to(device)
    model.load_state_dict(torch.load("model/pretrain_bert.pth"))
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    with torch.no_grad():
        pred_list = []
        label_list = []
        for step, batch in enumerate(test_dataloader):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)
            test_input_ids, test_attention_mask, label_ids = batch
            outputs = model(test_input_ids, attention_mask=test_attention_mask)
            logits = outputs[0]
            pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            label_ids = label_ids.detach().cpu().numpy().tolist()
            # 根据attention_mask添加结果
            for index_pred, pred in enumerate(pred_ids):
                pred_list_sentence = []
                label_list_sentence = []
                for index_p, p in enumerate(pred):
                    if test_attention_mask[index_pred][index_p] == 1:
                        pred_list_sentence.append(p)
                        label_list_sentence.append(label_ids[index_pred][index_p])
                pred_list.append(pred_list_sentence)
                label_list.append(label_list_sentence)
        # 拼接结果并预测
        input_word_label=lp.convert_ids_to_label(pred_list[0])

    result=translate(input_word_list_original,input_word_label)

    for i in range(len(result)):
        print("识别到实体 "+result[i]['word']+" 类型为"+result[i]['type'])

    i = 0
    entity_pair=[]
    while(i<len(result)):
        # 提取出所有企业对
        if(result[i]['type']=='company'):
            ii=i+1
            while(ii<len(result)):
                if(result[ii]['type']=='company'):
                    entity_pair.append([result[i]['word'],result[ii]['word']])
                ii+=1

        # 提取出所有企业-产品对
        if(result[i]['type']=='product'):
            ii = i+1
            while (ii < len(result)):
                if (result[ii]['type'] == 'company'):
                    entity_pair.append([result[ii]['word'], result[i]['word']])
                ii += 1

        # 提取出所有企业-有关部门对
        if (result[i]['type'] == 'department'):
            ii = i+1
            while (ii < len(result)):
                if (result[ii]['type'] == 'company'):
                    entity_pair.append([result[ii]['word'], result[i]['word']])
                ii += 1

        # 提取出所有企业-人物对
        if (result[i]['type'] == 'people'):
            ii = i+1
            while (ii < len(result)):
                if (result[ii]['type']== 'company'):
                    entity_pair.append([result[ii]['word'], result[i]['word']])
                ii += 1
        i+=1

    for i in range(len(entity_pair)):
        file = open("tmp/output_bert_"+str(i)+".txt", 'w')
        file.write(entity_pair[i][0]+"\t"+entity_pair[i][1]+"\t"+"管理"+"\t"+sentence)
        file.flush()

########################################################################################################################以上为bert部分
########################################################################################################################以下为LSTM部分
##该部分将bert的输出txt读入，处理成模型能够接受的pkl格式,并进行预测


    id2relation = {0:'合作',1:'竞争',2:'属于',3:'监管',4:'管理'}
    relation2id = {}
    with codecs.open('data/relation2id.txt', 'r', 'utf-8') as input_data:
        for line in input_data.readlines():
            relation2id[line.split()[0]] = int(line.split()[1])
        input_data.close()

    for i in range(len(entity_pair)):
        # 处理数据，将实体1、实体2与每个词的相对位置找出来，并把标签和句子本身存起来
        datas = deque()
        labels = deque()
        positionE1 = deque()
        positionE2 = deque()
        count = [0, 0, 0, 0, 0]  # 存储每个关系出现次数
        total_data = 0

        lines=open("tmp/output_bert_"+str(i)+".txt" , "r",encoding='utf-8').readlines()
        line = lines[0].strip().split()
        sentence = []
        index1 = line[3].index(line[0])  # 找出第一个实体在句子的位置
        position1 = []
        index2 = line[3].index(line[1])  # 找出第二个实体在句子的位置
        position2 = []

        for i, word in enumerate(line[3]):
            sentence.append(word)
            position1.append(i - index1)  # 找出每个词与第一个实体与第二个实体的相对距离
            position2.append(i - index2)
            i += 1
        datas.append(sentence)  # 训练数据加进去
        labels.append(relation2id[line[2]])  # 加标签
        positionE1.append(position1)  # 加实体1的相对位置
        positionE2.append(position2)  # 加实体2的相对位置

        # 这里是把所有的word转化为id存起来
        all_words = flatten(datas)
        sr_allwords = pd.Series(all_words)
        sr_allwords = sr_allwords.value_counts()

        set_words = sr_allwords.index
        set_ids = range(1, len(set_words) + 1)
        word2id = pd.Series(set_ids, index=set_words)
        id2word = pd.Series(set_words, index=set_ids)

        word2id["BLANK"] = len(word2id) + 1
        word2id["UNKNOW"] = len(word2id) + 1
        id2word[len(id2word) + 1] = "BLANK"
        id2word[len(id2word) + 1] = "UNKNOW"

        max_len = 50

        # 处理数据，把words转化为id形式并补全最大长度
        df_data = pd.DataFrame({'words': datas, 'tags': labels, 'positionE1': positionE1, 'positionE2': positionE2},
                               index=range(len(datas)))
        df_data['words'] = df_data['words'].apply(X_padding)
        df_data['tags'] = df_data['tags']
        df_data['positionE1'] = df_data['positionE1'].apply(position_padding)
        df_data['positionE2'] = df_data['positionE2'].apply(position_padding)

        datas = np.asarray(list(df_data['words'].values))
        labels = np.asarray(list(df_data['tags'].values))
        positionE1 = np.asarray(list(df_data['positionE1'].values))
        positionE2 = np.asarray(list(df_data['positionE2'].values))

        with open('tmp/entity_gingko_test.pkl', 'wb') as outp:
            pickle.dump(word2id, outp)
            pickle.dump(id2word, outp)
            pickle.dump(relation2id, outp)
            pickle.dump(datas, outp)
            pickle.dump(labels, outp)
            pickle.dump(positionE1, outp)
            pickle.dump(positionE2, outp)

        # 加载测试集  先用data里面的test.py将txt转化为pkl格式
        with open('tmp/entity_gingko_test.pkl', 'rb') as inp:
            word2id = pickle.load(inp)
            id2word = pickle.load(inp)
            relation2id = pickle.load(inp)
            test = pickle.load(inp)
            labels = pickle.load(inp)
            position1 = pickle.load(inp)
            position2 = pickle.load(inp)

        EMBEDDING_SIZE = len(word2id) + 1
        EMBEDDING_DIM = 100
        POS_SIZE = 82  # 不同数据集这里可能会报错。
        POS_DIM = 25
        HIDDEN_DIM = 200
        TAG_SIZE = len(relation2id)
        BATCH = 1
        EPOCHS = 100

        config = {}
        config['EMBEDDING_SIZE'] = EMBEDDING_SIZE
        config['EMBEDDING_DIM'] = EMBEDDING_DIM
        config['POS_SIZE'] = POS_SIZE
        config['POS_DIM'] = POS_DIM
        config['HIDDEN_DIM'] = HIDDEN_DIM
        config['TAG_SIZE'] = TAG_SIZE
        config['BATCH'] = BATCH
        config["pretrained"] = False

        learning_rate = 0.0005
        embedding_pre = []

        # 不使用预训练的词向量，下面使用随机初始化的vec
        model = BiLSTM_ATT(config, embedding_pre)
        model_tmp = torch.load('model/model_nre.pkl')

        new_list = ['word_embeds.weight', 'pos1_embeds.weight', 'pos2_embeds.weight', 'relation_embeds.weight',
                    'lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0',
                    'lstm.weight_ih_l0_reverse', 'lstm.weight_hh_l0_reverse', 'lstm.bias_ih_l0_reverse',
                    'lstm.bias_hh_l0_reverse', 'hidden2tag.weight', 'hidden2tag.bias']

        model.word_embeds.weight = model_tmp.word_embeds.weight
        model.pos1_embeds.weight = model_tmp.pos1_embeds.weight
        model.pos2_embeds.weight = model_tmp.pos2_embeds.weight
        model.relation_embeds.weight = model_tmp.relation_embeds.weight
        model.lstm.weight_ih_l0 = model_tmp.lstm.weight_ih_l0
        model.lstm.weight_hh_l0 = model_tmp.lstm.weight_hh_l0
        model.lstm.bias_ih_l0 = model_tmp.lstm.bias_ih_l0
        model.lstm.bias_hh_l0 = model_tmp.lstm.bias_hh_l0
        model.lstm.weight_ih_l0_reverse = model_tmp.lstm.weight_ih_l0_reverse
        model.lstm.weight_hh_l0_reverse = model_tmp.lstm.weight_hh_l0_reverse
        model.lstm.bias_ih_l0_reverse = model_tmp.lstm.bias_ih_l0_reverse
        model.lstm.bias_hh_l0_reverse = model_tmp.lstm.bias_hh_l0_reverse
        model.hidden2tag.weight = model_tmp.hidden2tag.weight
        model.hidden2tag.bias = model_tmp.hidden2tag.bias

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss(size_average=True)

        test = torch.LongTensor(test[:len(test) - len(test) % BATCH])
        position1 = torch.LongTensor(position1[:len(test) - len(test) % BATCH])
        position2 = torch.LongTensor(position2[:len(test) - len(test) % BATCH])
        labels = torch.LongTensor(labels[:len(test) - len(test) % BATCH])
        test_datasets = D.TensorDataset(test, position1, position2, labels)
        test_dataloader = D.DataLoader(test_datasets, BATCH, True, num_workers=0)

        # test
        acc_t = 0
        total_t = 0
        count_predict = [0, 0, 0, 0, 0]
        count_total = [0, 0, 0, 0, 0]
        count_right = [0, 0, 0, 0, 0]
        for sentence, pos1, pos2, tag in test_dataloader:
            sentence = Variable(sentence)
            pos1 = Variable(pos1)
            pos2 = Variable(pos2)
            y = model(sentence, pos1, pos2)

            tmp = y.data.numpy()
            tmp = tmp.tolist()
            prob = tmp[0]  # tmp是每种预测结果的可能性
            y = np.argmax(y.data.numpy(), axis=1)
            # re1 = list(map(tmp.index, heapq.nlargest(5, tmp)))
            yy=y.tolist()[0]
        if(prob[yy]>0.5):
            print(line[0]+" 与 "+line[1]+" 之间的关系为"+id2relation[yy]+"的概率为"+str(prob[yy]))

