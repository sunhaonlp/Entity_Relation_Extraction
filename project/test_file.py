from transformers import BertTokenizer
from transformers import BertForTokenClassification
from transformers import BertConfig
from transformers import AdamW, WarmupLinearSchedule
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json
import os
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import codecs
from utils import  produce_length, make_label


# 定义可见的cuda设备
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
    with open("tmp/input.json", 'r', encoding='UTF-8') as f:
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

    # 数据读入部分
    sentence = input("请输入你想预测的句子：")
    dict_bert = {}
    dict_bert['entity-mentions'] = []
    dict_bert['sentence'] = sentence
    final_json = json.dumps(dict_bert, indent=4, ensure_ascii=False)
    with codecs.open("tmp/input.json", 'w', 'utf-8') as file:
        file.write(final_json)

    # 模型部分
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_split", default=0.5, type=float)
    parser.add_argument("--BERT_HOME", default="/data1/shgpu/sh/new/project/gingko/code/Bert_EntityExtraction/model/chinese_L-12_H-768_A-12/", type=str)
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
    model.load_state_dict(torch.load("./model/pretrain_bert.pth"))

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
        # test_f1, test_p, test_r = caculate_report(label_list, pred_list, lp.convert_ids_to_label)


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
        file = open("/data1/shgpu/sh/new/project/gingko/code/ChineseNRE/data/origin_data/test_"+str(i)+".txt", 'w')
        file.write(entity_pair[i][0]+"\t"+entity_pair[i][1]+"\t"+"管理"+"\t"+sentence)
        file.flush()

    print(entity_pair)