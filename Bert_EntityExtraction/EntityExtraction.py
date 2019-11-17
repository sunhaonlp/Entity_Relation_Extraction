from transformers import BertTokenizer
from transformers import BertForTokenClassification
from transformers import BertConfig
from transformers import AdamW, WarmupLinearSchedule
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import os
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

from utils import get_input, produce_length, caculate_report

from tensorboardX import SummaryWriter

from sklearn.model_selection import KFold

# 定义可见的cuda设备
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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

    parser = argparse.ArgumentParser()
    # 增加参数
    # parser.add_argument("--train_file", default="./data/all_data_length_200.pkl", type=str)
    parser.add_argument("--val_split", default=0.2, type=float)
    parser.add_argument("--BERT_HOME", default="/data1/shgpu/sh/new/project/gingko/code/Bert_EntityExtraction/model/chinese_L-12_H-768_A-12/", type=str)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--seed", default=2019, type=int, help="随机模型初始化的随机种子")
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--test_batch_size", default=64, type=int)
    parser.add_argument("--do_train", action="store_false")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--max_len", default=256, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--model_name", default="bert-base-chinese", type=str)
    args = parser.parse_args()

    # tensorboardX
    writer = SummaryWriter()


    # 打印参数
    show_args(args)

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
    # 处理长度
    input_word_list, attention_mask = produce_length(input_word_list, args.max_len, "[PAD]", ret_attention_mask=True)
    input_label_list = produce_length(input_label_list, args.max_len, "O", ret_attention_mask=False)

    #for word_list in input_word_list:
    #    print(word_list)
    #    input()
    # 将词变成对应的编码
    input_word_ids = [tokenizer.convert_tokens_to_ids(word_list) for word_list in input_word_list]
    # 将label变成对应的编码
    lp = labelproducer(input_label_list)
    input_label_list = [lp.convert_label_to_ids(labels) for labels in input_label_list]

    # 划分训练集以及测试集
    train_input_ids, test_input_ids, train_label, test_label, train_attention_mask, test_attention_mask = train_test_split(input_word_ids, input_label_list, attention_mask, test_size=args.val_split, train_size=1-args.val_split, random_state=args.seed, shuffle=True)

    # for i in range(len(train_attention_mask)):
    #     if len(train_attention_mask[i]) !=256:
    #         print(i)
        # print(len(train_attention_mask[i]))

    # 将训练数据以及测试数据转换成tensor
    train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
    train_label = torch.tensor(train_label, dtype=torch.long)
    train_attention_mask = torch.tensor(train_attention_mask, dtype=torch.long)
    test_input_ids = torch.tensor(test_input_ids, dtype=torch.long)
    test_label = torch.tensor(test_label, dtype=torch.long)
    test_attention_mask = torch.tensor(test_attention_mask, dtype=torch.long)

    # 定义数据生成器（分批训练数据）
    train_data = TensorDataset(train_input_ids, train_attention_mask, train_label)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    test_data = TensorDataset(test_input_ids, test_attention_mask, test_label)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)

    # 定义模型
    print("load config from: {}".format(os.path.join(BERT_HOME, "bert_config.json")))
    config = BertConfig.from_pretrained(os.path.join(BERT_HOME, "bert_config.json"), num_labels=len(lp.target_to_ids))
    #print(config)
    #input()
    #config["num_labels"] = len(lp.target_to_ids)
    model = BertForTokenClassification.from_pretrained(BERT_HOME, config=config)
    model.to(device)
    # 打印模型
    # print(model)
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())

    # 定义优化器
    t_total = len(train_dataloader)*args.epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    model.load_state_dict(torch.load("model/model_epoch0.pkl"))

    # 多GPU
    if n_gpu>1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        for epoch in range(args.epochs):
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Train Batch Iteration")):
                # 梯度清零
                optimizer.zero_grad()
                if n_gpu==1:
                    # 多gpu会自动映射，单gpu需要手动映射
                    batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, label_ids = batch
                outputs = model(input_ids, attention_mask=attention_mask, labels=label_ids)
                loss = outputs[0]
                if n_gpu>1:
                    # 多gpu平均loss
                    loss = loss.mean()
                loss.backward()
                # 梯度裁剪，防止梯度爆炸或者梯度消失
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # 添加训练损失
                writer.add_scalar("training loss", loss.item(), epoch)

                optimizer.step()
                # 更新学习率
                scheduler.step()

            # 验证训练集，每个epoch验证一次
            model.eval()
            with torch.no_grad():
                pred_list = []
                label_list = []
                for step, batch in enumerate(tqdm(train_dataloader, desc="Eval Batch Iteration")):
                    if n_gpu==1:
                        batch = tuple(t.to(device) for t in batch)
                    test_input_ids, test_attention_mask, label_ids = batch
                    outputs = model(test_input_ids, attention_mask=test_attention_mask)
                    logits = outputs[0]

                    # pred_ids是预测得到的每个句子的标记，是一个双层list，里面的list有16个，代表batch_size
                    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()

                    # label_ids是每个句子的真实标记，是一个双层list，里面的list有16个，代表batch_size
                    label_ids = label_ids.detach().cpu().numpy().tolist()
                    # 根据attention_mask添加结果
                    # 这一部分是在截取有输入的部分，因为初始的时候会将不足512的长度用0补全，所以这里应该只把没补全部分的标记给爬下来
                    for index_pred, pred in enumerate(pred_ids):
                        pred_list_sentence = []
                        label_list_sentence = []
                        for index_p, p in enumerate(pred):
                            if test_attention_mask[index_pred][index_p]==1:
                                pred_list_sentence.append(p)
                                label_list_sentence.append(label_ids[index_pred][index_p])
                        pred_list.append(pred_list_sentence)
                        label_list.append(label_list_sentence)

                # 拼接结果并预测
                test_f1, test_p, test_r = caculate_report(label_list, pred_list, lp.convert_ids_to_label)

                # 添加验证f1, p, r
                writer.add_scalar("train f1", test_f1, epoch)
                writer.add_scalar("train p", test_p, epoch)
                writer.add_scalar("train r", test_r, epoch)

                print("training: F1_score: {}, Precision_score: {}, Recall_score: {}".format(test_f1, test_p, test_r))

            # 验证val，每个epoch验证一次
            model.eval()
            with torch.no_grad():
                pred_list = []
                label_list = []
                for step, batch in enumerate(tqdm(test_dataloader, desc="Eval Batch Iteration")):
                    if n_gpu==1:
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
                            if test_attention_mask[index_pred][index_p]==1:
                                pred_list_sentence.append(p)
                                label_list_sentence.append(label_ids[index_pred][index_p])
                        pred_list.append(pred_list_sentence)
                        label_list.append(label_list_sentence)
                # 拼接结果并预测
                test_f1, test_p, test_r = caculate_report(label_list, pred_list, lp.convert_ids_to_label)

                # 添加验证f1, p, r
                writer.add_scalar("val f1", test_f1, epoch)
                writer.add_scalar("val p", test_p, epoch)
                writer.add_scalar("val r", test_r, epoch)
                writer.close()

                print("testing: F1_score: {}, Precision_score: {}, Recall_score: {}".format(test_f1, test_p, test_r))
            print()
            if epoch % 20 == 0:
                model_name = "model/model_epoch" + str(epoch) + ".pkl"
                torch.save(model, model_name)
                print(model_name, "has been saved")
    torch.save(model.state_dict(),"model/pretrain_bert.pth")
