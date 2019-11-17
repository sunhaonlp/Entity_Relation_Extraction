# coding:utf8
import numpy as np
import pickle
import sys
import codecs
import heapq
import torch.nn as nn
import torch.optim as optim
import torch
import torch.utils.data as D
from torch.autograd import Variable
from BiLSTM_ATT import BiLSTM_ATT



# 加载测试集  先用data里面的test.py将txt转化为pkl格式
with open('./data/processed_data/entity_gingko_test.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    relation2id = pickle.load(inp)
    train = pickle.load(inp)
    labels = pickle.load(inp)
    position1 = pickle.load(inp)
    position2 = pickle.load(inp)


print("train len", len(train))


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


#使用预训练的词向量进行训练
if len(sys.argv) == 2 and sys.argv[1] == "pretrained":
    print("use pretrained embedding")
    config["pretrained"] = True
    word2vec = {}
    with codecs.open('vec.txt', 'r', 'utf-8') as input_data:
        for line in input_data.readlines():
            word2vec[line.split()[0]] = map(eval, line.split()[1:])

    unknow_pre = []
    unknow_pre.extend([1] * 100)
    embedding_pre.append(unknow_pre)  # wordvec id 0
    for word in word2id:
        if word in word2vec:
            embedding_pre.append(word2vec[word])
        else:
            embedding_pre.append(unknow_pre)

    embedding_pre = np.asarray(embedding_pre)
    print(embedding_pre.shape)


# 若不使用预训练的词向量，则在下面会使用随机初始化的vec
model = BiLSTM_ATT(config, embedding_pre)
model_tmp =torch.load('model/model_epoch280.pkl')

for name, parameters in model.named_parameters():
    print(name, ':', parameters.size())
new_list = ['word_embeds.weight', 'pos1_embeds.weight', 'pos2_embeds.weight', 'relation_embeds.weight','lstm.weight_ih_l0','lstm.weight_hh_l0','lstm.bias_ih_l0','lstm.bias_hh_l0','lstm.weight_ih_l0_reverse','lstm.weight_hh_l0_reverse','lstm.bias_ih_l0_reverse','lstm.bias_hh_l0_reverse','hidden2tag.weight','hidden2tag.bias']


model.word_embeds.weight=model_tmp.word_embeds.weight
model.pos1_embeds.weight=model_tmp.pos1_embeds.weight
model.pos2_embeds.weight=model_tmp.pos2_embeds.weight
model.relation_embeds.weight=model_tmp.relation_embeds.weight
model.lstm.weight_ih_l0=model_tmp.lstm.weight_ih_l0
model.lstm.weight_hh_l0=model_tmp.lstm.weight_hh_l0
model.lstm.bias_ih_l0=model_tmp.lstm.bias_ih_l0
model.lstm.bias_hh_l0=model_tmp.lstm.bias_hh_l0
model.lstm.weight_ih_l0_reverse=model_tmp.lstm.weight_ih_l0_reverse
model.lstm.weight_hh_l0_reverse=model_tmp.lstm.weight_hh_l0_reverse
model.lstm.bias_ih_l0_reverse=model_tmp.lstm.bias_ih_l0_reverse
model.lstm.bias_hh_l0_reverse=model_tmp.lstm.bias_hh_l0_reverse
model.hidden2tag.weight=model_tmp.hidden2tag.weight
model.hidden2tag.bias=model_tmp.hidden2tag.bias

# dict_new = model.state_dict().copy()
#
# for i in range(len(new_list)):
#     dict_new[new_list[i]] = dict_trained[new_list[i]]


# model.load_state_dict(dict_new)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(size_average=True)


train = torch.LongTensor(train[:len(train) - len(train) % BATCH])
position1 = torch.LongTensor(position1[:len(train) - len(train) % BATCH])
position2 = torch.LongTensor(position2[:len(train) - len(train) % BATCH])
labels = torch.LongTensor(labels[:len(train) - len(train) % BATCH])
train_datasets = D.TensorDataset(train, position1, position2, labels)
train_dataloader = D.DataLoader(train_datasets, BATCH, True, num_workers=0)


# test
acc_t = 0
total_t = 0
count_predict = [0, 0, 0, 0, 0]
count_total = [0, 0, 0, 0, 0]
count_right = [0, 0, 0, 0, 0]
for sentence, pos1, pos2, tag in train_dataloader:
    sentence = Variable(sentence)
    pos1 = Variable(pos1)
    pos2 = Variable(pos2)
    y = model(sentence, pos1, pos2)


    tmp=y.data.numpy()
    tmp=tmp.tolist()
    tmp=tmp[0]  # tmp是每种预测结果的可能性
    y = np.argmax(y.data.numpy(), axis=1)
    re1 = list(map(tmp.index, heapq.nlargest(5, tmp)))

print(re1)
