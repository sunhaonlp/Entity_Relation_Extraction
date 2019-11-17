import pickle as pkl
import os
import json
from seqeval.metrics import f1_score, precision_score, recall_score

transtab = str.maketrans("","","##")
def translate_string(string):
    """
    如果字符串中有两个##连在一起，则删掉这两个##
    """
    global transtab
    return string.translate(transtab)

def make_label(en_start, en_end, en_type, label_list):
    """
    根据开始位置和结束位置打标签
    BIOES
    """
    if en_end-en_start==1:
        label_list[en_start] = "S-"+en_type
    else:
        for index in range(en_start, en_end):
            if index==en_start:
                label_list[index] = "B-"+en_type
            elif index==en_end-1:
                label_list[index] = "E-"+en_type
            else:
                label_list[index] = "I-"+en_type
#
# def get_input(filepath, debug=True):
#     """
#     得到Bert模型的输入
#     """
#     with open(filepath, "rb") as f:
#         datas = pkl.load(f)
#
#     input_word_list = []
#     input_label_list = []
#     for data in datas:
#         bert_words = data["bert_words"]
#         label_list = ["O" for _ in bert_words]      # 首先制作全O的标签
#         for entity in data["golden-entity-mentions"]:
#             en_start = entity["start"]
#             en_end = entity["end"]
#             en_type = entity["entity-type"]
#             # 验证entity["text"]是否与bert_words能够对应
#             #if debug:
#             #    bert_words_text = []
#             #    for index in range(en_start, en_end):
#             #        bert_words_text.append(translate_string(bert_words[index]))
#             #    en_text = entity["text"].lower()    # 变成小写
#             #    if "".join(bert_words_text)!=en_text:
#             #        # 如果它们匹配不上
#             #        for index in range(len(bert_words_text)):
#             #            # 如果匹配不上的不是[UNK]
#             #            if bert_words_text[index]!="[UNK]" and en_text[index]!=bert_words_text[index]:
#             #                raise ValueError("bert_words_text: {} not equal entity['text']: {}".format(bert_words_text, en_text))
#             # 根据开始与结束位置打标签
#             make_label(en_start, en_end, en_type, label_list)
#         input_word_list.append(["[CLS]"]+bert_words+["[SEP]"])
#         input_label_list.append(["O"]+label_list+["O"])
#     return input_word_list, input_label_list


def get_input(debug=True):
    """
    得到Bert模型的输入
    """
#     with open(filepath, "rb") as f:
#         datas = pkl.load(f)
    files = os.listdir('/data1/shgpu/sh/new/project/gingko/data/label_data/entity')
    input_word_list = []
    input_label_list = []
    for file in files:
        with open("/data1/shgpu/sh/new/project/gingko/data/label_data/entity/"+file, 'r', encoding='UTF-8') as f:
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
        # print(input_word_list)
        #         # print(input_label_list)
        #         # print()
    return input_word_list, input_label_list

def produce_length(sequences, max_len, padding, ret_attention_mask):
    """
    处理长度，短则填充，长则截断，padding是填充的符号
    """
    if ret_attention_mask:
        attention_mask = [[1 for _ in range(len(sequence))] for sequence in sequences]  # 原始长度填充1
    for index, sequence in enumerate(sequences):
        while len(sequence)<max_len:
            sequence.append(padding)
            if ret_attention_mask:
                attention_mask[index].append(0)                                         # pad的部分填充0
        sequences[index] = sequence[:max_len]
        if ret_attention_mask:
            attention_mask[index] = attention_mask[index][:max_len]
    if ret_attention_mask:
        return sequences, attention_mask
    return sequences

def caculate_report(y_true, y_pred, transform_func):
    """
    计算预测的分数
    """
    for i in range(len(y_true)):
        y_true[i] = transform_func(y_true[i])
    for i in range(len(y_pred)):
        y_pred[i] = transform_func(y_pred[i])
    return f1_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)


if __name__=="__main__":

    filepath = "/data0/dlw/sunrui_joint_ee/datasets/xujin_law_v2/bert_data/all_data_length_300.pkl"

    input_word_list, input_label_list = get_input(filepath)
