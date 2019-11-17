from transformers import BertForTokenClassification
from transformers import BertModel
import torch
import torch.nn as nn
import torchcrf


class BertForTokenClassification_CRF(torch.nn.Module):
    def __init__(self, Bert_Home, bert_config, num_tags):
        super(BertForTokenClassification_CRF, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(Bert_Home, config=bert_config)
        self.crf = torchcrf.CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        loss = self.crf(logits, labels, mask=attention_mask.byte(), reduction="mean")
        return -loss

    def decode(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        decoded_seq = self.crf.decode(logits)
        return decoded_seq

class Bert_Lstm_CRF(torch.nn.Module):
    def __init__(self, Bert_Home, bert_config, num_tags, lstm_hidden):
        super(Bert_Lstm_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(Bert_Home, config=bert_config)
        self.lstm = nn.LSTM(bert_config.hidden_size, lstm_hidden, batch_first=True, bidirectional=True)
        self.clf = nn.Linear(lstm_hidden*2, num_tags)
        self.crf = torchcrf.CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        logits, _ = self.lstm(logits)
        logits = self.clf(logits)
        loss = self.crf(logits, labels, mask=attention_mask.byte(), reduction="mean")
        return -loss

    def decode(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        logits, _ = self.lstm(logits)
        logits = self.clf(logits)
        decoded_seq = self.crf.decode(logits)
        return decoded_seq
