import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from BertSynForSequenceClassification import BertSynForSequenceClassification
class BERTSyn(nn.Module):


  def __init__(self):
    super(BERTSyn, self).__init__()
    self.bert_layer = BertSynForSequenceClassification.from_pretrained(
          "bert-base-uncased",
          num_labels = 3,  
          output_attentions = False,
          output_hidden_states = False,
      )


  def forward(self, input_ids, attention_mask, input_ids2,attention_mask2,token_type_ids=None, labels=None):
    return self.bert_layer(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, input_ids2=input_ids2,attention_mask2=attention_mask2,labels=labels, return_dict=True)



