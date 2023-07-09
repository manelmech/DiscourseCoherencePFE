
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import json
from transformers import DistilBertForSequenceClassification


class DistilBertSem(nn.Module):

  def __init__(self):
    super(DistilBertSem, self).__init__()
    self.DistilBert_layer = DistilBertForSequenceClassification.from_pretrained(
          "distilbert-base-uncased", 
          num_labels = 3,  
          output_attentions = False,
          output_hidden_states = False,
      )
  def forward(self, input_ids, attention_mask, labels=None):
    return self.DistilBert_layer(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
