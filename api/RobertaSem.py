import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import json
from transformers import RobertaForSequenceClassification

class RobertaSem(nn.Module):

  def __init__(self):
    super(RobertaSem, self).__init__()
    self.Roberta_layer = RobertaForSequenceClassification.from_pretrained(
          "roberta-base", 
          num_labels = 3,  
          output_attentions = False,
          output_hidden_states = False,
      )

  def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
    return self.Roberta_layer(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels, return_dict=True)
