import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import  XLNetForSequenceClassification


#Tell pytorch to run this model on the GPU.
class XLNetSem(nn.Module):

  def __init__(self):
    super(XLNetSem, self).__init__()
    self.xlnet_layer =  XLNetForSequenceClassification.from_pretrained(
          "xlnet-base-cased", 
          num_labels = 3,  
          output_attentions = False,
          output_hidden_states = False,
      )

  def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
    return self.xlnet_layer(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels, return_dict=True)
