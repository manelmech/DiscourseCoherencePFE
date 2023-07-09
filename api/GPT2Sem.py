
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import json
from transformers import GPT2ForSequenceClassification
from transformers import GPT2Config
class GPT2Sem(nn.Module):

  def __init__(self,len):
    super(GPT2Sem, self).__init__()
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=3)
    self.gpt_layer = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='gpt2', config=model_config)
    self.gpt_layer.resize_token_embeddings(len) 
    self.gpt_layer.config.pad_token_id = self.gpt_layer.config.eos_token_id
  def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
    return self.gpt_layer(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels, return_dict=True)

     