from asyncio.log import logger
from lib2to3.pgen2 import token  
from xmlrpc.client import Boolean
from sqlalchemy import false, true
from LSTMSemRel import LSTMSemRel
from LSTMSentAvg import LSTMSentAvg
from LSTMParSeq import LSTMParSeq
from CNNPosTag import CNNPosTag
from BERTSem import BERTSem
from FusionSemSyn import FusionSemSyn
from transformers import BertTokenizer
from RobertaSem import RobertaSem
from DistilBertSem import DistilBertSem
from XLNetSem import XLNetSem
from GPT2Sem import GPT2Sem
from BertSyn import BERTSyn


import torch.nn.functional as F
import uvicorn
import pickle
import random
import shutil
import numpy
import spacy 
from pydantic import BaseModel
from fastapi import FastAPI, Form, UploadFile, File, Depends, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from DocumentWithParagraphs import DocumentWithParagraphs
from evaluation import eval_docs
from train_neural_models import train
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from data_loader import *
import sys
import json
from json import JSONEncoder
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd
from decimal import Decimal
from tempfile import NamedTemporaryFile
import csv
from typing import Optional, List
from database import SessionLocal
from database import DATABASE_URL
from schema import Admin as SchemaAdmin
from schema import Model as SchemaModel
from schema import Inputlist as Inputlist
from schema import Inputs as Inputs
from schema import  Filelist as Filelist
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


from transformers import BertTokenizer, XLNetTokenizer, DistilBertTokenizer,RobertaTokenizer,GPT2Tokenizer

from schema import Token
from schema import TokenData
from models import Admin as ModelAdmin
from models import Model as ModelModel
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
app = FastAPI()





#BERT tokenizer
berttokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#XLNet tokenizer
xlnettokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
distilberttokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
robertatokenizer = RobertaTokenizer.from_pretrained("roberta-base")
gpt2tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
gpt2tokenizer.padding_side = "left"
gpt2tokenizer.pad_token = gpt2tokenizer.eos_token
nlp = spacy.load("en_core_web_sm")


# db=SessionLocal()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "https://56bf-34-142-171-229.ngrok.io/"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#sys.stdout = open('test.txt', 'w')


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyArrayEncoder, self).default(obj)


# load the dictionary
embeddings = pickle.load(open('./pickle_files/word_embeds.pkl', 'rb'))
best_weights = pickle.load(open('./pickle_files/best_weights.pkl', 'rb'))
#best_weights_fusion = pickle.load(open('./pickle_files/best_weights_fusion.pkl', 'rb')) #best_weights_fusion.pkl

params = {
    'vector_type': 'glove'
}

dataObj = Data(params)
word_to_idx = pickle.load(open('./pickle_files/word_to_idx.pkl', 'rb'))
idx_to_word = pickle.load(open('./pickle_files/idx_to_word.pkl', 'rb'))
dataObj.word_embeds = embeddings
dataObj.word_to_idx = word_to_idx
dataObj.idx_to_word = idx_to_word

#BERT tokenizer

def convert_csv(bytes):
    data = {}
    file_copy = NamedTemporaryFile(delete=False)
    try:
        with file_copy as f:
            f.write(bytes)

        with open(file_copy.name, 'r', encoding='utf-8') as csvf:
            csvReader = csv.DictReader(csvf)
            i = 0
            for rows in csvReader:
                key = i
                data[key] = rows
                i = i+1
    finally:
        file_copy.close()
        os.unlink(file_copy.name)
    return data


def generate_tags(text):
    stop = set(stopwords.words('english') + list(string.punctuation))
    text_tags = sent_tokenize(text)
    seq_tag = []
    for sentence in text_tags:
        sent_seq = ''
        b = []
        i = word_tokenize(sentence)
        for j in i:
            if j not in stop:
                b.append(j)
        i = b
        word_tag = nltk.pos_tag(i)

        for word in word_tag:
            sent_seq = sent_seq + word[1] + ' '
        seq_tag.append(sent_seq)
    print(seq_tag)
    tagged_text = ''
    length = len(seq_tag)
    i = 1
    for sent in seq_tag:
        if(i == length):
            tagged_text = tagged_text + sent
        else:
            tagged_text = tagged_text + sent
        i = i+1
    return tagged_text


def preprocess_data_sentavg(text):
    # read data class
    documents = []
    add_new_words = True
    text = text.lower()
    text_id = random.randint(0, 1000)
    label = None
    doc = DocumentWithParagraphs(text, label, id=text_id)
    doc_indexed = []
    for para in doc.text:
        para_indexed = []
        for sent in para:
            sent_indexed = []
            for word in sent:
                sent_indexed.append(
                    dataObj.add_token_to_index(word, add_new_words))
            para_indexed.append(sent_indexed)
        doc_indexed.append(para_indexed)
    doc.text_indexed = doc_indexed
    documents.append(doc)
    documents_data, documents_labels, documents_ids = dataObj.create_doc_sents(
        documents, 'sentence', 'class')
    indices = [int('0')]
    sentences, orig_batch_labels = dataObj.get_batch(
        documents_data, documents_labels, indices, 'sent_avg')
    batch_padded, batch_lengths, original_index = dataObj.pad_to_batch(
        sentences, dataObj.word_to_idx, 'sent_avg')
    return batch_padded, batch_lengths, original_index


def preprocess_data_parseq(text):
    # read data class
    documents = []
    add_new_words = True
    text = text.lower()
    text_id = random.randint(0, 1000)
    label = None
    doc = DocumentWithParagraphs(text, label, id=text_id)
    doc_indexed = []
    for para in doc.text:
        para_indexed = []
        for sent in para:
            sent_indexed = []
            for word in sent:
                sent_indexed.append(
                    dataObj.add_token_to_index(word, add_new_words))
            para_indexed.append(sent_indexed)
        doc_indexed.append(para_indexed)
    doc.text_indexed = doc_indexed
    documents.append(doc)
    documents_data, documents_labels, documents_ids = dataObj.create_doc_sents(
        documents, 'paragraph', 'class')
    indices = [int('0')]
    sentences, orig_batch_labels = dataObj.get_batch(
        documents_data, documents_labels, indices, 'par_seq')
    batch_padded, batch_lengths, original_index = dataObj.pad_to_batch(
        sentences, dataObj.word_to_idx, 'par_seq')
    return batch_padded, batch_lengths, original_index


def preprocess_data_cnnpostag(text):
    # read data class
    documents = []
    add_new_words = True
    text = text.lower()
    text = generate_tags(text)
    text_id = random.randint(0, 1000)
    label = None
    doc = DocumentWithParagraphs(text, label, id=text_id)
    doc_indexed = []
    for para in doc.text:
        para_indexed = []
        for sent in para:
            sent_indexed = []
            for word in sent:
                sent_indexed.append(
                    dataObj.add_token_to_index(word, add_new_words))
            para_indexed.append(sent_indexed)
        doc_indexed.append(para_indexed)
    doc.text_indexed = doc_indexed
    documents.append(doc)
    documents_data, documents_labels, documents_ids = dataObj.create_doc_sents(
        documents, 'paragraph', 'class')
    indices = [int('0')]
    sentences, orig_batch_labels = dataObj.get_batch(
        documents_data, documents_labels, indices, 'cnn_pos_tag')
    batch_padded, batch_lengths, original_index = dataObj.pad_to_batch(
        sentences, dataObj.word_to_idx, 'cnn_pos_tag')
    return batch_padded, batch_lengths, original_index


def preprocess_data_transformer(text,type_transformer):
    input_ids = []
    attention_masks = []
    if type_transformer=="bert":
      tokenizer = berttokenizer
    if type_transformer=="xlnet":
      tokenizer = xlnettokenizer
    if type_transformer =="distilbert":
       tokenizer = distilberttokenizer
    if type_transformer =="roberta":  
       tokenizer= robertatokenizer
    if type_transformer =="gpt2":  
       tokenizer= gpt2tokenizer
    encoded_dict = tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 256,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
   
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
   
    sample = TensorDataset(input_ids, attention_masks)
    sample_loader = DataLoader(
            sample,
            sampler = RandomSampler(sample),
            batch_size = 1
        )
    return sample_loader

def preprocess_data_transformer_syntax(text):
    input_ids = []
    attention_masks = []
    input_ids2 = []
    attention_masks2 = []
    tokenizer = berttokenizer
    tagging = nlp(text)
    pos_tags = [token.pos_ for token in tagging]
    encoded_dict = tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 256,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    encoded_dict2 = tokenizer.encode_plus(
                                ' '.join(pos_tags),        # List of part of speech taggings to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 256,           # Pad & truncate .
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                          )
         
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    input_ids2.append(encoded_dict2['input_ids'])
    attention_masks2.append(encoded_dict2['attention_mask'])
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    input_ids2 = torch.cat(input_ids2, dim=0)
    attention_masks2 = torch.cat(attention_masks2, dim=0)
    sample = TensorDataset(input_ids, attention_masks,input_ids2, attention_masks2)
    sample_loader = DataLoader(
            sample,
            sampler = RandomSampler(sample),
            batch_size = 1
        )
    return sample_loader
    
def convert_json(bytes):
    data = {}
    file_copy = NamedTemporaryFile(delete=False)
    try:
        with file_copy as f:
            f.write(bytes)


        with open(file_copy.name, 'r') as jsonf:
            json_data = json.load(jsonf)
           
    finally:
        file_copy.close()
        os.unlink(file_copy.name)
    return json_data

        
# Setting up the home route


@app.get("/")
def read_root():
    return {"data": "Welcome to La Coherencia"}


# Uploader le pickle file d'un nouveau modèle
@app.post('/addpickle_model')
async def pickle(pickle: UploadFile = File(...)):
    with open("pickle_files/"+pickle.filename, "wb") as buffer:
        shutil.copyfileobj(pickle.file, buffer)
    return {"filename": pickle.filename}






@app.post("/evaluatecomp")
async def get_predict(data: Inputlist, db: Session = Depends(get_db)):
    
    modelnames = data.modelList
    sample = data.text
 
    scores = []
    texts = []
    
    
    i=0
    for f in modelnames:
        
        
        texts.append(sample)
        
        if f == "sent_avg":
            model = torch.load('./pickle_files/sent_avg.pt')
            model.eval()
            batch_padded, batch_lengths, original_index = preprocess_data_sentavg(
                sample)
            pred= model.forward(
                batch_padded, batch_lengths, original_index, dim=1)
            argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)
    
        elif f == "par_seq":
            model = torch.load('./pickle_files/par_seq.pt')
            model.eval()
            batch_padded, batch_lengths, original_index = preprocess_data_parseq(
                sample)
            pred, avg_deg = model.forward(
                batch_padded, batch_lengths, original_index, dim=1)
            argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)
        elif f == "sem_rel":
            model = torch.load(
                './pickle_files/sem_rel.pt')
            model.eval()
            batch_padded, batch_lengths, original_index = preprocess_data_parseq(
                sample)
            pred = model.forward(batch_padded, batch_lengths,
                                 original_index, weights=best_weights, dim=1)
            argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)
        elif f == "cnn_postag":
            model = torch.load(
                './pickle_files/cnn_postag.pt')
            model.eval()
            batch_padded, batch_lengths, original_index = preprocess_data_cnnpostag(
                sample)
            pred = model.forward(
                batch_padded, batch_lengths, original_index, dim=1)
            argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)
        elif f == "sem_syn":
            model = pickle.load(open('./pickle_files/sem_syn.pt', 'rb'))
            model.eval()
            batch_padded_cnn, batch_lengths_cnn, original_index_cnn = preprocess_data_cnnpostag(
                sample)
            batch_padded_semrel, batch_lengths_semrel, original_index_semrel = preprocess_data_parseq(
                sample)
            pred = model.forward(batch_padded_semrel, batch_padded_cnn, batch_lengths_semrel,
                                 batch_lengths_cnn, original_index, weights=best_weights, dim=1)
            argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)
        elif f == "par_seq":
                model = torch.load('./pickle_files/par_seq.pt')
                model.eval()
                batch_padded, batch_lengths, original_index = preprocess_data_parseq(
                    sample)
                pred, avg_deg = model.forward(
                    batch_padded, batch_lengths, original_index, dim=1)
                argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)
        elif f == "sem_rel":
                model = torch.load(
                    './pickle_files/sem_rel.pt')
                model.eval()
                batch_padded, batch_lengths, original_index = preprocess_data_parseq(
                    sample)
                pred = model.forward(batch_padded, batch_lengths,
                                    original_index, weights=best_weights, dim=1)
                argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)
        elif f == "cnn_postag":
                model = torch.load(
                    './pickle_files/cnn_postag.pt')
                model.eval()
                batch_padded, batch_lengths, original_index = preprocess_data_cnnpostag(
                    sample)
                pred = model.forward(
                    batch_padded, batch_lengths, original_index, dim=1)
                argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)
        elif f == "sem_syn":
                model = pickle.load(open('./pickle_files/sem_syn.pt', 'rb'))
                model.eval()
                batch_padded_cnn, batch_lengths_cnn, original_index_cnn = preprocess_data_cnnpostag(
                    sample)
                batch_padded_semrel, batch_lengths_semrel, original_index_semrel = preprocess_data_parseq(
                    sample)
                pred = model.forward(batch_padded_semrel, batch_padded_cnn, batch_lengths_semrel,
                                    batch_lengths_cnn, original_index, weights=best_weights, dim=1)
                argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)
        elif f == "RobertaSem":
                model = RobertaSem()
       
                model.load_state_dict(torch.load(open('./pickle_files/RobertaSem.pt', 'rb'), map_location=torch.device('cpu')))
       
       
                model.eval()
                sample_loader = preprocess_data_transformer(sample,"roberta")
                for s in sample_loader:
                    sample1 = tuple(t for t in s)
                b_input_ids, b_input_mask = sample1
                pred = model.forward(input_ids=b_input_ids,attention_mask=b_input_mask)
                result = F.softmax(pred.logits, dim=1)
                result = result.cpu().data.numpy()
                argmax = list(np.argmax(result, axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)  
        elif f == "XLNETSem":
                model = XLNetSem()
       
                model.load_state_dict(torch.load(open('./pickle_files/XLNetSem.pt', 'rb'), map_location=torch.device('cpu')))
       
                model.eval()
                sample_loader = preprocess_data_transformer(sample,"xlnet")
                for s in sample_loader:
                    sample1 = tuple(t for t in s)
                b_input_ids, b_input_mask = sample1
                pred = model.forward(input_ids=b_input_ids,attention_mask=b_input_mask)
                result = F.softmax(pred.logits, dim=1)
                result = result.cpu().data.numpy()
                argmax = list(np.argmax(result, axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)  
        elif f == "DistilBertSem":
                model = DistilBertSem()
       
                model.load_state_dict(torch.load(open('./pickle_files/DistilBertSem.pt', 'rb'), map_location=torch.device('cpu')))
       
       
                model.eval()
                sample_loader = preprocess_data_transformer(sample,"distilbert")
                for s in sample_loader:
                    sample1 = tuple(t for t in s)
                b_input_ids, b_input_mask = sample1
                pred = model.forward(input_ids=b_input_ids,attention_mask=b_input_mask)
                result = F.softmax(pred.logits, dim=1)
                result = result.cpu().data.numpy()
                argmax = list(np.argmax(result, axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)  
        elif f== "BERTSem":
                model = BERTSem()
       
                model.load_state_dict(torch.load(open('./pickle_files/BERTSem-1.pt', 'rb'), map_location=torch.device('cpu')))
       
       
                model.eval()
                sample_loader = preprocess_data_transformer(sample,"bert")
                for s in sample_loader:
                    sample1 = tuple(t for t in s)
                b_input_ids, b_input_mask = sample1
                pred = model.forward(input_ids=b_input_ids,attention_mask=b_input_mask)
                result = F.softmax(pred.logits, dim=1)
                result = result.cpu().data.numpy()
                argmax = list(np.argmax(result, axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)  
        elif f == "GPT2Sem":
                model = GPT2Sem(len(gpt2tokenizer))
       
                model.load_state_dict(torch.load(open('./pickle_files/gptSem.pt', 'rb'), map_location=torch.device('cpu')))
       
       
                model.eval()
                sample_loader = preprocess_data_transformer(sample,"gpt2")          
                for s in sample_loader:
                    sample1 = tuple(t for t in s)
                b_input_ids, b_input_mask = sample1
                pred = model.forward(input_ids=b_input_ids,attention_mask=b_input_mask)
                result = F.softmax(pred.logits, dim=1)
                result = result.cpu().data.numpy()
                argmax = list(np.argmax(result, axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)  
        elif f == "BERTSyn2_concat":  # Bert
        
            model = BERTSyn()        
            model.load_state_dict(torch.load(open('./pickle_files/BERTSyn.pt', 'rb'), map_location=torch.device('cpu')))
            model.eval()
            sample_loader = preprocess_data_transformer_syntax(sample)
            for s in sample_loader: 
                sample1 = tuple(t for t in s)
            b_input_ids, b_input_mask ,b_input_ids2, b_input_mask2= sample1
            pred = model.forward(input_ids=b_input_ids,attention_mask=b_input_mask,input_ids2=b_input_ids2,attention_mask2=b_input_mask2)
            result = F.softmax(pred.logits, dim=1)
            result = result.cpu().data.numpy()
            argmax = list(np.argmax(result, axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)  
                
    
            
    print(scores)
    return {"data":  {"scores": scores, "modelnames": modelnames, "texts": texts}}



@app.post("/evaluate")
async def get_predict(data: Inputs, db: Session = Depends(get_db)):
    
    model_id = data.selectedIndex
    sample = data.text
    model_db = get_one_model(model_id, db)
  

    print(model_db.saved_model_pickle)
    if model_db.saved_model_pickle == "sent_avg.pt":
        model = torch.load(
            './pickle_files/sent_avg.pt')
        model.eval()
        batch_padded, batch_lengths, original_index = preprocess_data_sentavg(
            sample)
        
        pred, score_pred = model.forward(
            batch_padded, batch_lengths, original_index, dim=1)
        #print(score_pred)
        argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
        score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
        
        #reg_score = list(score_pred.cpu().data.numpy())
        #print(reg_score)
        # round(Decimal(0.3223322), 2)
        #score_reg = round(Decimal(json.dumps(abs(reg_score[0][0]), cls=NumpyArrayEncoder)),2)
        #print(score_reg)

    elif model_db.saved_model_pickle == "par_seq.pt":
        model = torch.load('./pickle_files/par_seq.pt')
        model.eval()
        batch_padded, batch_lengths, original_index = preprocess_data_parseq(
            sample)
        pred, avg_deg = model.forward(
            batch_padded, batch_lengths, original_index, dim=1)
        argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
        score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
    elif model_db.saved_model_pickle == "sem_rel.pt":
        model = torch.load('./pickle_files/sem_rel.pt')
        model.eval()
        batch_padded, batch_lengths, original_index = preprocess_data_parseq(
            sample)
        pred = model.forward(batch_padded, batch_lengths,
                             original_index, weights=best_weights, dim=1)
        print(pred)
        argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
        score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
    elif model_db.saved_model_pickle == "cnn_postag.pt":
        model = torch.load(
            './pickle_files/cnn_postag.pt')
        model.eval()
        batch_padded, batch_lengths, original_index = preprocess_data_cnnpostag(
            sample)
        pred = model.forward(batch_padded, 
                             
                             batch_lengths,
                             original_index, dim=1)
                   
        argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
        score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
    elif model_db.saved_model_pickle == "sem_syn.pt":
        model = torch.load('./pickle_files/sem_syn.pt')
        model.eval()
        batch_padded_cnn, batch_lengths_cnn, original_index_cnn = preprocess_data_cnnpostag(
            sample)
        batch_padded_semrel, batch_lengths_semrel, original_index_semrel = preprocess_data_parseq(
            sample)
        pred = model.forward(batch_padded_semrel, batch_padded_cnn, batch_lengths_semrel,
                             batch_lengths_cnn, original_index_semrel, weights=best_weights_fusion, dim=1)
        print(pred)
        argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
        score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
    else:  # Bert 
        model=BERTSem()
        #model = torch.load(open('./pickle_files/BERT.pt', 'rb'), map_location= torch.device('cpu'))

        model.load_state_dict( torch.load(open('./pickle_files/BERTSemcv.pt', 'rb'), map_location= torch.device('cpu')) )     
        print(torch.load(open('./pickle_files/BERTSemcv.pt', 'rb'), map_location= torch.device('cpu')) )
        model.eval()
        sample_loader = preprocess_data_bert(sample)
        
        for s in sample_loader: 
            sample = tuple( t for t in s)

        print(sample)
       
        b_input_ids, b_input_mask = sample
        print(b_input_mask)
        
        pred = model.forward(b_input_ids=b_input_ids, b_input_mask=b_input_mask)
        result = F.softmax(pred.logits, dim=1)
        result = result.data.numpy()
        print(result)
        argmax = list(np.argmax(result, axis=1))
        score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
        
    return {
        "data": {
            'score': score
        }
    }



@app.post("/files")
async def get_files(modelList: Filelist):
      
    app.state=modelList.modelList 
    
    return(1)

@app.post("/uploadfilecomp")
async def get_predict_filecomp(data:list[UploadFile], modelnames:list[str]=Form(...), csv:bool=Form(...)):
     
  
    print(modelnames)
    print("manel")
    print(data)
    scores = []
    texts = []
    mnames =[]
    original_scores = []
    text_ids = []

    if(csv):

        content_assignment = await data[0].read()
        files = convert_csv(content_assignment)
    else:
        files = data
    

    for i in range(len(files)):
        if not csv:
          file = await files[i].read()
          sample = file.decode('ISO-8859-1')
      
        for m in modelnames:
            

            if(csv):
                sample = files[i]['text']
                text_ids.append(files[i]['text_id'])
                texts.append(files[i]['text'])
                original_scores.append(files[i]['labelA'])
            else:
                print(m)
                texts.append(sample)
           
            mnames.append(m)

            if m == "sent_avg":
                model = torch.load('./pickle_files/sent_avg.pt')
                model.eval()
                batch_padded, batch_lengths, original_index = preprocess_data_sentavg(
                    sample)
                pred, avg_deg, score_pred = model.forward(
                    batch_padded, batch_lengths, original_index, dim=1)
                argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)
                
              

            elif m == "par_seq":
                model = torch.load('./pickle_files/par_seq.pt')
                model.eval()
                batch_padded, batch_lengths, original_index = preprocess_data_parseq(
                    sample)
                pred, avg_deg = model.forward(
                    batch_padded, batch_lengths, original_index, dim=1)
                argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)
            elif m == "sem_rel":
                model = torch.load(
                    './pickle_files/sem_rel.pt')
                model.eval()
                batch_padded, batch_lengths, original_index = preprocess_data_parseq(
                    sample)
                pred = model.forward(batch_padded, batch_lengths,
                                    original_index, weights=best_weights, dim=1)
                argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)
            elif m == "cnn_postag":
                model = torch.load(
                    './pickle_files/cnn_postag.pt')
                model.eval()
                batch_padded, batch_lengths, original_index = preprocess_data_cnnpostag(
                    sample)
                pred = model.forward(
                    batch_padded, batch_lengths, original_index, dim=1)
                argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)
            elif m == "sem_syn":
                model = pickle.load(open('./pickle_files/sem_syn.pt', 'rb'))
                model.eval()
                batch_padded_cnn, batch_lengths_cnn, original_index_cnn = preprocess_data_cnnpostag(
                    sample)
                batch_padded_semrel, batch_lengths_semrel, original_index_semrel = preprocess_data_parseq(
                    sample)
                pred = model.forward(batch_padded_semrel, batch_padded_cnn, batch_lengths_semrel,
                                    batch_lengths_cnn, original_index, weights=best_weights, dim=1)
                argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)
            elif m == "RobertaSem":
                model = RobertaSem()
       
                model.load_state_dict(torch.load(open('./pickle_files/RobertaSem.pt', 'rb'), map_location=torch.device('cpu')))
       
       
                model.eval()
                sample_loader = preprocess_data_transformer(sample,"roberta")
                for s in sample_loader:
                    sample1 = tuple(t for t in s)
                b_input_ids, b_input_mask = sample1
                pred = model.forward(input_ids=b_input_ids,attention_mask=b_input_mask)
                result = F.softmax(pred.logits, dim=1)
                result = result.cpu().data.numpy()
                argmax = list(np.argmax(result, axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)  
            elif m == "XLNETSem":
                model = XLNetSem()
       
                model.load_state_dict(torch.load(open('./pickle_files/XLNetSem.pt', 'rb'), map_location=torch.device('cpu')))
       
       
                model.eval()
                sample_loader = preprocess_data_transformer(sample,"xlnet")
                for s in sample_loader:
                    sample1 = tuple(t for t in s)
                b_input_ids, b_input_mask = sample1
                pred = model.forward(input_ids=b_input_ids,attention_mask=b_input_mask)
                result = F.softmax(pred.logits, dim=1)
                result = result.cpu().data.numpy()
                argmax = list(np.argmax(result, axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)  
            elif m == "DistilBertSem":
                model = DistilBertSem()
       
                model.load_state_dict(torch.load(open('./pickle_files/DistilBertSem.pt', 'rb'), map_location=torch.device('cpu')))
       
       
                model.eval()
                sample_loader = preprocess_data_transformer(sample,"distilbert")
                for s in sample_loader:
                    sample1 = tuple(t for t in s)
                b_input_ids, b_input_mask = sample1
                pred = model.forward(input_ids=b_input_ids,attention_mask=b_input_mask)
                result = F.softmax(pred.logits, dim=1)
                result = result.cpu().data.numpy()
                argmax = list(np.argmax(result, axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)  
            elif m == "BERTSem":
                model = BERTSem()
       
                model.load_state_dict(torch.load(open('./pickle_files/BERTSem-1.pt', 'rb'), map_location=torch.device('cpu')))
       
       
                model.eval()
                sample_loader = preprocess_data_transformer(sample,"bert")
                for s in sample_loader:
                    sample1 = tuple(t for t in s)
                b_input_ids, b_input_mask = sample1
                pred = model.forward(input_ids=b_input_ids,attention_mask=b_input_mask)
                result = F.softmax(pred.logits, dim=1)
                result = result.cpu().data.numpy()
                argmax = list(np.argmax(result, axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)  
            elif m == "GPT2Sem":
                    model = GPT2Sem(len(gpt2tokenizer))
        
                    model.load_state_dict(torch.load(open('./pickle_files/gptSem.pt', 'rb'), map_location=torch.device('cpu')))
        
        
                    model.eval()
                    sample_loader = preprocess_data_transformer(sample,"gpt2")          
                    for s in sample_loader:
                        sample1 = tuple(t for t in s)
                    b_input_ids, b_input_mask = sample1
                    pred = model.forward(input_ids=b_input_ids,attention_mask=b_input_mask)
                    result = F.softmax(pred.logits, dim=1)
                    result = result.cpu().data.numpy()
                    argmax = list(np.argmax(result, axis=1))
                    score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                    scores.append(score)  
            elif m == "BERTSyn2_concat":  # Bert
            
                model = BERTSyn()        
                model.load_state_dict(torch.load(open('./pickle_files/BERTSyn.pt', 'rb'), map_location=torch.device('cpu')))
                model.eval()
                sample_loader = preprocess_data_transformer_syntax(sample)
                for s in sample_loader: 
                    sample1 = tuple(t for t in s)
                b_input_ids, b_input_mask ,b_input_ids2, b_input_mask2= sample1
                pred = model.forward(input_ids=b_input_ids,attention_mask=b_input_mask,input_ids2=b_input_ids2,attention_mask2=b_input_mask2)
                result = F.softmax(pred.logits, dim=1)
                result = result.cpu().data.numpy()
                argmax = list(np.argmax(result, axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score)  
                    


            

    if csv:
        Accuracy=[]
        F1score=[]
        Precision=[]
        Recall=[]
        
        print(scores)
        print(original_scores)

        for i in range(len(modelnames)):

            scoresmodel=[]
            originalscoresmodel=[]  
            
            l=len(modelnames)
            j = i
            while j < len(scores): 
                scoresmodel.append(str(int(scores[j])+1)) 
                originalscoresmodel.append( original_scores[j]) 
                j=j+l
            
            
        
            accuracy=accuracy_score(originalscoresmodel,scoresmodel)
            precision = precision_score(originalscoresmodel,scoresmodel ,average='weighted')
            recall = recall_score(originalscoresmodel,scoresmodel ,average='weighted' )
            f1score = f1_score(originalscoresmodel,scoresmodel,average='weighted')

            Accuracy.append(accuracy)
            F1score.append(f1score)
            Precision.append(precision)
            Recall.append(recall)

        
        print(scores)
        print(accuracy)

        
        return {"data":  {"scores": scores, "text_ids": text_ids, "texts": texts, "original_scores": original_scores, "modelnames": mnames, "Accuracy":Accuracy
        ,"F1score":F1score
        ,"Precision":Precision
        ,"Recall":Recall
        }}

    else:    
        return {"data":  {"scores": scores, "modelnames": mnames, "texts": texts}}







   


   

@app.post("/uploadfile")
async def get_predict_file(niveau: int, data:list[UploadFile], db: Session = Depends(get_db)):
     
    scores = []
    scores_pred = []
    text_ids = []
    texts = []
    original_scores = []
    model_db = get_one_model(niveau, db)
    i=0
    for f in range(3):
        file = await f.read()
        print(file)
        print('khokha')
        sample = file.decode('ISO-8859-1')
        print(sample)
        i+=1
        text_ids.append(i)
        texts.append(sample)
        
        if model_db.saved_model_pickle == "sent_avg.pt":
            model = torch.load('./pickle_files/sent_avg.pt')
            model.eval()
            batch_padded, batch_lengths, original_index = preprocess_data_sentavg(
                sample)
            pred, avg_deg, score_pred = model.forward(
                batch_padded, batch_lengths, original_index, dim=1)
            argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)
            
            preds = list(score_pred.cpu().data.numpy())
            score_pred = json.dumps(preds[0], cls=NumpyArrayEncoder)
            scores_pred.append(score_pred)

        elif model_db.saved_model_pickle == "par_seq.pt":
            model = torch.load('./pickle_files/par_seq.pt')
            model.eval()
            batch_padded, batch_lengths, original_index = preprocess_data_parseq(
                sample)
            pred, avg_deg = model.forward(
                batch_padded, batch_lengths, original_index, dim=1)
            argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)
        elif model_db.saved_model_pickle == "sem_rel.pt":
            model = torch.load(
                './pickle_files/sem_rel.pt')
            model.eval()
            batch_padded, batch_lengths, original_index = preprocess_data_parseq(
                sample)
            pred = model.forward(batch_padded, batch_lengths,
                                 original_index, weights=best_weights, dim=1)
            argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)
        elif model_db.saved_model_pickle == "cnn_postag.pt":
            model = torch.load(
                './pickle_files/cnn_postag.pt')
            model.eval()
            batch_padded, batch_lengths, original_index = preprocess_data_cnnpostag(
                sample)
            pred = model.forward(
                batch_padded, batch_lengths, original_index, dim=1)
            argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)
        elif model_db.saved_model_pickle == "sem_syn.pt":
            model = pickle.load(open('./pickle_files/sem_syn.pt', 'rb'))
            model.eval()
            batch_padded_cnn, batch_lengths_cnn, original_index_cnn = preprocess_data_cnnpostag(
                sample)
            batch_padded_semrel, batch_lengths_semrel, original_index_semrel = preprocess_data_parseq(
                sample)
            pred = model.forward(batch_padded_semrel, batch_padded_cnn, batch_lengths_semrel,
                                 batch_lengths_cnn, original_index, weights=best_weights, dim=1)
            argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)
        else:
            model = torch.load(open('./pickle_files/BERTSem.pt', 'rb'), map_location=torch.device('cpu'))
            model.eval()
            sample_loader = preprocess_data_bert(sample)
            for s in sample_loader: 
                sample = tuple(t for t in s)
            b_input_ids, b_input_mask = sample
            pred = model.forward(b_input_ids=b_input_ids, b_input_mask=b_input_mask)
            result = F.softmax(pred.logits, dim=1)
            result = result.cpu().data.numpy()
            print(result)
            argmax = list(np.argmax(result, axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)

    print(scores)
    return {"data":  {"scores": scores, "text_ids": text_ids, "texts": texts}}



@app.post("/uploadcsvcomp")
async def get_predict_filecomp(file: UploadFile = File(...), modelnames:list[str]=Form(...)):
     
     
    content_assignment = await file.read()
    data = convert_csv(content_assignment)
    scores = []
    scores_pred = []
    text_ids = []
    texts = []
    original_scores = []
    mnames = []

    for i in range(2):
    
       
        for m in modelnames:

            sample = data[i]['text']
            text_ids.append(data[i]['text_id'])
            texts.append(data[i]['text'])
            original_scores.append(data[i]['labelA'])
            mnames.append(m)

          
            if m == "sent_avg":
                model = torch.load('./pickle_files/sent_avg.pt')
                model.eval
                batch_padded, batch_lengths, original_index = preprocess_data_sentavg(
                    sample)
                pred, avg_deg, score_pred = model.forward(
                    batch_padded, batch_lengths, original_index, dim=1)
                argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(score+1)
                
                preds = list(score_pred.cpu().data.numpy())
                score_pred = json.dumps(preds[0], cls=NumpyArrayEncoder)
                scores_pred.append(score_pred)

            elif m == "par_seq":
                model = torch.load('./pickle_files/par_seq.pt')
                model.eval()
                batch_padded, batch_lengths, original_index = preprocess_data_parseq(
                    sample)
                pred, avg_deg = model.forward(
                    batch_padded, batch_lengths, original_index, dim=1)
                argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(str(int(score)+1))
            elif m == "sem_rel":
                model = torch.load(
                    './pickle_files/sem_rel.pt')
                model.eval()
                batch_padded, batch_lengths, original_index = preprocess_data_parseq(
                    sample)
                pred = model.forward(batch_padded, batch_lengths,
                                    original_index, weights=best_weights, dim=1)
                argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(str(int(score)+1))
            elif m == "cnn_postag":
                model = torch.load(
                    './pickle_files/cnn_postag.pt')
                model.eval()
                batch_padded, batch_lengths, original_index = preprocess_data_cnnpostag(
                    sample)
                pred = model.forward(
                    batch_padded, batch_lengths, original_index, dim=1)
                argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(str(int(score)+1))
            elif m == "sem_syn":
                model = pickle.load(open('./pickle_files/sem_syn.pt', 'rb'))
                model.eval()
                batch_padded_cnn, batch_lengths_cnn, original_index_cnn = preprocess_data_cnnpostag(
                    sample)
                batch_padded_semrel, batch_lengths_semrel, original_index_semrel = preprocess_data_parseq(
                    sample)
                pred = model.forward(batch_padded_semrel, batch_padded_cnn, batch_lengths_semrel,
                                    batch_lengths_cnn, original_index, weights=best_weights, dim=1)
                argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
                score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
                scores.append(str(int(score)+1))
            
    Accuracy=[]
    F1score=[]
    Precision=[]
    Recall=[]
    
    print(scores)
    print(original_scores)
    for i in range(len(modelnames)):

        scoresmodel=[]
        originalscoresmodel=[]  
        
        l=len(modelnames)
        j = i
        while j < len(scores): 
           scoresmodel.append(scores[j]) 
           originalscoresmodel.append( original_scores[j]) 
           j=j+l
           
        
       
        accuracy=accuracy_score(originalscoresmodel,scoresmodel)
        precision = precision_score(originalscoresmodel,scoresmodel ,average='weighted')
        recall = recall_score(originalscoresmodel,scoresmodel ,average='weighted' )
        f1score = f1_score(originalscoresmodel,scoresmodel,average='weighted')

        Accuracy.append(accuracy)
        F1score.append(f1score)
        Precision.append(precision)
        Recall.append(recall)

      
    print(scores)
    print(accuracy)

    
    return {"data":  {"scores": scores, "text_ids": text_ids, "texts": texts, "original_scores": original_scores, "modelnames": mnames, "Accuracy":Accuracy
    ,"F1score":F1score
    ,"Precision":Precision
    ,"Recall":Recall
     }}









@app.get('/models/', response_model=List[SchemaModel], status_code=200)
# token: str = Depends(oauth2_scheme)
def get_all_models(db: Session = Depends(get_db)):
    list_models = db.query(ModelModel).all()
    return list_models

# Retourner les détails d'un modèle spécifié par son Id pour l'interface de modification côté admin


@app.get('/models/{model_id}', response_model=SchemaModel, status_code=200)
def get_one_model(model_id: int, db: Session = Depends(get_db)):
    db_model = db.query(ModelModel).filter(
        ModelModel.id == model_id).first()
    if db_model is None:
        raise HTTPException(status_code=404, detail="Modéle non existant")
    return db_model

# Retourner la description d'un modèle précis pour sidebar


@app.get('/description/{model_id}', status_code=200)
def get_one_model_desc(model_id: int, db: Session = Depends(get_db)):
    db_model = db.query(ModelModel.description).filter(
        ModelModel.id == model_id and ModelModel.visibility == True).first()
    if db_model is None:
        raise HTTPException(status_code=404, detail="Modéle non existant")
    return db_model

# Retourner les noms des modèles à afficher dans la liste déroulante des modèles existants


@app.get('/names/', status_code=200)
def get_models_name(db: Session = Depends(get_db)):
    db_model = db.query(ModelModel.id, ModelModel.name).filter(
        ModelModel.visibility == True).first()
    if db_model is None:
        raise HTTPException(status_code=404, detail="Aucun modèle n'éxiste")
    return db_model

# Ajouter un modèle dans l'interface d'ajout côté admin



@app.post('/add_model')


async def add_model(    name: str = Form(...),
    description: str = Form(...),
    F1_score: str = Form(...),
    precision: str = Form(...),
    accuracy: str = Form(...),
    rappel: str = Form(...),
    preprocess: str = Form(...),
    hybridation: bool = Form(...),
    visibility: bool = Form(...),
    cloud:bool= Form(...),
    file_info: UploadFile = File(...) ,db: Session = Depends(get_db)):
   
    saved_model_pickle=file_info.filename
    file_extension = os.path.splitext(saved_model_pickle)[1]
    if file_extension != ".pt":
            raise HTTPException(status_code=400, detail="Veuillez respectez le format des données.")
       
    contents = await file_info.read()


    with open('./pickle_files/'+saved_model_pickle, "wb") as f:
        f.write(contents)
    model_data = {
        "id":50,
        "name": name,
        "description": description,
        "F1_score": F1_score,
        "precision": precision,
        "accuracy": accuracy,
        "rappel": rappel,
        "saved_model_pickle": saved_model_pickle,
        "preprocess": preprocess,
        "hybridation": hybridation,
        "visibility": visibility,
        "cloud": cloud
    }
    model = SchemaModel(**model_data)
    db_model_name = db.query(ModelModel).filter(
        ModelModel.name == model.name).first()
    if db_model_name:
        raise HTTPException(status_code=400, detail="Modèle déjà existant")
    else:
        db_model = ModelModel(name=model.name, description=model.description,
                              F1_score=model.F1_score, precision=model.precision, accuracy=model.accuracy,  rappel=model.rappel, saved_model_pickle=model.saved_model_pickle, preprocess=model.preprocess, hybridation=model.hybridation)
        db.add(db_model)
        db.commit()  






@app.put("/update_model/{model_id}",response_model=SchemaModel)
# token: str = Depends(oauth2_scheme)
def update_model(model_id: int, model: SchemaModel, db: Session = Depends(get_db)):
    model_to_update = db.query(ModelModel).filter(
        ModelModel.id == model_id).first()
    if model.name:
        model_to_update.name = model.name
    if model.description:
        model_to_update.description = model.description
    if model.F1_score:
        model_to_update.F1_score = model.F1_score
    if model.precision:
        model_to_update.precision = model.precision
    if model.accuracy:
        model_to_update.accuracy = model.accuracy
    if model.rappel:
        model_to_update.rappel = model.rappel
    if model.preprocess:
        model_to_update.preprocess = model.preprocess
    if model.hybridation:
        model_to_update.hybridation = model.hybridation
    # if model.visibility:
    model_to_update.visibility = model.visibility

    db.commit()
    return model_to_update

# Modifier la visibilité d'un modèle


@app.put("/update_model_visibility/{model_id}", response_model=SchemaModel)
# token: str = Depends(oauth2_scheme)
def update_model_vis(model_id: int, visib: bool, db: Session = Depends(get_db)):
    model_to_update = db.query(ModelModel).filter(
        ModelModel.id == model_id).first()
    model_to_update.visibility = visib
    db.commit()
    return model_to_update
# Login


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_admin(username: str, db: Session = Depends(get_db)):
    result = db.query(ModelAdmin).filter(
        ModelAdmin.user_name == username).first()
    return result


def authenticate_user(username: str, password: str, db: Session = Depends(get_db)):
    admin = get_admin(username, db)
    if not admin:
        return False
    if not verify_password(password, admin.pwd):
        return False
    return admin


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Login


@app.post("/login", response_model=Token)
async def login_for_access_token(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Nom d'utilisateur ou mot de passe est incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.user_name}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


def get_password_hash(password):
    return pwd_context.hash(password)

# To add a new admin


@app.post('/sign_up', response_model=SchemaAdmin)
def add_admin(admin: SchemaAdmin, db: Session = Depends(get_db)):
    db_admin_name = db.query(ModelModel).filter(
        ModelAdmin.user_name == admin.user_name).first()
    if db_admin_name:
        raise HTTPException(status_code=400, detail="Modèle déjà existant")
    else:
        db_admin = ModelAdmin(user_name=admin.user_name,
                              pwd=get_password_hash(admin.pwd))
        db.add(db_admin)
        db.commit()
        return db_admin

@app.post('/add_model_file_option')


async def add_model(
    file_config: UploadFile = File(...),
    db: Session = Depends(get_db)):
     
    #saved_model_pickle=file_info.filename
    #file_extension = os.path.splitext(saved_model_pickle)[1]
    #if file_extension != ".pt":
    #        raise HTTPException(status_code=400, detail="Veuillez respectez le format des données.")
       
    #contents = await file_info.read()
    try:
        #with open('./pickle_files/'+saved_model_pickle, "wb") as f:
        #    f.write(contents)
            # Read the fields from the JSON file
           
       
        file_config_contents = await file_config.read()
        file_config_data = convert_json(file_config_contents)
 
   
        # Extract the fields from the JSON data
        name = file_config_data.get("name")
        description = file_config_data.get("description")
        F1_score = file_config_data.get("F1_score")
        precision = file_config_data.get("precision")
        accuracy = file_config_data.get("accuracy")
        rappel = file_config_data.get("rappel")
        preprocess = file_config_data.get("preprocess")
        path_file=file_config_data.get("pickle_path")
       
        destination_folder ="./pickle_files"


        if os.path.exists(path_file):
       
            file_name = os.path.basename(path_file)
            destination_path = os.path.join(destination_folder, file_name)
            shutil.copy(path_file, destination_path)
           
        else:
            raise HTTPException(status_code=400, detail="Le fichier pickle n'existe pas")
       
       
        model_data = {
            "id":50,
            "name": name,
            "description": description,
            "F1_score": F1_score,
            "precision": precision,
            "accuracy": accuracy,
            "rappel": rappel,
            "saved_model_pickle":file_name ,
            "preprocess": preprocess,
            "hybridation": False,
            "visibility": True,
            "cloud":False
        }


       
        model = SchemaModel(**model_data)
        db_model_name = db.query(ModelModel).filter(
            ModelModel.name == model.name).first()
        if db_model_name:
            raise HTTPException(status_code=400, detail="Modèle déjà existant")
        else:
            db_model = ModelModel(name=model.name, description=model.description,
                                F1_score=model.F1_score, precision=model.precision, accuracy=model.accuracy,  rappel=model.rappel, saved_model_pickle=model.saved_model_pickle, preprocess=model.preprocess, hybridation=model.hybridation)
            db.add(db_model)
            db.commit()  
   
    except HTTPException as http_exception:
        if http_exception.status_code in [400, 401]:
            raise  # Reraise the HTTPException for status codes 400 and 401
        else:
            raise HTTPException(status_code=500, detail="Veuillez respecter le format des données")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Veuillez respecter le format des données")
   
















# Configuring the server host and port
if __name__ == '__main__':
    uvicorn.run("main:app", port=8080, host='0.0.0.0', reload=True)
