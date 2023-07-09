# build a schema using pydantic
from typing import Text, Optional
from fastapi import Form, Query
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException 




class Admin(BaseModel):
 
    user_name : str
    pwd : str


    class Config:
        orm_mode = True

class Model(BaseModel):
    id : int
    name : str
    description : Text
    F1_score : str
    precision : str
    accuracy : str
    rappel : str # new
    saved_model_pickle : str # new
    preprocess :str # new
    hybridation : bool # new
    visibility : bool
    cloud:bool

    class Config:
        orm_mode = True
        
class Inputs(BaseModel):
    text: str
    selectedIndex: int

class Inputlist(BaseModel):
    text: str
    modelList: list[str]

class Filelist(BaseModel):
    modelList: list[str]

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
