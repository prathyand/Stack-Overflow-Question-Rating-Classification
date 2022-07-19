import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import nltk   
import contractions
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import html2text

def repls(row):
    cont_replace = [] 
    for word in row.strip().split():
#         print(word)
        cont_replace.append(contractions.fix(word))
#     print(cont_replace)
    cont_replace=" ".join(cont_replace)
    stopword = set(stopwords.words('english'))
    tokenword = word_tokenize(cont_replace)
    cont_replace=[word_test for word_test in tokenword if not word_test.lower() in stopword]
    cont_replace=" ".join(cont_replace)
    return cont_replace

inp_train = pd.read_csv("/kaggle/input/60k-stack-overflow-questions-with-quality-rate/train.csv")
inp_validation= pd.read_csv("/kaggle/input/60k-stack-overflow-questions-with-quality-rate/valid.csv")

inp_train_aug=inp_train
inp_train_aug['BodyText']=inp_train_aug.apply(lambda row: html2text.html2text(row.Body), axis = 1)
inp_train_aug['BodyText']=inp_train_aug.apply(lambda row: row.BodyText.replace("\n", " "), axis = 1)
inp_train_aug.to_csv('old_selected_train.csv',columns=['Title','Body','BodyText','Y'],index=False)

inp_validation_aug=inp_validation
inp_validation_aug['BodyText']=inp_validation_aug.apply(lambda row: html2text.html2text(row.Body), axis = 1)
inp_validation_aug['BodyText']=inp_validation_aug.apply(lambda row: row.BodyText.replace("\n", " "), axis = 1)
inp_validation_aug.to_csv('old_selected_validation.csv',columns=['BodyText','Y'],index=False)

inp_train_aug2=inp_train
reg_replace={'\\n':' ',r'\W':' ',r'http\s+|www.\s+':r'',r'https\s+|www.\s+':r'',
             r'\s+[a-zA-Z]\s+':' ',r'\^[a-zA-Z]\s+':' ',r'\s+':' ',r"\’":"\'",r"\r":"",r"\n":"",r"[0-9]":"num",
            r'[.|,|)|(|\|/]':r' '}
inp_train_aug2['Merged']=inp_train['Title']+" "+inp_train['Body']
inp_train_aug2['Merged']=inp_train_aug2['Merged'].str.lower()
for k,v in reg_replace.items():
    inp_train_aug2['Merged']=inp_train_aug2['Merged'].map(lambda row: re.sub(k,v,str(row)))
reg_repl2= {"\'":"",r"\"":"",r'[?|!|\'|"|#]':r''}
inp_train_aug2['Merged']=inp_train_aug2.apply(lambda row: repls(row.Merged), axis = 1)
for k,v in reg_repl2.items():
    inp_train_aug2['Merged']=inp_train_aug2['Merged'].map(lambda row: re.sub(k,v,str(row)))
inp_train_aug2.to_csv('allop_train.csv',columns=['Merged','Y'],index=False)

inp_val_2=inp_validation
inp_val_2=inp_validation
reg_replace={'\\n':' ',r'\W':' ',r'http\s+|www.\s+':r'',r'https\s+|www.\s+':r'',
             r'\s+[a-zA-Z]\s+':' ',r'\^[a-zA-Z]\s+':' ',r'\s+':' ',r"\’":"\'",r"\r":"",r"\n":"",r"[0-9]":"num",
            r'[.|,|)|(|\|/]':r' '}
inp_val_2['Merged']=inp_val_2['Title']+" "+inp_val_2['Body']
inp_val_2['Merged']=inp_val_2['Merged'].str.lower()
for k,v in reg_replace.items():
    inp_val_2['Merged']=inp_val_2['Merged'].map(lambda row: re.sub(k,v,str(row)))
reg_repl2= {"\'":"",r"\"":"",r'[?|!|\'|"|#]':r''}
inp_val_2['Merged']=inp_val_2.apply(lambda row: repls(row.Merged), axis = 1)
for k,v in reg_repl2.items():
    inp_val_2['Merged']=inp_val_2['Merged'].map(lambda row: re.sub(k,v,str(row)))
inp_val_2.to_csv('allop_val.csv',columns=['Merged','Y'],index=False)