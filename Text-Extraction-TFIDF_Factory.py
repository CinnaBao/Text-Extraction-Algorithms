# coding: utf-8
# Author: Bao Wan-Yun

import pandas as pd
import numpy as np
import re
import pyodbc
import string
from nltk.tokenize import RegexpTokenizer
import time
import os
import math

#tokenizer = RegexpTokenizer(r'[\d.]+\b\.*|[\w\']+')
tokenizer = RegexpTokenizer(r'[\d]+\.[\w]*|[a-zA-Z.]+\.[\w]|[\w\']+')

class Text_Extraction_TFIDF():
    def __init__(self):
        self.author = 'WY Bao'
        
    def Title_Term_summarize(self, Data,DF_name, Label_name):
        for subctlg in set(Data.Subcate):
            Dict_name = {} ;
            datasplit = Data.loc[Data['Subcate']==subctlg]
            for com in datasplit[Label_name]:
                raw = tokenizer.tokenize(com.lower())
                raw = [tt for tt in raw if len(tt) < 30]
                for i in raw: #each term
                    if i in Dict_name:
                        Dict_name[i] += 1
                    else:
                        Dict_name[i] = 1
            DF_name = DF_name.append(pd.DataFrame([subctlg, k, v] for k, v in Dict_name.items() ))
        #print(len(Dict_name))
        DF_name.columns= ['Subcate', 'Term', 'Cnt']
        DF_name = DF_name.reset_index(drop=True)
        return DF_name

    def AdjustWordCNT(self, Title,Mfty):
        grp = Title.groupby(['Subcate','Term']).sum().reset_index()
        grp2 = pd.merge(grp, Mfty, on = ['Subcate', 'Term'], how = 'left')
        grp2['Adjust'] = np.where(np.isnan(grp2['Cnt_y']), 1, 5)
        grp2 = grp2.assign(Cnt_adj = lambda x: x.Cnt_x * x.Adjust)[['Subcate', 'Term', 'Cnt_adj']]
        return (grp , grp2)

    def TF_IDF(self, WordCNT,subCNT):
        SubCtlg_MAX = WordCNT.groupby(['Subcate']).max().reset_index()
        SubCtlg_MAX.columns= [['Subcate', 'Term', 'Cnt_max']]
        IDF = WordCNT.groupby(['Term']).size().reset_index()
        IDF.columns = ['Term', 'CNT']
        IDF = IDF.assign(IDF = lambda x: np.log(subCNT/x.CNT))
        TFIDF = pd.merge(pd.merge(WordCNT, SubCtlg_MAX, on = ['Subcate'], how = 'left'), IDF, left_on = ['Term_x'], right_on = ['Term'], how = 'left')
        #print(TFIDF[0:2])
        TFIDF = TFIDF.assign(TF = lambda x: x.Cnt_adj/x.Cnt_max)
        TFIDF = TFIDF.assign(TFIDF = lambda x: x.Cnt_adj/x.Cnt_max*x.IDF)
        Subctlg_Term_Weight = TFIDF[['Subcate', 'Term_x', 'Cnt_adj', 'TFIDF']]
        Subctlg_Term_Weight.columns = ['Subcate', 'Term', 'Cnt', 'weight']
        return Subctlg_Term_Weight

    def SaveAsCSV_InsertIntoSQL(self, data):
        pathProg = '/home/Text-Extraction-Algorithms/'
        os.chdir(pathProg)
        input_data = data.drop(data[data.Cnt <= 1].index).reset_index()
        input_data['index'] = input_data.index + 1  
        TT1 = time.time()
        csv_header = ['ID', 'Subcate', 'Term', 'Cnt', 'TFIDF_Weight']
        input_data.to_csv(pathProg+'/TFIDF_weight.csv', header = csv_header, encoding = 'utf-8', index=False)
        print('total time to csv file: %d mins' %((time.time()-TT1)/60))
        return ('finish')

