# coding: utf-8

import pandas as pd
import Text_Extraction_TFIDF as Factory

f = Factory.Text_Extraction_TFIDF()

df = pd.read_csv(r'C:\Users\baowanyun\amazon_co-ecommerce_sample.csv') ; #print(df[0:5])
#print(df.columns) ; print(df.shape) ## (10000, 17)
df = df[['uniq_id','product_name','manufacturer','amazon_category_and_sub_category']]
df = df.rename(columns = {"uniq_id":"ID", "product_name": "Title", "manufacturer":"Mfty", "amazon_category_and_sub_category": "Subcategory"})
df = df.dropna()
df = df.reset_index(drop=True) #; print(df.shape)  ## (9306, 3)
Subcate = [i[::-1][ 0 : i[::-1].find(' >') ][::-1] for i in df['Subcategory']]
df['Subcate'] = Subcate
#Subcate_name = list(df.groupby(['Subcate']).groups.keys())
#print(len(Subcate_name)) ; print(Subcate_name)
'''
AA = df.groupby('Subcate')['ID'].count().rename('Subcate_cnt').to_frame() ; 
print(len(AA)) ## 235
print(len(AA[AA.Subcate_cnt > 50])) ## 44
'''
Subcate_cnt = df.groupby('Subcate')['ID'].count().rename('Subcate_cnt')
zip_df = df.merge(Subcate_cnt.to_frame(),left_on='Subcate',right_index=True)
df = zip_df[zip_df.Subcate_cnt > 50].reset_index(drop=True)
#len(df.groupby('Subcate')['ID'].count()) ## 44
df_Title = df[['Subcate', 'Title']]
df_Mfty = df[['Subcate', 'Mfty']]

## TF-IDF ##
Title_subcatWordCnt = pd.DataFrame()
Mfty_subcatWordCnt = pd.DataFrame()

Title_WordCNT = f.Title_Term_summarize(df_Title , Title_subcatWordCnt , 'Title')
Mfty_WordCNT = f.Title_Term_summarize(df_Mfty , Mfty_subcatWordCnt , 'Mfty')

subCNT = len(set(f.AdjustWordCNT(Title_WordCNT,Mfty_WordCNT)[0]['Subcate'])) ##44
Adj_WordCNT = f.AdjustWordCNT(Title_WordCNT,Mfty_WordCNT)[1]

TFIDF_Weight = TF_IDF(Adj_WordCNT,subCNT)

storeCSV = f.SaveAsCSV_InsertIntoSQL(TFIDF_Weight)

