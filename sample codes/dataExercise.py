import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re  
import glob
import os

ps=PorterStemmer()

input_file = r'.\data' # csv파일들이 있는 디렉토리 위치
output_file = r'.\data\Harry Potter.csv' # 병합하고 저장하려는 파일명

allFile_list = glob.glob(os.path.join(input_file, 'Harry Potter *')) # glob함수로 sales_로 시작하는 파일들을 모은다
allData = [] # 읽어 들인 csv파일 내용을 저장할 빈 리스트를 하나 만든다
for file in allFile_list:
    df = pd.read_csv(file,sep=';', names = ['Character', 'Sentence'], encoding='iso-8859-1', index_col=0) # for구문으로 csv파일들을 읽어 들인다
    allData.append(df) # 빈 리스트에 읽어 들인 내용을 추가한다
print(allData)
dataCombine = pd.concat(allData, axis=0, ignore_index=True) # 리스트의 내용을 수직으로 병합. ignore_index=True는 인데스 값이 기존 순서를 무시하고 순서대로 정렬되도록 한다.
dataCombine.to_csv(output_file, index=False) # to_csv함수로 저장한다. 인데스를 빼려면 False로 설정



harry=pd.read_csv('./data/Harry Potter.csv')
print(harry)


trainPotter = harry['Sentence'].str.replace('[^a-zA-Z ]', '')
print(trainPotter)


tokenData = nltk.word_tokenize(str(trainPotter))
print(tokenData)

stemData=[]
for w in tokenData:
    tempData=[]
    tempData=ps.stem(w)
    stemData.append(tempData)
    
print(stemData)
