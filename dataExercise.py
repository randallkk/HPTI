import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re      

ps=PorterStemmer()

harryPotter = pd.read_csv("C:/Users/User/Desktop/datasets/HP1.csv",sep=";" ,encoding="ISO-8859-1")
trainPotter = harryPotter['Sentence'].str.replace('[^a-zA-Z ]', '')

tokenData = nltk.word_tokenize(str(trainPotter))
print(tokenData)

for w in tokenData:
    print(ps.stem(w))