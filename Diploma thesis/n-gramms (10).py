import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import re
import operator
import nltk
import csv
import copy
import seaborn as sns
import pandas as pd
import sklearn
import networkx as nx
import langid
import datetime as dt
from matplotlib import style
from scipy import signal
from scipy import interpolate
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim import corpora,models,similarities,matutils
from gensim.corpora.dictionary import Dictionary
from sklearn.cluster import DBSCAN
from sklearn.decomposition import NMF,PCA
from collections import Counter,defaultdict,OrderedDict

data=pd.read_csv(r'C:\Users\dioni\Downloads\tweets.csv')

#Creating function to extract different items from tweets
regex_str=[
r'<[^>]+>', #HTML tags
r'(?:@[\w_]+)', #@-mentions
r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", #hash-tags
r'http[s]?://(?:[a-z]|[0-9]|$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',#URLs
r'(?:(?:\d+,?)+(?:\.?\d+)?)', #numbers
r"(?:[a-z][a-z'\-_]+[a-z])", #words with - and '
r'(?:[\w_]+)', #other words
r'(?:\S)' #anything else
]

tokens_re=re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
def tokenize(s):
    return tokens_re.findall(s)
def preprocess(s,lowercase=False):
    tokens=tokenize(s)
    if lowercase:
        tokens=[token.lower() for token in tokens]
        return tokens
    
#Stopwords will only be removed for 1-gram but not for 2-5 grams to get the n-grams as they were written.

#1-gram Getting top20 words
vect1=CountVectorizer(analyzer='word', stop_words="english", min_df=200, decode_error='ignore', ngram_range=(1,1),dtype=np.int32)
sub11=vect1.fit_transform(data["tweets"].map(lambda x:"".join(preprocess(x,lowercase=True))).tolist())#Applying Vectorizer to preprocessed tweets
sub12=zip(vect1.get_feature_names_out(),np.asarray(sub11.sum(axis=0)).ravel())#Creating (word,count)list
words=sorted(sub12,key=lambda x:x[1], reverse=True)[0:20]#Getting Top20 words
print(words)

#2-gram Getting top20 words
vect2=CountVectorizer(analyzer='word', min_df=2, decode_error='ignore', ngram_range=(2,2),dtype=np.int32)
sub21=vect2.fit_transform(data["tweets"].map(lambda x:"".join(preprocess(x,lowercase=True))).tolist())#Applying Vectorizer to preprocessed tweets
sub22=zip(vect2.get_feature_names_out(),np.asarray(sub21.sum(axis=0)).ravel())#Creating (word,count)list
words2=sorted(sub22,key=lambda x:x[1], reverse=True)[0:20]#Getting Top20 words
print(words2)

#3-gram Getting top20 words
vect3=CountVectorizer(analyzer='word', min_df=2, decode_error='ignore', ngram_range=(3,3),dtype=np.int32)
sub31=vect3.fit_transform(data["tweets"].map(lambda x:"".join(preprocess(x,lowercase=True))).tolist())#Applying Vectorizer to preprocessed tweets
sub32=zip(vect3.get_feature_names_out(),np.asarray(sub31.sum(axis=0)).ravel())#Creating (word,count)list
words3=sorted(sub32,key=lambda x:x[1], reverse=True)[0:20]#Getting Top20 words
print(words3)

#4-gram Getting top20 words
vect4=CountVectorizer(analyzer='word', min_df=2, decode_error='ignore', ngram_range=(4,4),dtype=np.int32)
sub41=vect4.fit_transform(data["tweets"].map(lambda x:"".join(preprocess(x,lowercase=True))).tolist())#Applying Vectorizer to preprocessed tweets
sub42=zip(vect4.get_feature_names_out(),np.asarray(sub41.sum(axis=0)).ravel())#Creating (word,count)list
words4=sorted(sub42,key=lambda x:x[1], reverse=True)[0:20]#Getting Top20 words
print(words4)

#5-gram Getting top20 words
vect5=CountVectorizer(analyzer='word', min_df=2, decode_error='ignore', ngram_range=(5,5),dtype=np.int32)
sub51=vect5.fit_transform(data["tweets"].map(lambda x:"".join(preprocess(x,lowercase=True))).tolist())#Applying Vectorizer to preprocessed tweets
sub52=zip(vect5.get_feature_names_out(),np.asarray(sub51.sum(axis=0)).ravel())#Creating (word,count)list
words5=sorted(sub52,key=lambda x:x[1], reverse=True)[0:20]#Getting Top20 words
print(words5)

#usernames in tweets
tags=data["tweets"].map(lambda x: [tag for tag in preprocess(x, lowercase=True) if tag.startswith('@')])
tags=sum(tags,[])
tags[0:5]
usernames=Counter(tags).most_common(20) #Top20
print(usernames)

#hashtags in tweets
hashs=data["tweets"].map(lambda x:[hashs for hashs in preprocess(x,lowercase=True) if hashs.startswith('#')])
hashs=sum(hashs,[])
hashs[0:5]
hashtags=Counter(hashs).most_common(20) #Top20
print(hashtags)

#urls in tweets
urls=data["tweets"].map(lambda x:[url for url in preprocess(x, lowercase=True) if url.startswith('http:') or url.startswith('https:')])
urls=sum(urls,[])
urls[0:5]
Urls=Counter(urls).most_common(20) #Top20
print(Urls)
