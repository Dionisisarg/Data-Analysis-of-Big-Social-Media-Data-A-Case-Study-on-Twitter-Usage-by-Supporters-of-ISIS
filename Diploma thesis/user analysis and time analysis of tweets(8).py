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

plt.style.use('ggplot')
data=pd.read_csv(r'C:\Users\dioni\Downloads\tweets.csv',parse_dates=['time'])

def tsimple(x):
    return dt.datetime(x.year, x.month, x.day)

#create the diagram with Top 10 Twitter handlers
data['time']=data.time.apply(tsimple)
top_handles=data.username.value_counts().sort_values(ascending=False)
top_handles.head(10).plot.barh(title='Top 10 Twitter handlers',figsize=(16,8))
plt.show()

#Tweets over time
#isis related tweets over time
data.time.value_counts().plot(title='ISIS related tweets over time',xlim=[dt.date(2015,10,1),dt.date(2016,4,30)],figsize=(16,8))
plt.show()

#total number of isis related tweets over time
data.time.value_counts().sort_index().cumsum().plot.area(title='Total number of ISIS related tweets over time',xlim=[dt.date(2015,10,1),dt.date(2016,4,30)],figsize=(16,8))
plt.show()
