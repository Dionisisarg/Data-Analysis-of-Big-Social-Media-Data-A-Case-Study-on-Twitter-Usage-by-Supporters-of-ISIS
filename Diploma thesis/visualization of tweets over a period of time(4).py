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

#plotting params, make the way of draw diagrams
mpl.rcParams['figure.figsize']=(8,5)
mpl.rcParams['lines.linewidth']=3
plt.style.use('ggplot')

df=pd.read_csv(r'C:\Users\dioni\Downloads\tweets.csv',parse_dates=[-2])

def f(x):
    return dt.datetime(x.year,x.month,x.day)

#make the diagram of tweets and time. Time is from 06/01/2015- 13/05/2016
df['time']=df.time.apply(f)
time_series=pd.DataFrame(df.time.value_counts().sort_index().resample('D').mean().fillna(0))
time_series.columns=['Tweets']
time_series.plot()
plt.show()

#Wow. what re those oscillations going on near January? Let's take a closer look!
time_series=time_series['2016-01-28':]
time_series.plot()
ts=time_series.values.ravel()
plt.show()

#Let's try and figure out what weekday these tweets are made.
#We will first need to find the places where the peaks occur. Luckily, scipy has a function for that
p=signal.find_peaks_cwt(ts,np.arange(1,4))
t=np.arange(len(ts))
plt.plot(t,ts)
plt.plot(t[p],ts[p],'o')
plt.show()

#Looks like we got em! Now, let's figure out on which weekday thoseoccur.
r=time_series.iloc[p].reset_index().copy()
r.columns=['date','tweet']
r['weekday']=r.date.apply(lambda x:x.weekday())
we=r.weekday.value_counts()
print(we)
