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
df=pd.read_csv(r'C:\Users\dioni\Downloads\tweets.csv')

#make the diagram of time and tweets. Time is from 01/01/2016-13/05/2016.
df.time=pd.to_datetime(df.time)
perhr=df.set_index(df['time']).resample('D').agg('count')
moving_avg=perhr.rolling(window=7).mean().tweets['2016-01-01':].plot()
plt.show()

#Frequency of ISIS tweets in 2016
fig,ax=plt.subplots(figsize=(20,8))
perhr['2016-01-01':].numberstatuses.interpolate(method='linear').plot(ax=ax, color='black', fontsize=12, alpha=0.1)
moving=perhr.rolling(window=7).mean().tweets['2016-01-01':].plot(color='r')
yemen='2016-01-29'
brussels='2016-03-22'
ax.annotate('Bombing in Brussels',xy=(brussels,200),xytext=('2016-03-03',310),arrowprops=dict(facecolor='white',shrink=0.05),size=15)
ax.annotate('Car bombing in Yemen',xy=(yemen,200),xytext=('2016-01-10',310),arrowprops=dict(facecolor='white',shrink=0.05),size=15)
ax.margins(None,0.1)
ax.legend(['Tweets Per Day','7-Day Rolling Average'],loc='upper right',numpoints=1,labelspacing=2.0,fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Number of Tweets')
ax.set_title('Frequency of ISIS Tweets in 2016')
fig.savefig('temp.png')
plt.show()
                                                 
