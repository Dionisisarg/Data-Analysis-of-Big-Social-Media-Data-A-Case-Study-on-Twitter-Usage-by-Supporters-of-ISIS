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

df=pd.read_csv(r'C:\Users\dioni\Downloads\tweets.csv')

junk=re.compile("al|RT|\n|&.*?;|http[s](?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)*")
tweets=[junk.sub("",t) for t in df.tweets]
vec=TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=.5)
tfv=vec.fit_transform(tweets)
terms=vec.get_feature_names_out()
#make the image of WordCloud
wc=WordCloud(height=1000, width=1000, max_words=1000).generate(" ".join(terms))
plt.figure(figsize=(10,10))
plt.imshow(wc)
plt.axis('off')
plt.show()

#A Quick and Dirty Topic Model
#Sounds unpleasant. Lets try a topic model to get some more grabularity.
from sklearn.decomposition import NMF
nmf=NMF(n_components=10).fit(tfv)
for idx,topic in enumerate(nmf.components_):
    print('Topic #%d:'%idx)
    print(''.join([terms[i] for i in topic.argsort()[:-10-1:-1]]))
    print('')

#Top 10 Users and their Topic Distributions
#If we have topics, we might as well see what the top 10 users are into.
#Evidently Uncle_SamCoCo likes to talk about Al Qaida...
style.use('bmh')
df['topic']=np.argmax(nmf.transform(vec.transform(tweets)),axis=1)
top10_users=df[df.username.isin(df.username.value_counts()[:10].keys().tolist())]
pd.crosstab(top10_users.username,top10_users.topic).plot.bar(stacked=True,figsize=(16,10),colormap='coolwarm')
plt.show()
