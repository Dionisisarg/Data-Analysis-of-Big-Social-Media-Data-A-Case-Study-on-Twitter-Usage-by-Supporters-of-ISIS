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
dataset=pd.read_csv(r'C:\Users\dioni\Downloads\tweets.csv')

#unique tweets count on dataset
print("Unique tweets: {}".format(len(dataset['tweets'].unique())))

#all tweets count on dataset
print("All tweets: {}".format(len(dataset['tweets'])))

retweets=[]
actual_tweets=[]
for user,tweet in zip(dataset['username'],dataset['tweets']):
    match=re.search(r'^\bRT\b',tweet)
    if match==None:
        actual_tweets.append([user,tweet])
    else:
        retweets.append([user,tweet])
actual_tweets=np.array(actual_tweets)
retweets=np.array(retweets)
plt.bar([1,2], [len(actual_tweets[:,1]), len(retweets[:,1])], align='center')
ax=plt.gca()
ax.set_xticks([1,2])
ax.set_xticklabels(['Actual Tweets','Retweets'])

in_set=[]
not_in_set=[]
for record in actual_tweets:
    match=re.findall(r'@\w*',record[1])
    if match!=[]:
        for name in match:
            if(name[1:] in dataset['username'].unique()) and (record[0]!=name[1:]):
                in_set.append([record[0],name[1:]])
            elif record[0]!=name[1:]:
                not_in_set.append([record[0],name[1:]])
in_set=np.array(in_set)
not_in_set=np.array(not_in_set)
fig,ax=plt.subplots(1,2)
ax[0].bar([1,2], [len(np.unique(in_set[:,1])), len(np.unique(not_in_set[:,1]))], align='center')
ax[0].set_xticks([1,2])
ax[0].set_xticklabels(['In','Not in'])
ax[0].set_title('Users in vs. not in tweets.csv', fontsize=9)
ax[1].bar([1,2], [len(np.unique(in_set[:,1])), len(dataset['username'].unique())], align='center')
ax[1].set_xticks([1,2])
ax[1].set_xticklabels(['Mentioned','Total'])
ax[1].set_title('Mentioned vs. Total in tweets.csv', fontsize=9)
plt.show()

sender_count=Counter(in_set[:,0])
receiver_count=Counter(in_set[:,1])
top_5_senders=sender_count.most_common(5)
top_5_receivers=receiver_count.most_common(5)
print("Top 5 senders:",top_5_senders)
print("Top 5 receivers:",top_5_receivers)

for name,_ in top_5_receivers:
    print("Username: {} - {}\n".format(name,dataset[dataset['username']==name]['description'].dropna().unique()[0]))


graph = nx.Graph()

all_users=list(set(in_set[:,0])| set(in_set[:,1]))
graph.add_nodes_from(all_users, count=10)
node_colours=[]

for node in graph.nodes():
    if node in (set(in_set[:,0]) & set(in_set[:,1])):
        node_colours.append('g')
    elif node in np.unique(in_set[:,0]):
        node_colours.append('r')
    elif node in np.unique(in_set[:,1]):
        node_colours.append('b')
edges={}
occurrence_count=Counter(map(tuple, in_set))
for(sender,receiver), count in occurrence_count.items():
    if (receiver,sender)in edges.keys():
        edges[(receiver,sender)]=edges[(receiver,sender)]+count
    else:
        edges[(sender,receiver)]=count

for (sender, receiver), count in edges.items():
    graph.add_edge(sender,receiver,weight=count)

followers={}
tweet_num={}
for username in all_users:
    followers[username]=dataset[dataset['username']==username]['followers'].unique()[-1]
    tweet_num[username]=dataset[dataset['username']==username]['tweets'].count()

sizes=[(followers[n]/tweet_num[n])*50 for n in graph.nodes()]

plt.figure(figsize=(12,12))
nx.draw(graph, pos=nx.spring_layout(graph), node_color=node_colours, with_labels=True, width=1)
plt.show()
