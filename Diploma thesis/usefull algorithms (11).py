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

df=pd.read_csv(r'C:\Users\dioni\Downloads\tweets.csv',header=0)

#number of tweets
print(len(df))

#number of unique tweet accounts in the dataset
unique_tweeterites=df.groupby('username')
print(len(unique_tweeterites))

#tweets by day
df['date']=df['time'].map(lambda x:pd.to_datetime(str(x).split()[0]))
tweets_by_day=df.groupby('date').count().reset_index()
print(len(tweets_by_day))

#create the diagram, tweets by day trend
tweets_by_day=tweets_by_day.sort_values(by='date')
x=tweets_by_day['date']
y=tweets_by_day['name']
plt.xlabel('Date')
plt.ylabel('Number of tweets')
plt.xticks(rotation=45)
plt.title('Number of tweets trend by dates')
plt.plot(x,y,label='Tweets trend by days')
plt.show()

#histogram for number of tweets every two months
df['date'].hist(bins=8)
plt.xticks(rotation=45)
plt.show()

#create the diagram, most number of tweets in a day
top_10_max_tweets_days=tweets_by_day.sort_values(by='username').tail(10)
x=top_10_max_tweets_days['date']
y=top_10_max_tweets_days['name']
plt.xlabel('Date')
plt.ylabel('Number of tweets')
plt.title('Most number of tweets in a day')
plt.xticks(range(10),x,rotation=45)
plt.bar(range(10),y,label='Most tweets in a day')
plt.show()

#create the diagram, most number of tweets by user
tweeterites=df.groupby(['username']).count().reset_index()
tweeterites=tweeterites.sort_values(by='tweets').tail(10)
x=tweeterites['username']
y=tweeterites['tweets']
plt.xlabel('Twitter users')
plt.ylabel('Number of tweets')
plt.title('Most number of tweets by user')
plt.xticks(range(10),x,rotation=45)
plt.bar(range(10),y,label='Most tweets and retweets by user')
plt.show()

#create the diagram, most followed users
most_followed_users=df.drop_duplicates('username',keep='last')
most_followed_users_top_10=most_followed_users.sort_values(by='followers').tail(10)
x=most_followed_users_top_10['username']
y=most_followed_users_top_10['followers']
plt.xlabel('Username')
plt.ylabel('Followers')
plt.title('Most followed user')
plt.xticks(range(10),x,rotation=60)
plt.bar(range(10),y,label='Most followed user')
plt.show()

#create the diagram, most used tags
MyColumns=['hashtag','cnt']
hashtagcount=pd.DataFrame(columns=MyColumns)
for index,row in df.iterrows():
    if '#' in row['tweets']:
        words=row['tweets'].split()
        for words in words:
            if words[0]=='#':
                hashtagcount.loc[len(hashtagcount)]=[words,1]
hashtags=hashtagcount.groupby(['hashtag']).count().reset_index()
hashtags=hashtags.sort_values(by='cnt').tail(10)
x=hashtags['hashtag']
y=hashtags['cnt']
plt.xlabel('hashtag')
plt.ylabel('Number of times used')
plt.title('Most number of hashtags used')
plt.xticks(range(10),x,rotation=60)
plt.bar(range(10),y,label='Most hashtags used')
plt.show()


plt.style.use('fivethirtyeight')
df=pd.read_csv(r'C:\Users\dioni\Downloads\tweets.csv')

#displays the count of all elements
df.columns=['Name','UName','Desc','Location','Followers','NStatus','Time','Tweets']
df['Mentions']=df.Tweets.str.count('@')
df.index.name='row_num'
print(df.count())

#create a diagram with the number of statuses and the number of followers
fig,ax=plt.subplots(figsize=(14,6))
ax.hist(df.NStatus,bins=np.arange(0,18000,500),label='Status')
ax.hist(df.Followers,bins=np.arange(0,18000,200),alpha=0.9)
plt.legend()
ax.set_xlabel('Followers')
plt.show()
