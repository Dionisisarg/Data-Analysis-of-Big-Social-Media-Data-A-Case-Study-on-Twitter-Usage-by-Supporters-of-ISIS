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

#I will preprocess all tweets to lowercase, remove stopwords such as the, in etc and also stem words.
#Also, i will try to separate hashtags to individual words wherever possible.
def preprocess(tweet):
    tweet=re.sub(r'ENGLISH TRANSLATION:','',tweet) #A number of tweets start with ENGLISH TRANSLATIONS:so i will remove it
    tweet=re.sub(r'[^A-Za-z# ]','',tweet) #I will also strip tweets of non-alphabetic characters except (#)
    words=tweet.strip().split()
    hashtags=[word for word in words if re.match(r'#',word)!=None]
    words=[word.lower() for word in words if word not in hashtags]

#remove stopwords and stem words using porter stemmer
    p_stem=PorterStemmer()
    words=[p_stem.stem(word.lower())for word in words if word not in stopwords.words('english')]
    for hashtag in hashtags:
        hashtag=re.sub(r'#',hashtag,'')
        words_tag=[]
        current_word=''
        for a in hashtag:
            if a.isupper() and current_word!='':
                words_tag.append(current_word)
                current_word='' + a.lower()
            else:
                current_word=current_word + a.lower()

        words_tag.append(current_word)
        words.extend(words_tag)
    words=list(set(words))
    return words
data['wordlist']=[preprocess(tweet) for tweet in data['tweets']] #using the above function, i will add another column "wordlist" to the dataframe

#plot of frequency of varius words used in tweets
all_words=[word for wordlist in data['wordlist'] for word in wordlist]
length_all=len(all_words)
wordcount=dict([(word,all_words.count(word)) for word in set(all_words)])
print(length_all)

#plotting the top 20 most frequent words
wordcount=sorted(wordcount.items(),key=operator.itemgetter(1))
wordcount.reverse()
wordcount=wordcount[2:]#since first two words are '' and 'rt'
top20=wordcount[:20]
top20_words=[word for (word,count) in top20]
top20_freq=[count for (word,count) in top20]
indexes=np.arange(len(top20_words))
width=0.7
#make the diagram
plt.figure(figsize=(15,15))
plt.bar(indexes,top20_freq,width)
plt.xticks(indexes+width/2,top20_words)
plt.show()

#location analysis
unique_locations=data['location'].unique()
unique_counts=dict([(loc,list(data['location']).count(loc)) for loc in unique_locations])
unique_counts=sorted(unique_counts.items(),key=operator.itemgetter(1))
unique_counts.reverse()
for(loc,counts) in unique_counts:
    print(loc,counts)

#subject of tweet analysis using pos tagging
def tweet_subject(tweet):
    tweet=re.sub('ENGLISH TRANSLATION:','',tweet)
    tweet=re.sub('ENGLISH TRANSLATIONS:','',tweet)
    tokenized=nltk.word_tokenize(tweet.lower())
    tagged=nltk.pos_tag(tokenized)
    nouns=[(word)for (word,tag) in tagged if re.match(r'NN',tag)!=None]
    return nouns
data['tweet_subjects']=[tweet_subject(tweet) for tweet in data['tweets']]

#most frequent subjects
all_subjects=[word for wordlist in data['tweet_subjects']for word in wordlist]
all_subjects_counts=dict([(word,all_subjects.count(word)) for word in set(all_subjects)])
all_subjects_counts=sorted(all_subjects_counts.items(),key=operator.itemgetter(1))
all_subjects_counts.reverse()
print('TOTAL UNIQUE SUBJECTS:',len(all_subjects_counts))
for(a,b) in all_subjects_counts[:30]:
    print(a,b)

#plotting the top 20 ,ost frequent subjects
top20_sub=all_subjects_counts[:20]
top20_words=[word for (word,count) in top20_sub]
top20_freq=[count for (word,count) in top20_sub]
indexes=np.arange(len(top20_words))
width=0.7
#make the diagram
plt.figure(figsize=(20,20))
plt.bar(indexes,top20_freq,width)
plt.xticks(indexes+width/2,top20_words)
plt.show()
