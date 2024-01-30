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

index=[]
for i in range(len(df['tweets'])):
    if '@'in df['tweets'][i]:
        index.append(i)

With_mention=df.iloc[index,:].reset_index(drop=True)
With_mention['Tagged_User']=With_mention['tweets'].apply(lambda x:re.findall(r'@([A-Za-z0-9_]+)',str(x)))
User=With_mention['username'].unique()
With_mention['Tagged_User_Co']=With_mention['Tagged_User'].apply(lambda x:list(set(x).intersection(User)))
With_mention['Co_length']=With_mention['Tagged_User_Co'].apply(lambda x:len(x))
With_mention_2=With_mention[With_mention['Co_length']>0].reset_index(drop=True)

#we first create a dataframe which contains the information of individual tagging activities in each row.
for i in range(len(With_mention['tweets'])):
    frame=With_mention.iloc[i,:]
    for j in range(len(frame['Tagged_User'])):
        tmp=pd.DataFrame({'User':[frame['username']],'Mentions':[frame['Tagged_User'][j]],'Time':[frame['time']],'User_numberstatuses':[frame['numberstatuses']],'User_followers':[frame['followers']],'Weight':[1]})
        if i==0 and j==0:
            Mention_net=tmp
        else:
            Mention_net=Mention_net._append(tmp,ignore_index=True)

#Then, in order to be more prepare for some extra analysis, a similar dataframe will be created that only contains the interactivity between users in the dataframe.
for i in range(len(With_mention_2['tweets'])):
    frame=With_mention_2.iloc[i,:]
    for j in range(len(frame['Tagged_User_Co'])):
        Mentioned_follower=list (df[df['username']==frame['Tagged_User_Co'][j]]['followers'])[0]
        Mehtioned_statuses=list (df[df['username']==frame['Tagged_User_Co'][j]]['numberstatuses'])[0]
        tmp=pd.DataFrame({'Mentioned_statuses':[Mehtioned_statuses],'Mentioned_followers':[Mentioned_follower],
                          'User':[frame['username']],'Mentions':[frame['Tagged_User_Co'][j]],'Time':[frame['time']],'User_numberstatuses':[frame['numberstatuses']],
                          'User_followers':[frame['followers']],'Weight':[1]})
        if i==0 and j==0:
            Mention_net_2=tmp
        else:
            Mention_net_2=Mention_net_2._append(tmp,ignore_index=True)

Mention_net=Mention_net[Mention_net['User']!=Mention_net['Mentions']]
Mention_net_2=Mention_net_2[Mention_net_2['User']!=Mention_net_2['Mentions']]
Mention_net=Mention_net.reset_index(drop=True)
Mention_net_2=Mention_net_2.reset_index(drop=True)

#A glimpse to the new dataframe
pd.set_option('display.max_columns', None)
p=Mention_net
print(p)

#It is insightful to see how many times a user mentions others and how many times a user is mentioned by others.
#For targeting the potential terrorist, in and out degree on social media may be a helpful indicators.
In_degree=Mention_net.groupby(by=['Mentions'],as_index=False)['Weight'].sum()
Out_degree=Mention_net.groupby(by=['User'],as_index=False)['Weight'].sum()
In_degree=pd.DataFrame(In_degree).sort_values(by='Weight',ascending=False).reset_index(drop=True)
Out_degree=pd.DataFrame(Out_degree).sort_values(by='Weight',ascending=False).reset_index(drop=True)
print('Most mentioned user is ' + str(In_degree['Mentions'][0])+ ' with ' +str(In_degree['Weight'][0])+' times mentioned by other users.')
print( ' Most active user is ' + str(Out_degree['User'][0])+ ' with ' +str (Out_degree['Weight'][0])+' times mentioning other users.')

#Similarly, in the seconddataframe where all users and tagged users are in the original data set, we can also see name of most active and mentioned user.
In_degree_2=Mention_net_2.groupby(by=['Mentions'],as_index=False)['Weight'].sum()
Out_degree_2=Mention_net_2.groupby(by=['User'],as_index=False)['Weight'].sum()
In_degree_2=pd.DataFrame(In_degree_2).sort_values(by='Weight',ascending=False).reset_index(drop=True)
Out_degree_2=pd.DataFrame(Out_degree_2).sort_values(by='Weight',ascending=False).reset_index(drop=True)
print('Most mentioned user is '+str(In_degree_2['Mentions'][0])+' with ' +str(In_degree['Weight'][0])+' times mentioned by other users.')
print('Most active user is '+str(Out_degree_2['User'][0])+' with '+str(Out_degree['Weight'][0])+' times mentioning other users.')      
