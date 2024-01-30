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
from gensim import corpora, models, similarities, matutils
from gensim.corpora.dictionary import Dictionary
from sklearn.cluster import DBSCAN
from sklearn.decomposition import NMF, PCA
from collections import Counter, defaultdict, OrderedDict

plt.style.use('ggplot')  # using the style for the plot
df = pd.read_csv(r'C:\Users\dioni\Downloads\tweets.csv')
predicted_languages = [langid.classify(tweet) for tweet in df['tweets']]  #language detection
lang_df = pd.DataFrame(predicted_languages, columns=['language', 'values'])#DataFrame is a 2-dimensional labeled data structured with columns of different types
print(lang_df['language'].value_counts().head(10))

colors = sns.color_palette('hls', 10)#used for coloring the plot
pd.Series(lang_df['language']).value_counts().head(10).plot(kind='bar', color=colors, fontsize=14, rot=45, title='The 10 most common languages', figsize=(12, 9))#make the diagram

plt.show()
