# Data-Analysis-of-Big-Social-Media-Data-A-Case-Study-on-Twitter-Usage-by-Supporters-of-ISIS
This dissertation includes an approach to the process of analyzing large volumes of data in social media, specifically focusing on Twitter. Subsequently, an analysis of a dataset  under study is conducted, which contains all the tweets from supporters of Isis extremists, and significant knowledge is extracted from it through appropriate algorithms.

After the workspace environment has been prepared, the introductory code (data(1)) is written to load the necessary libraries 
and read the data file. (Figure 1.1) is the result of executing the command "data.info()" in the code (data(1))."

One of the most valuable conclusions that can be drawn from the dataset studied in this thesis is languages chosen by extremists
to generate content in their attempt to approach a new audience and spread their ideas. Algorithm (language(2)). The execution 
of the algorithm returns as a result languages used by extremists in their tweets as well as usage metrics of these languages. 
(Figure 2.1, 2.2).

The most significant conclusion that can be drawn from the dataset studied in the context of this thesis is the visualization 
of words that appear most frequently in the file. For this purpose, the Word Cloud tool is utilized, which visualizes words of
a text in a manner where the most frequently used words appear larger in font size. The algorithm used is (WordCloud(3)), and 
the result of this algorithm is represented by (Figure 3.1). Another conclusion that can be drawn from the same algorithm is 
topics discussed by the 10 most active users of the ISIS network on Twitter. Initially, the 10 most 'sensitive' topics that 
Twitter users engage with are displayed (Figure 3.2) Subsequently, after these topics have been displayed, a chart is created 
illustrating the 10 most active users and the topics each user is primarily engaged with. (Figure 3.3).

The following is the algorithm for visualizing published tweets over a period of time, (visualization of tweets over a period of time(4))
and its results are Figures (4.1, 4.2, and 4.3). Finally the preceding algorithm also displays which day of the week do 
Isis supporters post more tweets (Figure 4.4).


An important piece of information that emerges from the algorithm (analysis users of twitter(5)) is the total number of tweets contained 
within the processed file, as well as the count of unique tweets within that file (Figure 5.1 and 5.2). Subsequently, the 
algorithm presents the 5 most frequent users characterized as senders and the 5 most frequent users referred to as receivers. 
Senders are users who create tweets and use "@" to mention other users. Receivers are users with the greatest influence, 
receiving the most mentions of their names from other users (Figure 5.3). Additionally, regarding the Receivers, there is 
also a reference to their descriptions, so it can be perceived that they are mostly "impartial" news websites (Figure 5.4). 
Finally, there is a mapping of the "most tweeted" into a graph. Users have been categorized into three groups:
1. Only senders-red
2. Only receivers-blue
3. Senders and receivers-green
The next step involves mapping the edges between nodes where users have tweeted among themselves. The thickness of the edge 
indicates how many communications have been made. The weight indicates the number of tweets sent between two connected users (Figure 5.5).

Before executing the algorithm (most common words(6)), which will calculate the words with the highest frequency of occurrence, 
preprocessing is required on the dataset.
