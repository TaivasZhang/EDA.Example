#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns 
import matplotlib as mpl 


# In[7]:


plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['figure.dpi'] = 300


# In[1]:


original_data = pd.read_csv('reviews_merged.csv', encoding='gbk')
original_data.head(5)


# In[2]:


# Change string to dictionary
import ast
most_asin = original_data['title'].tolist()
most_asin


# In[4]:


# Get the mean price of each asin and fillin the empty data
diffasin = list(set(original_data['asin']))
all_length = len(original_data)
meanprice = []
for asin in diffasin:
    meanprice.append(original_data[original_data['asin']  == asin]['price'].mean())


# In[8]:


meanpriceint = []
for i in meanprice:
    if i is np.nan:
        i = 0
    meanpriceint.append(i)
while 0 in meanpriceint:
    meanpriceint.remove(0)


# In[17]:


mpl.rc("figure", figsize=(16,9))  
sns.set_palette("hls") 
sns.distplot(meanpriceint,color="r",bins=40, kde_kws={"color":"r", "lw":1 }, hist_kws={ "color": "black" })
plt.savefig('meanprice_hist.png',  bbox_inches = 'tight')
plt.show()


# In[11]:


original_data['title'].value_counts()


# In[12]:


# Mean overall of each reviewer
diffreviewerID = list(set(original_data['reviewerID']))
meanstar = []
for reviewerID in diffreviewerID:
    meanstar.append(original_data[original_data['reviewerID']  == reviewerID]['overall'].mean())


# In[16]:


mpl.rc("figure", figsize=(16,9)) 
sns.set_palette("hls") 
sns.distplot(meanstar,color="r",bins=40, kde_kws={"color":"red", "lw":1 }, hist_kws={ "color": "black" })
plt.savefig('meanstar_hist.png', bbox_inches = 'tight')
plt.show()


# In[18]:


# Word Cloud
import re

reviewtext= original_data['summary'].tolist()
text_word = []
for word in reviewtext:
    each_review = re.split('[, ]', word)
    for each in each_review:
        text_word.append(each)


# In[19]:


# Plot the wordcloud
from wordcloud import WordCloud

f = (' '.join(text_word))
wordcloud = WordCloud(
        background_color="white",
        width=1920,              
        height=1080,             
        margin=10               
        ).generate(f)
plt.imshow(wordcloud)
plt.axis("off")
wordcloud.to_file('review_text.png')
plt.show()


# In[60]:


from collections import Counter
import pylab as pl

# Get the first three words from title and link it with asin
title_list = original_data['title'].tolist()
product_name = []
for title in title_list:
    temp = str(title).split(' ')[0:3]
    key_name = (' '.join(temp))
    product_name.append(key_name)

original_data['product_name'] = product_name

# Look at the top 15 most popular products
mostreview_productname = Counter(original_data['product_name']).most_common(15)
name = []
freq = []
for i in mostreview_productname:
    name.append(i[0])
    freq.append(i[1])

a = plt.bar(name, freq, color=['#a3a1a1','#333030'],alpha=0.8)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2-0.5, 1.01*height, '%s' % int(height))

pl.xticks(rotation=60)
autolabel(a)
plt.ylabel('Frequence', fontsize=12) 
plt.xlabel('Product Names', fontsize=12) 
plt.savefig('reviewfreq.png', bbox_inches = 'tight')


# In[35]:


# Transform helpful to numerical and calculate the ratio
help_list = original_data['helpful'].tolist()
HaveSeen = []
feelnice = []
for i in help_list:
    HaveSeen.append(i[4])# Capture number of viewers   
    feelnice.append(i[1])# Capture number of helpfuls
helpful_rate = []
for i in range(len(HaveSeen)):
    if HaveSeen[i] != '0' and HaveSeen[i] != ' ' and HaveSeen[i] != ',':
        if int(HaveSeen[i]) > 5:
            rate = int(feelnice[i]) / int(HaveSeen[i])
            helpful_rate.append(rate)

print(helpful_rate)


# In[38]:


mpl.rc("figure", figsize=(16,9)) 
plt.xlim((0,1))
sns.set_palette("hls")
sns.distplot(helpful_rate,color="r",bins=50, kde_kws={"color":"red", "lw":1 }, hist_kws={ "color": "black" })
plt.savefig('helpfulrate_hist_5.png')
plt.show()


# In[39]:


# Use the summary to do a sentiment analysis
# First calculate the score of sentiments
from textblob import TextBlob
summary = original_data['summary'].tolist()
score = []
for text in summary:
    blob = TextBlob(text)
    score.append(blob.sentiment.polarity)


# In[45]:


# calculate the mean senitment score of each product
# and then store it in dictionary
original_data['score'] = score
keyname = list(set(original_data['product_name']))
meanscore = dict()
for i in keyname:
    newdf = original_data[original_data['product_name'] == i]
    eachmeanscore = newdf['score'].mean()
    meanscore.update({i :eachmeanscore})


# In[56]:


# Randomly select 30 products in asin and check the score
import random
a = meanscore.keys()
a = list(a)
sample_keyname = random.sample(a, 30)

b = []
for i in sample_keyname:
    b.append(meanscore[i])


# In[61]:


# Plot the scores of the selected product
import pylab as pl

plt.figure(figsize=(16,9))
plt.bar(sample_keyname, b, color=['#a3a1a1','#333030'], alpha = 0.8)
#plt.grid(True)
pl.xticks(rotation=90)
plt.savefig('sample_asin_SentimentScore.png',bbox_inches = 'tight')
plt.show()


# In[ ]:




