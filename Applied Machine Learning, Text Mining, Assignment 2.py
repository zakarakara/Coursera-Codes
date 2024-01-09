#!/usr/bin/env python
# coding: utf-8

# In[2]:


# In part 1 of this assignment you will use nltk to explore the CMU Movie Summary Corpus. 
# All data is released under a Creative Commons Attribution-ShareAlike License. 
# Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling.


# In[ ]:


import nltk
import pandas as pd
import numpy as np

nltk.data.path.append("assets/")

# If you would like to work with the raw text you can use 'plots_raw'
with open('assets/plots.txt', 'rt', encoding="utf8") as f:
    plots_raw = f.read()

# If you would like to work with the plot summaries in nltk.Text format you can use 'text1'.
plots_tokens = nltk.word_tokenize(plots_raw)
text1 = nltk.Text(plots_tokens)


# In[ ]:


# Question 1

def answer_one():
    return len(set(nltk.word_tokenize(plots_raw)))/len(nltk.word_tokenize(plots_raw))
answer_one()


# In[ ]:


# Question 2

from nltk.probability import FreqDist

def answer_two():
    dist = FreqDist(text1)
    return (((dist['love'] + dist['Love'])*100)/float(len(nltk.word_tokenize(plots_raw))))

answer_two()


# In[ ]:



# Question 3
from nltk.probability import FreqDist

def answer_three():
    dist = FreqDist(text1)
    return sorted(list(dict(dist).items()),key=lambda x:x[1], reverse=True)[:20]
answer_three()


# In[ ]:


# Question 4
from nltk.probability import FreqDist

def answer_four():
    dist = FreqDist(text1)
    return sorted([w for w in set(plots_raw.split()) if len(w)>5 and dist[w]>200])
answer_four()


# In[ ]:


# Question 5
def answer_five():

    return ' '.join([w for w in set(plots_raw.split()) if len(w)>33]), len(' '.join([w for w in set(plots_raw.split()) if len(w)>33]))

answer_five()


# In[ ]:


# Question 6
from nltk.probability import FreqDist
import operator

def answer_six():
    import operator
 
    dist  =FreqDist(text1)
    res_lis = {}
    for w in dist.keys() :
        if w.isalpha() and dist[w] > 2000 :
            res_lis[w] = dist[w]    
    sorted_res_list = sorted(res_lis.items(), key=operator.itemgetter(1))
    sorted_res_list.reverse()
    result = [(f,w) for w,f in sorted_res_list]
    return result
answer_six()


# In[ ]:


# Question 7

def answer_seven(): 
    sentences = nltk.sent_tokenize(' '.join(text1))
    return len(text1) / len(sentences)
answer_seven()


# In[ ]:


# Question 8

def answer_eight():
    mydf= pd.DataFrame(pd.DataFrame(nltk.pos_tag(plots_tokens), columns=['col1','pos_tag'])['pos_tag'])
    counts = mydf.groupby('pos_tag').size()
    df=pd.DataFrame(counts).reset_index()
    df.columns=['pos_tag','value']
    df_list=df.sort_values(by=['value'], ascending=False).iloc[0:5,]
    pos_tag=df_list['pos_tag']
    value=df_list['value']

    return list(zip(pos_tag,value))

answer_eight()


# In[ ]:


# Question 9

from nltk.corpus import words
correct_spellings = words.words()

from nltk.util import ngrams
from nltk.metrics.distance import jaccard_distance

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    
    first_letters=[]
    for word in entries:
        first_letters.append(word[0])
        closer_words=[]
        
    for words in correct_spellings:
        for char in first_letters:
            if words.startswith(char):
                closer_words.append(words)
                
    trigram_entries=[]
    for i in entries:
        trigram_entries.append(list(ngrams(i,3)))
        
    trigram_closer_words=[]
    for x in closer_words:
        trigram_closer_words.append(list(ngrams(x,3)))
        
    jaccard_result = []
    for i, closer_trigrams in enumerate(trigram_closer_words):
        for x, entry_trigrams in enumerate(trigram_entries):
            if closer_words[i][0] == entries[x][0]:
                jaccard_result.append((jaccard_distance(set(closer_trigrams), set(entry_trigrams)), closer_words[i], entries[x]))
                    
                    
    b= sorted(jaccard_result, key=lambda x: x[0], reverse=False)[5][1]
    a= sorted(jaccard_result, key=lambda x: x[0], reverse=False)[2][1]
    c= sorted(jaccard_result, key=lambda x: x[0], reverse=False)[0][1]
    
    return [a,b,c] 
    
answer_nine()


# In[ ]:


# Question 10

from nltk.util import ngrams
from nltk.metrics.distance import jaccard_distance


def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
        
    first_letters=[]
    for word in entries:
        first_letters.append(word[0])
        closer_words=[]
        
    for words in correct_spellings:
        for char in first_letters:
            if words.startswith(char):
                closer_words.append(words)
                
    trigram_entries=[]
    for i in entries:
        trigram_entries.append(list(ngrams(i,4)))
        
    trigram_closer_words=[]
    for x in closer_words:
        trigram_closer_words.append(list(ngrams(x,4)))
        
    jaccard_result = []
    for i, closer_trigrams in enumerate(trigram_closer_words):
        for x, entry_trigrams in enumerate(trigram_entries):
            if closer_words[i][0] == entries[x][0]:
                jaccard_result.append((jaccard_distance(set(closer_trigrams), set(entry_trigrams)), closer_words[i], entries[x]))
                
                
    a= sorted(jaccard_result, key=lambda x: x[0], reverse=False)[1][1]
    b= sorted(jaccard_result, key=lambda x: x[0], reverse=False)[2][1]
    c= sorted(jaccard_result, key=lambda x: x[0], reverse=False)[0][1]
    
    return [a,b,c] 
answer_ten()


# In[ ]:


# Question 11

import nltk

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    
    first_letters=[]
    for word in entries:
        first_letters.append(word[0])
    
    closer_words=[]
    for words in correct_spellings:
        for char in first_letters:
            if words.startswith(char):
                closer_words.append(words)
    
    result=[]
    for i, word_close in enumerate(closer_words):
        for x, mis_spelled in enumerate(entries):
            if closer_words[i][0] == entries[x][0]:
                result.append((nltk.edit_distance(mis_spelled,word_close),closer_words[i], entries[x]))
    answer= [sorted(result, key=lambda x:x[0], reverse=False)[0][1], sorted(result, key=lambda x:x[0], reverse=False)[2][1],
    sorted(result, key=lambda x:x[0], reverse=False)[1][1]]
                   
    return answer
answer_eleven()

