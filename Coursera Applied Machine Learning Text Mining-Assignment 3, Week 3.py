#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#In this assignment you will explore text message data and create models to predict if a message is spam or not.

import pandas as pd
import numpy as np

spam_data = pd.read_csv('assets/spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[ ]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# In[ ]:


# Question 1

def answer_one():
    return (len(spam_data[spam_data['target']==1])*100)/len(spam_data['target'])

answer_one()


# In[ ]:


# Question 2
from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    vectorizer=CountVectorizer()
    X=vectorizer.fit(X_train)
    return ' '.join([i for i in list(vectorizer.get_feature_names_out()) if len(i)>30])

answer_two()


# In[ ]:


# Question 3
#Use fit and transform seperately

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer

def answer_three():
    vectorizer=CountVectorizer()
    X_train_vectorized=vectorizer.fit_transform(X_train)
    MB_classifier=MultinomialNB(alpha=0.1).fit(X_train_vectorized.toarray(),y_train)
    X_test_vectorized=vectorizer.transform(X_test).toarray()
    predictions=MB_classifier.predict_proba(X_test_vectorized)
    return roc_auc_score(y_test,predictions[:, 1])

answer_three()


# In[ ]:


# Question 4

from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    Tfid=TfidfVectorizer()
    X_train_transformed=Tfid.fit(X_train)
    X_train_transformed=Tfid.transform(X_train)
    token_names=Tfid.get_feature_names_out()
    X_train_tran_df=pd.DataFrame(X_train_transformed.toarray(), columns=token_names)
    max_list=list(zip(X_train_tran_df.max().index,X_train_tran_df.max()))
    smallest_tf_idf= sorted(max_list, key = lambda x: x[1], reverse=False)[:20]
    biggest_tf_idf=sorted(max_list, key = lambda x: x[1], reverse=False)[-20:]
    biggest_tf_idf=sorted(biggest_tf_idf, key=lambda x:x[1], reverse=True)
    return list(zip(smallest_tf_idf,biggest_tf_idf))

answer_four()


# In[ ]:


# Question 5

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_five():
    Tfid=TfidfVectorizer(min_df=3)
    X_train_transformed=Tfid.fit(X_train)
    X_train_transformed=Tfid.transform(X_train)
    model=MultinomialNB(alpha=0.1).fit(X_train_transformed,y_train)
    predictions=model.predict_proba(Tfid.transform(X_test))        
    return roc_auc_score(y_test,predictions[:, 1])

answer_five()


# In[ ]:


# Question 6

def answer_six():
    spam_mean=len(spam_data[spam_data['target']==1]['text'].sum())/len(spam_data[spam_data['target']==1])
    not_spam_mean=len(spam_data[spam_data['target']==0]['text'].sum())/len(spam_data[spam_data['target']==0])
    return not_spam_mean,spam_mean
answer_six()


# In[ ]:


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# In[ ]:


# Question 7

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def answer_seven():
    Tfidf=TfidfVectorizer(min_df=5)
    X_tran_transformed=Tfidf.fit_transform(X_train)
    
    len_doc=[]
    
    for i in X_train:
        len_doc.append(len(i))
        
    token_names=np.append(Tfidf.get_feature_names_out(), 'len')
    
    X_train_with_len=pd.DataFrame(add_feature(X_tran_transformed,len_doc).toarray(), columns= token_names)
    
    SV_clf=SVC(C=10000)
    model=SV_clf.fit(X_train_with_len,y_train)
    
    len_for_X_test=[]
    
    for i in X_test:
        len_for_X_test.append(len(i))
        
    predictions= model.decision_function(add_feature(Tfidf.transform(X_test),len_for_X_test).toarray())
    
    return roc_auc_score(y_test,predictions)

answer_seven()


# In[ ]:


# Question 8

import re
def answer_eight():
    spam_data_spam=spam_data[spam_data['target']==1]['text']
    spam_data_not_spam=spam_data[spam_data['target']==0]['text']
    return len(re.findall(r'\d', spam_data_not_spam.sum()))/len(spam_data_not_spam),len(re.findall(r'\d', spam_data_spam.sum()))/len(spam_data_spam)

answer_eight()


# In[ ]:


# Question 9

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def answer_nine():
    Tfidf=TfidfVectorizer(min_df=5, ngram_range=(1,3))
    lr=LogisticRegression(C=100, max_iter=1000)
    
    X_train_transformed=Tfidf.fit(X_train)
    X_train_transformed=Tfidf.transform(X_train)
    
    len_each_doc=[]
    
    for i in X_train:
        len_each_doc.append(len(i))
        
    digits=[] 
    
    for i in X_train.str.findall(r'\d'):
        digits.append((len(i)))
        
    token_names=np.append(np.append(Tfidf.get_feature_names_out(), 'len'),'digit')
    X_train_final= pd.DataFrame(add_feature(add_feature(X_train_transformed,len_each_doc),digits).toarray(), columns=token_names)
    
    len_each_X_test=[]
    
    for i in X_test:
        len_each_X_test.append(len(i))
        
    digits_X_test=[]
    
    for i in X_test.str.findall(r'\d'):
        digits_X_test.append((len(i)))
    
    X_test_final= pd.DataFrame(add_feature(add_feature(Tfidf.transform(X_test),len_each_X_test),digits_X_test).toarray())  
    
    model=lr.fit(X_train_final,y_train)
    predictions=model.predict_proba(X_test_final)
    
            
    return roc_auc_score(y_test,predictions[:,1])

answer_nine()


# In[ ]:


# Question 10

import re

def answer_ten():
    spam_data_spam=spam_data[spam_data['target']==1]['text']
    spam_data_notspam=spam_data[spam_data['target']==0]['text']
    return len(re.findall(r'\W',spam_data_notspam.sum()))/len(spam_data_notspam),len(re.findall(r'\W',spam_data_spam.sum()))/len(spam_data_spam)
answer_ten()


# In[ ]:


# Question 11

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import re

X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)

def extract_features(data):
    length_of_doc = [len(doc) for doc in data]
    digit_count = [len(re.findall(r'\d', doc)) for doc in data]
    non_word_char_count = [len(re.findall(r'\W', doc)) for doc in data]
    return length_of_doc, digit_count, non_word_char_count


def answer_eleven():
    Cnt_vec=CountVectorizer(min_df=5, ngram_range=(2,5),analyzer='char_wb')
    X_train1=Cnt_vec.fit(X_train)
    X_train1=Cnt_vec.transform(X_train)
    
    extra_features=['length_of_doc','digit_count','non_word_char_count']
    token_names=np.append(Cnt_vec.get_feature_names_out(),extra_features)
    
    X_train_trans=pd.DataFrame(add_feature(add_feature(add_feature(X_train1,extract_features(X_train)[0]),extract_features(X_train)[1])
                                       ,extract_features(X_train)[2]).toarray(),columns=token_names).iloc[:2000,:]
    
    lr=LogisticRegression(C=100,max_iter=1000)
    model=lr.fit(X_train_trans,pd.DataFrame(y_train).iloc[:2000,:])
    
    X_test1=Cnt_vec.transform(X_test)
    X_test_ready=pd.DataFrame(add_feature(add_feature(add_feature(X_test1,extract_features(X_test)[0]),extract_features(X_test)[1])
                                      ,extract_features(X_test)[2]).toarray(),columns=token_names)
    
    predictions=model.predict_proba(X_test_ready)
 
    auc_score= roc_auc_score(y_test,predictions[:,1])
    coef= [x for sublist in model.coef_ for x in sublist]
    coef_names = list(zip(token_names, coef))
    sorted_coef = sorted(coef_names, key=lambda x: x[1])
    smallest_coef = sorted_coef[:10]
    largest_coef = sorted_coef[-10:][::-1]

    return auc_score, smallest_coef, largest_coef
answer_eleven()

