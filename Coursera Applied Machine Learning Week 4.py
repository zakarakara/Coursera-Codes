#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor

def engagement_model():
    
    # Reading the datasets
    train_df = pd.read_csv('assets/train.csv').dropna()
    test_df = pd.read_csv('assets/test.csv').dropna()
    
    #Converting engagement column (target variable) into binary
    
    train_df['engagement'] = np.where(train_df['engagement'], '1', '0')
    train_df['engagement'] = pd.to_numeric(train_df['engagement'])
    
    # Finding the variables which has predictive power, i disregard multicollniarity
    # Have selected variables which are more than 10% correlated with target variable
    
    corr_matrix = train_df.corr(method='pearson')['engagement']
    
    train_df = train_df[['document_entropy', 'freshness', 'easiness', 'fraction_stopword_presence', 'silent_period_rate', 'engagement', 'id']]
    test_df = test_df[['document_entropy', 'freshness', 'easiness', 'fraction_stopword_presence', 'silent_period_rate', 'id']]
    
    # dividing data set into train test

    predictions_df = pd.DataFrame({'id': test_df['id']})
    train_set, val_set = train_test_split(train_df, test_size=0.2, random_state=42) 
    
    y_train = train_set['engagement']
    X_train = train_set.drop(['id', 'engagement'], axis=1)
    
    y_val = val_set['engagement']
    X_val = val_set.drop(['id', 'engagement'], axis=1)
    
    # Call the model
    
    model=RandomForestRegressor()
    model.fit(X_train, y_train)
    
    X_test = test_df.drop('id', axis=1)
    y_test_pred_rf_tuned = model.predict(X_test)
    
    ans_rf_tuned = pd.Series(y_test_pred_rf_tuned, index=test_df['id'], name='engagement', dtype='float32')
    return ans_rf_tuned

