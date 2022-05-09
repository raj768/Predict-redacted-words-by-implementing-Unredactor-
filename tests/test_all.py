#!/usr/bin/env python
# coding: utf-8

# In[119]:


import pytest
import Unredacted
import sys
import os
from Unredacted import *


# In[120]:


filename = 'unredactor.tsv'


# In[121]:


def read__test():
    data = Unredacted.read_data(filename)
    return data
    assert data.shape is not None


# In[122]:


df = read__test()


# In[123]:


def preprocessed_test():
    processed = Unredacted.preprocessed(df['redaction_context'])
    return processed
    assert len(processed)!=0        


# In[124]:


processed_data = preprocessed_test()


# In[125]:


df['preprocessed_redeacted_context'] = processed_data


# In[126]:


def preprocessed_label_test():
    label = Unredacted.preprocessed_label(df['entity_name'])
    return label
    assert len(label)!=0        


# In[127]:


label_in = preprocessed_label_test()


# In[128]:


df['entity_name_label'] = label_in


# In[129]:


def number_of_letter_test():
    c = Unredacted.number_of_letter(df['entity_name_label'])
    return c
    assert c!=0 


# In[130]:


c = number_of_letter_test()


# In[131]:


df['count_of_letter_of_redeacted words'] = c


# In[132]:


def number_of_spaces_test():
    count = Unredacted.number_of_spaces(df['entity_name_label'])
    return count
    assert count is not None


# In[133]:


df['count_number of_spaces'] = df['entity_name_label'].apply(lambda c:number_of_spaces_test())


# In[134]:


def drop_unnecessary_columns_test():
    data = Unredacted.drop_unnecessary_columns(df)
    return data
    assert data.shape is not None


# In[135]:


df = drop_unnecessary_columns_test()


# In[136]:


def split_into_train_val_test_testing():
    X_train,y_train,X_val,y_val,X_test,y_test,X_test_out = Unredacted.split_into_train_val_test(df)
    return (X_train,y_train,X_val,y_val,X_test,y_test,X_test_out)
    assert X_train.shape is not None
    assert y_train.shape is not None
    assert X_val.shape is not None
    assert y_val.shape is not None
    assert X_test.shape is not None
    assert y_test.shape is not None
    assert X_test_out.shape is not None


# In[137]:


X_train,y_train,X_val,y_val,X_test,y_test,X_test_out = split_into_train_val_test_testing()


# In[138]:


def get_featurization_n_gram_test():
    X_train_redeacted_context,X_val_redeacted_context,X_test_redeacted_context = Unredacted.get_featurization_n_gram(X_train['preprocessed_redeacted_context'],X_val['preprocessed_redeacted_context'],X_test['preprocessed_redeacted_context'])
    return (X_train_redeacted_context,X_val_redeacted_context,X_test_redeacted_context)
    assert X_train_redeacted_context.shape is not None
    assert X_val_redeacted_context.shape is not None
    assert X_test_redeacted_context.shape is not None


# In[139]:


X_train_redeacted_context,X_val_redeacted_context,X_test_redeacted_context = get_featurization_n_gram_test()


# In[140]:


def get_featurization_count_letter_test():
    X_train_count_of_letter_of_redeacted,X_val_count_of_letter_of_redeacted,X_test_count_of_letter_of_redeacted = Unredacted.get_featurization_count_letter(X_train['count_of_letter_of_redeacted words'],X_val['count_of_letter_of_redeacted words'],X_test['count_of_letter_of_redeacted words'])
    return (X_train_count_of_letter_of_redeacted,X_val_count_of_letter_of_redeacted,X_test_count_of_letter_of_redeacted)
    assert X_train_count_of_letter_of_redeacted.shape is not None
    assert X_val_count_of_letter_of_redeacted.shape is not None
    assert X_test_count_of_letter_of_redeacted.shape is not None


# In[141]:


X_train_count_of_letter_of_redeacted,X_val_count_of_letter_of_redeacted,X_test_count_of_letter_of_redeacted = get_featurization_count_letter_test()


# In[142]:


def get_featurization_count_spaces_test():
    X_train_count_number_of_spaces,X_val_count_number_of_spaces,X_test_count_number_of_spaces = Unredacted.get_featurization_count_spaces(X_train['count_number of_spaces'],X_val['count_number of_spaces'],X_test['count_number of_spaces'])
    return (X_train_count_number_of_spaces,X_val_count_number_of_spaces,X_test_count_number_of_spaces)
    assert X_train_count_number_of_spaces.shape is not None
    assert X_val_count_number_of_spaces.shape is not None
    assert X_test_count_number_of_spaces.shape is not None


# In[143]:


X_train_count_number_of_spaces,X_val_count_number_of_spaces,X_test_count_number_of_spaces = get_featurization_count_spaces_test()


# In[144]:


def merge_features_test_for_train():
    data1 = Unredacted.merge_features(X_train_redeacted_context,X_train_count_of_letter_of_redeacted,X_train_count_number_of_spaces)
    data2 = Unredacted.merge_features(X_val_redeacted_context,X_val_count_of_letter_of_redeacted,X_val_count_number_of_spaces)
    data3 = Unredacted.merge_features(X_test_redeacted_context,X_test_count_of_letter_of_redeacted,X_test_count_number_of_spaces)
    return (data1,data2,data3)
    assert data1.shape is not None
    assert data2.shape is not None
    assert data3.shape is not None  


# In[145]:


X_tr,X_cv,X_te = merge_features_test_for_train()


# In[146]:


def model_evaluation_test():
    y_test_predicted,train_precision_score,val_precision_score,train_recall_score,val_recall_score,train_f1_score,val_f1_score=Unredacted.model_evaluation(X_tr,X_cv,X_te,y_train)
    return (y_test_predicted,train_precision_score,val_precision_score,train_recall_score,val_recall_score,train_f1_score,val_f1_score)
    assert y_test_predicted.shape is not None
    assert train_precision_score>=0
    assert val_precision_score>=0
    assert train_recall_score>=0
    assert val_recall_score>=0
    assert train_f1_score>=0
    assert val_f1_score>=0    


# In[147]:


y_test_predicted,train_precision_score,val_precision_score,train_recall_score,val_recall_score,train_f1_score,val_f1_score = model_evaluation_test()


# In[148]:


def show_output_test():
    data = Unredacted.show_output(X_test_out,y_test_predicted)
    return data
    assert data is not None


# In[149]:


show_output_test()


# In[ ]:





# In[ ]:




