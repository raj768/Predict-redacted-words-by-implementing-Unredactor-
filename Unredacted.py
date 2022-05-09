#!/usr/bin/env python
# coding: utf-8

# In[940]:


import pandas as pd
import re
import nltk
import warnings
warnings.filterwarnings("ignore")
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score


# ## Reading the data

# In[941]:


def read_data(filename):
    data = pd.read_csv(filename,sep='\t',on_bad_lines='skip',header=None)
    data = data.loc[~data[1].isin(['teating', 'training  Langdon','testing Camilla','training  Selander  ████████ keeps the action moving ahead at a full gallop.'])]
    df = data.rename(columns = {0 : "github_username",1: "file_type",2 : "entity_name",3:"redaction_context"},inplace = False)
    return df


# In[942]:


df = read_data('unredactor.tsv')


# In[943]:


df


# ## Processing Redacted text

# In[944]:


def preprocessed(data):
    preprocessed = []
    for text in data.values:
        text=re.sub(r'n\'t',' not',str(text))
        text=re.sub(r"can't","can not",text)
        text=re.sub("n\'t","not",text)
        text=re.sub("\'re","are",text)
        text=re.sub("\'s"," ",text)
        text=re.sub("\'d"," would",text)
        text=re.sub("\'ll"," will",text)    
        text=re.sub("\'t"," not",text)
        text=re.sub("\'ve"," have",text)
        text=re.sub("\'m"," am",text)
        text = text.replace('<br>','')
        text = text.replace('</br>','')
        text = text.replace('-',' ')
        text = re.sub('[?|!|\'|"|#]',r'',text)
        text = re.sub('[.|,|)|(|\|/]',r'',text)
        text = ' '.join(e for e in text.split())    
        preprocessed.append(text)
    return preprocessed


# In[945]:


processed_data = preprocessed(df['redaction_context'])


# In[946]:


df['preprocessed_redeacted_context'] = processed_data


# In[947]:


df


# ## Processing of Entity Name column

# In[948]:


def preprocessed_label(data):
    label = []
    for text in data.values: 
        text=re.sub("\'s"," ",str(text))
        text=text.replace('.','')
        label.append(text)
    return label


# In[949]:


label_in = preprocessed_label(df['entity_name'])


# In[950]:


df['entity_name_label'] = label_in


# ## Extract number of letter in redacted words and adding as a feature

# In[951]:


def number_of_letter(data):
    count_letter = []
    for i in data.values:
        a = len(list(filter(str.isalpha, i)))
        count_letter.append(a)
    return count_letter


# In[952]:


c =number_of_letter(df['entity_name_label'])


# In[953]:


df['count_of_letter_of_redeacted words'] = c


# ## Extract number of spaces in redacted words and adding as another feature

# In[954]:


def number_of_spaces(data):
    count = 0
    for i in data:
        for j in i:
            if(i.isspace()):
                count=count+1
    return count


# In[955]:


df['count_number of_spaces'] = df['entity_name_label'].apply(lambda c:number_of_spaces(c))


# ## Dropping redundant columns

# In[956]:


def drop_unnecessary_columns(data):
    data = data.drop(['entity_name','github_username','redaction_context'],axis=1, inplace=False)
    return data    


# In[957]:


df = drop_unnecessary_columns(df)


# In[958]:


df


# ## Splitting into train and test and also create a copy of the test data for displaying the output later

# In[959]:


def split_into_train_val_test(data):
    X_train=data[data['file_type']=='training'].drop(['entity_name_label'],axis=1)
    y_train=data[data['file_type']=='training']['entity_name_label']
    X_val=data[data['file_type']=='validation'].drop(['entity_name_label'],axis=1)
    y_val=data[data['file_type']=='validation']['entity_name_label']
    X_test=data[data['file_type']=='testing'].drop(['entity_name_label'],axis=1)
    y_test=data[data['file_type']=='testing']['entity_name_label']
    X_test_out = X_test.copy()
    return (X_train,y_train,X_val,y_val,X_test,y_test,X_test_out)


# In[960]:


X_train,y_train,X_val,y_val,X_test,y_test,X_test_out=split_into_train_val_test(df)


# In[961]:


y_train.shape


# ## Featurization with n-gram for train,validation and test dataset

# In[962]:


def get_featurization_n_gram(data1,data2,data3):
    vect = CountVectorizer()
    vect.fit(data1.values)
    X_train_redeacted_context = vect.transform(data1.values)
    X_val_redeacted_context = vect.transform(data2.values)
    X_test_redeacted_context = vect.transform(data3.values)
    return (X_train_redeacted_context,X_val_redeacted_context,X_test_redeacted_context)


# In[963]:


X_train_redeacted_context,X_val_redeacted_context, X_test_redeacted_context= get_featurization_n_gram(X_train['preprocessed_redeacted_context'],X_val['preprocessed_redeacted_context'],X_test['preprocessed_redeacted_context'])


# ## Featurization with total number letters in redacted words for train,validation and test dataset

# In[964]:


def get_featurization_count_letter(data1,data2,data3):
    X_train_count_of_letter_of_redeacted = data1.values.reshape(-1,1)
    X_val_count_of_letter_of_redeacted= data2.values.reshape(-1,1)
    X_test_count_of_letter_of_redeacted= data3.values.reshape(-1,1)
    return (X_train_count_of_letter_of_redeacted,X_val_count_of_letter_of_redeacted,X_test_count_of_letter_of_redeacted)


# In[965]:


X_train_count_of_letter_of_redeacted,X_val_count_of_letter_of_redeacted,X_test_count_of_letter_of_redeacted = get_featurization_count_letter(X_train['count_of_letter_of_redeacted words'],X_val['count_of_letter_of_redeacted words'],X_test['count_of_letter_of_redeacted words'])


# ## Featurization with total number of spaces in redacted words for train,validation and test dataset

# In[966]:


def get_featurization_count_spaces(data1,data2,data3):
    X_train_count_number_of_spaces = data1.values.reshape(-1,1)
    X_val_count_number_of_spaces= data2.values.reshape(-1,1)
    X_test_count_number_of_spaces= data3.values.reshape(-1,1)
    return (X_train_count_number_of_spaces,X_val_count_number_of_spaces,X_test_count_number_of_spaces)


# In[967]:


X_train_count_number_of_spaces,X_val_count_number_of_spaces,X_test_count_number_of_spaces =  get_featurization_count_spaces(X_train['count_number of_spaces'],X_val['count_number of_spaces'],X_test['count_number of_spaces'])


# In[968]:


def merge_features(data1,data2,data3):
    data  = hstack((data1,data2,data3)).tocsr()
    return data


# In[969]:


X_tr = merge_features(X_train_redeacted_context,X_train_count_of_letter_of_redeacted,X_train_count_number_of_spaces)


# In[970]:


X_cv = merge_features(X_val_redeacted_context,X_val_count_of_letter_of_redeacted,X_val_count_number_of_spaces)


# In[971]:


X_te = merge_features(X_test_redeacted_context,X_test_count_of_letter_of_redeacted,X_test_count_number_of_spaces)


# ## Model Prediction and Evaluation of model using Precision, Recall and F1-score

# In[972]:


def model_evaluation(data1,data2,data3,y_train):
    clf = DecisionTreeClassifier(random_state=0)
    model = clf.fit(data1,y_train)
    y_train_predicted=model.predict(data1)
    y_val_predicted=model.predict(data2)
    y_test_predicted=model.predict(data3)
    train_precision_score=precision_score(y_train, y_train_predicted,average='weighted')
    val_precision_score=precision_score(y_val, y_val_predicted,average='weighted')
    train_recall_score=recall_score(y_train, y_train_predicted,average='weighted')
    val_recall_score=recall_score(y_val, y_val_predicted,average='weighted')
    train_f1_score=f1_score(y_train, y_train_predicted,average='weighted')
    val_f1_score=f1_score(y_val, y_val_predicted,average='weighted')
    return (y_test_predicted,train_precision_score,val_precision_score,train_recall_score,val_recall_score,train_f1_score,val_f1_score)   


# In[973]:


y_test_predicted,train_precision_score,val_precision_score,train_recall_score,val_recall_score,train_f1_score,val_f1_score= model_evaluation(X_tr,X_cv,X_te,y_train)


# In[974]:


y_test_predicted.shape


# ## Creating a column in the test dataset and displaying unredacted text(after prediction) in that column

# In[975]:


def show_output(data,y_pred):
    data['unredeacted'] = y_pred
    return data


# In[976]:


show_output(X_test_out,y_test_predicted)


# In[ ]:





# In[ ]:




