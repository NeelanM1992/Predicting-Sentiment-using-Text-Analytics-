# %load Assignment1_865_Q2_NeelanMuthurajah.py
#!/usr/bin/env python
# coding: utf-8

# In[5]:


# -*- coding: utf-8 -*-
"""
[Neelan, Muthurajah]
[20195484]
[MMA]
[Section 2]
[MMA 865]
[October 18th 2020]


Submission to Question [2], Part [1]
"""
# TODO: import other libraries as necessary


### Import Libraries to Run Code Below
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import unidecode
import re
import textstat
from textblob import TextBlob
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix, make_scorer, f1_score,accuracy_score, cohen_kappa_score, log_loss

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"




### Import Data (Train and Test sets) 
#Import Train and Test Data
df_train = pd.read_csv("sentiment_train.csv")
df_test = pd.read_csv("sentiment_test.csv")

#Add column differentiating train and test data
df_train['Type']='Train'
df_test['Type']='Test'

#Append train and test data together for feature engineering & preprocessing 
df_complete = df_train.append(df_test)

#Drop any rows with NA's 
df_complete=df_complete.dropna()




### Data Exploration
#From a quick preview of the data, it is observed that positive sentiments are labeled as 1 whereas negative sentiments are labeled as 0
df_complete.head(10)

#Train set includes 2400 entries in total 
print(df_train.info())

# Create dataframe showing count of positive sentences vs negative sentences in the training set 
train=df_train.groupby('Polarity')["Polarity"].count()
train=pd.DataFrame(train)
train

#Create dataset
Polarity_Train = ['Postive','Negative'] 
  
Count_Train = [1187,1213]

# Create pie chart showing split of positive sentences vs negative sentences in the training set
plt.pie(Count_Train,labels = Polarity_Train,radius=1,autopct='%0.2f%%')
plt.title("Training Data - 2400 Entries in Total", bbox={'facecolor':'0.8', 'pad':5})

# The plot shows that the data is not imbalanced. There is almost a 1:1 class balance of positive vs negative sentences
plt.show() 

#Test set includes 600 entries 
print(df_test.info())

# Create dataframe showing count of positive sentences vs negative sentences in the test set 
test=df_test.groupby('Polarity')["Polarity"].count()
test=pd.DataFrame(test)
test

#Create dataset
Polarity_Test = ['Postive','Negative'] 
  
Count_Test = [313,287]

# Create plot showing split of positive sentences vs negative sentences in the testing set
plt.pie(Count_Test, labels = Polarity_Test,radius=1,autopct='%0.2f%%') 
plt.title("Test Data - 600 Entries in Total", bbox={'facecolor':'0.8', 'pad':5})

# The plot shows that the data is also not imbalanced in the test set. There is almost a 1:1 class balance of positive vs negative sentences
plt.show() 




### Text Preprocessing
#Create function in order to preprocess textual data by removing stop words, lowering letters, removing digits, lemmatizing
stop_words=set(stopwords.words('english')+stopwords.words('spanish'))
lemmer=WordNetLemmatizer()

def preprocess (x):
    x=x.lower()
    x=re.sub(r'[^\w\s\d+]','',x)
    x=unidecode.unidecode(x)
    x=[lemmer.lemmatize(w) for w in x.split() if w not in stop_words ]
    return ' '.join(x)

#Add a column for the sentence column after pre-processing 
df_complete['Sentence_Clean']=df_complete['Sentence'].apply(preprocess)

#Preview updated dataframe with the sentence clean column
df_complete.head()

#Add other features to the dataset such as number of characters, syllable count, difficulty understanding passage
df_complete['len'] = df_complete['Sentence_Clean'].apply(lambda x:len(x))
df_complete['syllable_count'] = df_complete['Sentence_Clean'].apply(lambda x: textstat.syllable_count(x))

#A metric to indicate how difficult a passage is to understand
df_complete['flesch_reading_case'] = df_complete['Sentence_Clean'].apply(lambda x: textstat.flesch_reading_ease(x))

#Preview the updated dataframe with these added features (length, syllable count, difficulty understanding passage)
df_complete.head()

#Convert 'sentence clean' column to a string datatype 
df_complete['Sentence_Clean'] = df_complete['Sentence_Clean'].astype(str)

#Run each review through textblob to determine overall sentiment
sentiment = []
sentiment2 = []

#Take textblob score and classify sentences into positive or negative based on a threshold
for sentence in df_complete['Sentence_Clean'] :
    sent = TextBlob(sentence)
    if sent.sentiment.polarity > 0.0:
        sentiment.append(1)
    else:
        sentiment.append(0)

#Take textblob score as is and add it to the final dataframe for modeling 
for sentence2 in df_complete['Sentence_Clean'] :
    sent = TextBlob(sentence2)
    if sent.sentiment.polarity > 0.0:
        sentiment2.append(sent.sentiment.polarity)
    else:
        sentiment2.append(sent.sentiment.polarity)
    
# Create 2 columns from the sentiment analysis above. If score was above 0, positive sentiment otherwise 0 for negative sentiment
df_complete['sentiment'] = sentiment
df_complete['sentiment_score'] = sentiment2

#Preview dataframe  
df_complete.head(20)

#Split data back into train and test after pre-processing 
df_train=df_complete[(df_complete.Type=='Train')]
df_test=df_complete[(df_complete.Type=='Test')]

#Create BOW using TF-IDF 
#max_df=if a word is in 99.5% or more of the documents remove it
#min_df=if a word in in fewer than 0.5% of the documents remove it 
#max features= only keep 2000 most frequent words 
#ngrams of 1 to 2 consider 1 grams & 2 grams 

vectorizer = TfidfVectorizer(max_df=0.995, min_df=0.005, 
                             max_features=2000, ngram_range=[1,2])

dtm_train = vectorizer.fit_transform(df_train['Sentence_Clean'])
print(dtm_train.shape)

dtm_test=vectorizer.transform(df_test['Sentence_Clean'])
print(dtm_test.shape)

#Show list of words from the BOW using TF-IDF
vectorizer.get_feature_names()

#Append BOW to train set 
bow_df_train = pd.DataFrame(dtm_train.toarray(),columns=vectorizer.get_feature_names(),index=df_train.index)
df_train = pd.concat([df_train,bow_df_train],axis=1)
df_train.head()

#Append BOW to test set
bow_df_test = pd.DataFrame(dtm_test.toarray(),columns=vectorizer.get_feature_names(),index=df_test.index)
df_test = pd.concat([df_test,bow_df_test],axis=1)
df_test.head()

#Append predicted polarity to a new dataframe for the test data (this new dataframe will be used to answer Question 2 Part 3)
df_test2 = pd.read_csv("sentiment_test.csv")
df_test2['Sentence Clean'] = df_test['Sentence_Clean'].values
df_test2['Predicted Polarity from Textblob'] = df_test['sentiment'].values
df_test2['Predicted Polarity Score from Textblob'] = df_test['sentiment_score'].values

#Drop sentence, type and sentence clean columns as they are no longer needed since BOW, text preprocessing on sentiment was done in the pre processing steps above
df_train=df_train.drop(['Sentence','Type','Sentence_Clean'], axis=1)
df_test=df_test.drop(['Sentence','Type','Sentence_Clean'], axis=1)




### Model Development (Random Forest)
#Split data into train and test sets 
X = df_train.drop('Polarity', axis=1)
y = df_train['Polarity']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

type(X_train)
X_train.shape
X_train.head()

type(y_train)
y_train.shape
y_train.head()

#Grid Search with cross validation
#Score_func defines the performance measure (in this case auc) which the gridsearchCV should evaluate the models on 
score_func = make_scorer(roc_auc_score, greater_is_better=True)

# Create a parameter grid to test various hyper parameter values for the random forest model 
param_grid_rf = {
    'max_depth': [130, 150, 180],
    'max_features': [10, 15, 20],
    'min_samples_leaf': [5, 10, 15],
    'min_samples_split': [10, 20, 30],
    'n_estimators': [500, 1000, 1500]
}

# Defining the Random Forest Classifier model
classifier_RF = RandomForestClassifier(random_state=42)

# Initiate RF model using a variety of hyperparameters from the parameter grid above as well as a 5-fold cross validation
grid_search_rf = GridSearchCV(estimator = classifier_RF, param_grid = param_grid_rf, 
                          cv = 5, scoring = score_func, n_jobs=-1,return_train_score = True, verbose = 2)

#Apply model above on just the training data 
grid_search_RF = grid_search_rf.fit(X_train, y_train)

#Output the best hyper parameter values where auc for the model was the highest
print('\nBest Hyper-Parameter values Random Forest:'+str(grid_search_RF.best_params_))
grid_search_RF.best_params_

#Best Estimator for Random Forest Model
best_grid_rf = grid_search_RF.best_estimator_

#Score of the best model
best_result_rf = grid_search_RF.best_score_
print("\nBest Score Random Forest: " + str(best_result_rf))

#Using the above (best) model with the best hyper parameter values to predict on the testing data
class_threshold = 0.50
y_pred_prob_rf = grid_search_RF.predict_proba(X_val)[:,1]
y_pred_rf = np.where(y_pred_prob_rf > class_threshold, 1, 0) # classification

#Display confusion matrix, classification report and F1 score from Sklearn package 
print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred_rf))

print("\nF1 Score = {:.5f}".format(f1_score(y_val, y_pred_rf, average="micro")))

print("\nClassification Report:")
print(classification_report(y_val, y_pred_rf))

#Feature Importance - Using Random Forest Model
importances = grid_search_RF.best_estimator_.feature_importances_ 

#Plot the variables according to their importance
plt.figure(figsize=(15,5))
plt.title('Feature Importance Random Forest')
plt.xlabel('Decrease in Gini')
feature_importances = pd.Series(grid_search_RF.best_estimator_.feature_importances_ , index=X_train.columns)
feature_importances.nlargest(30).sort_values().plot(kind='barh', align='center')




### Make Prediction on Validation Set 
X = df_test.drop('Polarity', axis=1)
y = df_test['Polarity']

y_pred_prob_rf_val = grid_search_RF.predict_proba(X)[:,1]
y_pred_rf_val = np.where(y_pred_prob_rf_val > class_threshold, 1, 0) 

#Validate performance of random forest model on the provided test data (600 records in total)
print("Confusion matrix:")
print(confusion_matrix(y, y_pred_rf_val))

print("\nF1 Score = {:.5f}".format(f1_score(y, y_pred_rf_val, average="micro")))

print("\nClassification Report:")
print(classification_report(y, y_pred_rf_val))

#Append predicted polarity to original test data
df_test2['Predicted Polarity from RF Model'] = pd.Series(y_pred_rf_val, index=df_test.index)

conditions = [
    (df_test2['Polarity'] == 1) & (df_test2['Predicted Polarity from RF Model'] == 1),
    (df_test2['Polarity'] == 1) & (df_test2['Predicted Polarity from RF Model'] == 0),
    (df_test2['Polarity'] == 0) & (df_test2['Predicted Polarity from RF Model'] == 0),
    (df_test2['Polarity'] == 0) & (df_test2['Predicted Polarity from RF Model'] == 1),
    ]

# create a list of the values we want to assign for each condition
values = ['TP', 'FN', 'TN', 'FP']

# create a new column and use np.select to assign values to it using our lists as arguments
df_test2['Condition'] = np.select(conditions, values)

df_FN=df_test2[(df_test2.Condition=='FN')]
df_FP=df_test2[(df_test2.Condition=='FP')]

#Preview the two dataframes. One for FP's and one for FN's. 
df_FP.head(20)
df_FN.head(20)

#Export files for Question 2 Part 3
df_FP.to_csv ('FP.csv', index = False, header=True)
df_FN.to_csv ('FN.csv', index = False, header=True)

