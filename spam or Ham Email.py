# Project Name ==> Email Fraud Detection
# Created By   ==> Bhushan Shinde
# Date         ==> 23-02-2021
# Do not Copy Code Copy My Idea



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Dataset = pd.read_csv("spam_ham_dataset.csv")

# x = Dataset.iloc[:, 2].values
# print(x)
# print(Dataset.head())
# #
# # LIST_2 = ['label_num']
# # Dataset.drop(LIST_2, axis=1, inplace=True)
#

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
negative_words = ['not' , 'no']
for i in range (0,5171):
    review = re.sub('[^a-zA-Z]' , ' ' , Dataset.iloc[: , 2][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_words = stopwords.words('english')
    for j in negative_words:
        all_words.remove(j)
    review = [ps.stem(word)for word in review if not word in set(all_words)]
    review = ' '.join(review)
    corpus.append(review)
# print(corpus)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3700)
x  = cv.fit_transform(corpus).toarray()
y  = Dataset.iloc[:, -1].values
print(len(x[0]))

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.25 , random_state=0)

from sklearn.naive_bayes import GaussianNB
Classifier = GaussianNB()
Classifier.fit(x_train , y_train)
y_pred = Classifier.predict(x_test)

# print(np.concatenate((y_test.reshape(len(y_test) , 1) , y_pred.reshape(len(y_pred) , 1)) , 1))

from sklearn.metrics import confusion_matrix , accuracy_score

# print(confusion_matrix(y_test , y_pred))
# print(accuracy_score(y_test , y_pred))

# single Prediction.............

new_review = input("Enter Your Text Here==>  ")

new_review = re.sub('[^a-zA-Z]' , ' ' , new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
new_all_words = stopwords.words('english')
for j in negative_words:
    new_all_words.remove(j)
new_review = [ps.stem(word) for word in new_review if not word in set(new_all_words)]
new_review = ' '.join(new_review)
corpus_2 = [new_review]

print(corpus_2)
new_x_train  = cv.transform(corpus_2).toarray()
y_new_pred =   Classifier.predict(new_x_train)

print('\n')
print('---------------------------------------------------------------------')
print('\n')
if y_new_pred==0:
    print('\n')
    print('[ HAM ]')
else:
    print('\n')
    print('[ SPAM ]')

a = input()
