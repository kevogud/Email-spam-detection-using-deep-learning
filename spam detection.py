import numpy as np
import pandas as pd
import chardet
from gensim import parsing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score,  accuracy_score



with open('spam.csv', 'rb') as f:
      result= chardet.detect(f.read())

df = pd.read_csv('spam.csv', encoding = result['encoding'])  #read and preprocess the data
df = df.drop(["Unnamed: 2"], axis=1 )                          # for unwanted components we drop some value
df['v1'] = df.v1.map({'ham' : 0, 'spam' : 1})                  #v1  0&1  v2 emails
dataset = df.values
df.head()      

np.random.shuffle(dataset)
  #Splitting dataset into data and labels
X = dataset[:,1]
Y = dataset[:,0]
Y = Y.astype('int32')
    #parsing data by first converting into lowercase and then stemming it for effective vectorized
for i in range(X.shape[0]):
      X[i] = parsing.stem_text(X[i].lower())
  
#Data ino bag of wordformat
vectorizer = CountVectorizer()     
X_transformed = vectorizer.fit_transform(X)
#Split into test and training data (total data 5572 == 4000 for training 1572 for testing)
X_train = X_transformed[0:4000, :]  
Y_train = Y[0:4000]
X_test = X_transformed[4000:, :]
Y_test = Y[4000:]

X_train.shape
#print metrics == take into arguments like true labels and predicted variables and print differnet scores
def print_metrics(Y_true, Y_predicted):
    print("Accuracy score :" + str(accuracy_score(Y_true, Y_predicted)))
    print("Precision score :" + str(precision_score(Y_true, Y_predicted)))
    print("Recall score :" + str(recall_score(Y_true, Y_predicted)))
    print("ROC AUC score :" + str(roc_auc_score(Y_true, Y_predicted))) #receiving area characteristics area under the curve
    print("Confusion Matrix : \n")
    print(confusion_matrix(Y_true, Y_predicted))
#fitting multinomial naive bayes model
bayes_clf = MultinomialNB()
bayes_clf.fit(X_train, Y_train)

Y_predicted = bayes_clf.predict(X_test)
print_metrics(Y_test, Y_predicted)

svm_clf = SVC(C = 2000)
svm_clf.fit(X_train, Y_train)

Y_predicted_svm = svm_clf.predict(X_test)
print_metrics(Y_test, Y_predicted_svm)

