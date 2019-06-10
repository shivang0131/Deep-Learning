# part 1: Data Preprocessing

 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  #upper bound excluded
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]      #to avoid dummy variable trap


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2: Making an ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

# initializing the ANN
classifier = Sequential()




# Evaluating, Improving and tuning an ANN

# Evaluating an ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():

    classifier = Sequential()
    classifier.add(Dense(activation='relu', input_dim = 11, units=6, kernel_initializer='uniform'))
    classifier.add(Dense(activation='relu', units=6, kernel_initializer='uniform'))
    classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))
    classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    return classifier

if __name__=='__main__':
    classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)


mean = accuracies.mean()
variance = accuracies.std()













