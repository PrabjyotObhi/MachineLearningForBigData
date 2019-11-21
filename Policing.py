#import Everything 
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import scipy as sp
from sklearn.tree import DecisionTreeClassifier #This is the library that implements the Decision Tree classifier 
from sklearn.model_selection import train_test_split #used to split the dataset into training and test data
from sklearn.tree import export_graphviz
from sklearn import tree
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus
import graphviz
from matplotlib import pyplot
from preamble import *
import matplotlib.pyplot as plt


# Step 1 Preprocessing the Data

#Referenced the Kaggle Assignment in order to start off this portion of preprocessing
policedata = pd.read_csv('police_project.csv')
policedata= policedata.drop(columns=["county_name"])
policedata= policedata.dropna()
policedata["is_arrested"]=policedata["is_arrested"].astype("category").cat.codes
# create y value from specific target
y=policedata["is_arrested"]
policedata = policedata.drop(columns=["is_arrested"])
# policedata = policedata.drop(columns=["stop_outcome_Arrest_Driver"])
# policedata = policedata.drop(columnes=["stop_outcome_Arrest_Passenger"])
policedata=pd.get_dummies(policedata)
policedata.head()
# print('Target names:',policedata.columns)


# split the data and the lables for  a training and testing set (Using policedata data set into the dataframe)
X_train, X_test, y_train, y_test = train_test_split(policedata, y, test_size=0.20, random_state=0)
y_test
# Dimensions of each
print('\n x_train',X_train.shape, '\n y_train', y_train.shape, '\n X_test',X_test.shape, '\n y_test', y_test.shape)


#Decision Tree
dct = DecisionTreeClassifier(max_depth=3, random_state = 1, criterion = 'entropy', min_impurity_decrease=0.1)
dct.fit(X_train, y_train)
print('Model trained')

# Predict for the test set
y_predictions = dct.predict(X_test)
# y_predictions

# Accuracy is calculated by comparing the true results with the predicted results.
# Accuracy = True prediction in test / Total no of observations in test
print('Accuracy for given train:test split = ',accuracy_score(y_predictions, y_test))
y_predictions = dct.predict(X_train)
print('Accuracy for given train:test split = ',accuracy_score(y_predictions, y_train))


#Modified from Decision Tree Iris Assignment
# print("Is_arreasted:\n{}".format(policedata.is_arrested))
from graphviz import Source
dot_data = tree.export_graphviz(dct, out_file='treepic.dot', feature_names=policedata.columns)
graph = graphviz.Source(dot_data)
# graph.render("treepic", view = True)
Source.from_file('treepic.dot')

# set_printoptions(threshold='nan')
print("Feature importances:\n{}".format(dct.feature_importances_))
print(len(dct.feature_importances_))


#RandomForest Classifier
rdf = RandomForestClassifier(n_estimators=100, max_depth=2)

print("Size of training set: {}   size of test set: {}".format(
      X_train.shape[0], X_test.shape[0]))

rdf.fit(X_train, y_train)

# evaluate the model on the test set using accuracy
print("Accuracy the training set: {:.3f}".format(rdf.score(X_train, y_train)))
print("Accuracy the testing set: {:.3f}".format(rdf.score(X_test, y_test)))

#Taken from the RandomForest Decision Tree in class activity
best_score = 0

#Train a Random Forest for each combination of parameters
for n_estimators in [1, 3, 5, 10]:
    for max_depth in [1, 2, 3]:
        rdf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = 349)
        #Perform cross-validation and compute mean cross-validation accuracy
        scores = cross_val_score(rdf, X_train, y_train, cv=5)
        score = np.mean(scores)
        #If we got a better score, store the score and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'n_estimators': n_estimators, 'max_depth': max_depth}
            
#Rebuild a model on the combined training and validation set
rdf = RandomForestClassifier(**best_parameters)
rdf.fit(X_train, y_train)

test_score = rdf.score(X_test, y_test)
print("Best score on training set: ", best_score)
print("Best parameters: ", best_parameters)
print("Test set score with best parameters: ", test_score)


# # "Naive Bayes" Algorithm

from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(policedata, y, test_size=0.4, random_state=1)
# training the model on the training set
gnb=GaussianNB(priors=None, var_smoothing=1e-09)
gnb.fit(X_train, y_train)
# make a prediction on the testing set
gnb_pred= gnb.predict(X_test)



from sklearn import metrics
print("GaussianNB model accuracy: ",metrics.accuracy_score(y_test,gnb_pred))
policedata.describe()



#Referenced https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
# logicstic
lr = LogisticRegression(penalty='l2', dual=True, max_iter=16) #dual is the primal formulation, the dual formulation only implemented for l2 penalty
lr.fit(X_train, y_train)


print("Logistic Regression accuracy score: ",lr.score(policedata,y))

# # Applied with Cross Validator

#Modified from the Cross Validation Slides
from sklearn.model_selection import KFold
k_fold = KFold(n_splits=2,random_state=3)
get_result = cross_val_score(lr,policedata,y,cv=k_fold,scoring ='accuracy')
print("KFold: ", get_result.mean())

