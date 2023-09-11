#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on a windy day

@author: Augusto Kuusberg Elias
@id: R00181405
@Cohort: SOFTWARE DEVELOPMENT EVENING
THAT ASSIGNMENT WAS MADE USING IDE SPYDER
AT END OF EACH TASK WE WILL RUN THAT SPECIFIC TASK BEFORE START NEXT TASK
AT END OF EACH TASK THERE ARE COMMENTS TO ANSWER THE QUESTIONS
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rnd

from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
    
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer

df = pd.read_csv("weatherAUS.csv", encoding = 'utf8')

def Task1():
    df = pd.read_csv("weatherAUS.csv", encoding = 'utf8')
    # select only the necessary attributes
    flt = df [['MinTemp', 'WindGustSpeed', 'Rainfall', 'RainTomorrow']].copy()
    # data cleansing
    flt = flt.dropna()
    
    # create four datasets as specified in the question 
    a = (flt[['MinTemp', 'WindGustSpeed', 'Rainfall']])
    b = (flt[['MinTemp', 'WindGustSpeed']])
    c = (flt[['MinTemp','Rainfall']])
    d= (flt[['WindGustSpeed','Rainfall']])
    
    # consider RainTomorrow as class attribute
    y = flt[['RainTomorrow']]
    
    # create a list of subsets to easily access each dataset
    subsets = [a, b, c , d]

    # create a for loop to run each dataset
    for option in subsets:
        X_train, X_test, y_train, y_test = train_test_split(option, y, test_size=0.33, random_state=42)
        
        c= 1
        # create 2 lists to store data from training and test
        mylistTr =[]
        mylistTs =[]
        index = range(1, 36)
       # create a nested for loop to run different max_depth for each dataset
        for n in range(35):
            clf =  tree.DecisionTreeClassifier(max_depth=c, random_state=42)
            c = c + 1
            clf.fit(X_train, y_train)
            mylistTr.append(clf.score(X_train, y_train))
            mylistTs.append(clf.score(X_test, y_test))         
            
        # create a vizualization plot for each dataset
        plt.plot(index, mylistTr)
        plt.plot(index, mylistTs)
        attributes = option.columns.tolist()
        plt.ylabel('accuracy')
        plt.xlabel('number of max_depth')
        plt.title(f'That is the attributes used for this graph {attributes}')
        plt.legend(['Train', 'Test'])
        plt.show()

# Question 1-A
# The dataset with a better accuracy is the option d with WindGustSpeed and Rainfall as attribute
# because the test accuracy is higher than the other ones (mostly of the time ~ >= 0.79)
# And the smallest gap (smallest overfitting) between Training and Testting accuracy (~ 0.03) 

# Question 1-B
# The attribute that has a more important role in predicting the RainTomorrow is Rainfall
# if you run attribute by attribute, with RainTomorrow as class attribute, you can see Rainfall attribute with 
# the best test accuracy (0.790)
# you can also vizualise it based on graphs, because the graphs where the Rainfall is present
# there is better accuracy for test

# Question 1-C
# The most appropriate value for max_depth should be around 3
# that because meets the highest point of the testing and its the smallest gap with training (overfitting)
# It is also important to mention in all the scenarios, when the max_depth increase, we can also see a increase of gap between training and testing

Task1()
    

def Task2():
    df = pd.read_csv("weatherAUS.csv", encoding = 'utf8')
    # select only the necessary attributes
    flt = df [['Pressure9am', 'Pressure3pm','Humidity9am', 'Humidity3pm', 'RainToday']].copy()
   
    # In the attribute RainToday, check if there is no or yes in the cells and change it to numeric value
    flt.loc[flt['RainToday'].str.lower().str.contains('no', na=False), 'RainToday'] = 0
    flt.loc[flt['RainToday'].str.lower().str.contains('yes', na=False), 'RainToday'] = 1

    # select the cells where is not numeric value and change it to empty
    flt['RainToday'] = flt['RainToday'].apply(pd.to_numeric, errors='coerce')
  
    # data cleansing
    flt = flt.dropna()
    # Create Pressure and Humidity attribute, based on average of specific attributes
    flt['Pressure'] = flt[['Pressure9am', 'Pressure3pm']].mean(axis=1)
    flt['Humidity'] = flt[['Humidity9am', 'Humidity3pm']].mean(axis=1)
    # from my previous created dataset, drop unnecessary attributes
    flt = flt.drop(['Pressure9am', 'Pressure3pm', 'Humidity9am', 'Humidity3pm'], axis=1)


    # create datasets as specified in the question
    X = (flt[['Humidity', 'Pressure']])
    # consider RainToday as class attribute
    y = flt[['RainToday']]
    
    # create a list to store different names and objects
    models = []

    models.append(('KNN', KNeighborsClassifier()))
    models.append(('DTC', tree.DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    #models.append(('SVM', SVC(kernel='linear')))
    models.append(('RFS',RandomForestClassifier()))

    # create a dictionary to store the results for each technique applied
    results = {}

    # create a loop to run each technique from the list models
    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=3, scoring='accuracy', return_train_score=True)
        # store in the results dictionary the name of model as a key and the result as a value
        results[name] = cv_results

    trainlist = []
    testlist= []
    modellist = []
    # create a loop for each technique stored at results dictionary
    # to print the averaged train and test accuracy
    for models in results:
        test = results[models]['test_score'].mean()
        train = results[models]['train_score'].mean()
        print(models)
        print(f'Training: {train}')
        print(f'Test: {test}')
        trainlist.append(train)
        testlist.append(test)
        modellist.append(models)
        
    # create a vizualization 
    w=0.4
    bar1 = np.arange(len(modellist))
    bar2 = [i+w for i in bar1]
    plt.bar(bar1, trainlist, w, label ="train")
    plt.bar(bar2, testlist, w, label ="test")
    plt.xticks(bar1, modellist)
    plt.ylabel('accuracy')
    plt.title('Accuracy train and test for different models')
    plt.legend()
    plt.show()



# Question 2
# Important: for some reason the SVM technique tooks too long on my device, where i decided to give up to it
# so i cant take it in consideration to check the best technique i set it just as a comment 
# For that example the best technique using cross-validation is the Gaussian naïve bayes
# Because we have the higher accuracy for the testing (0.8047)
# And the smallest gap (smallest overfitting) between Training and Testting accuracy (0.003) 

Task2()
    

def Task3():
    df = pd.read_csv("weatherAUS.csv", encoding = 'utf8')
    # select only the necessary attributes
    flt = df [['WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'MinTemp']].copy()
    # data cleansing
    flt = flt.dropna()
    
    # create datasets as specified in the question
    X = (flt[['WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm']])
    # consider MinTemp as class attribute
    y = flt[['MinTemp']]
    
    # we need to change float numbers to integers for discretization process
    y = y.values.reshape(-1,1)



    # As i couldnt set a proper loop, i did hard code one by one changing the n_bins
    enc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform') 
    enc.fit(y)
    y = enc.transform(y)
    # for that model we are going to use the train (67%) test (33%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # we set the number of neighbors to 5
    clf =  KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    train = clf.score(X_train, y_train)
    test = clf.score(X_test, y_test)  
    print(f'Training accuracy = {train} \nTest accuracy = {test}') 
    # Results
    #n_bins = 2 Train:0.7329982925807803 Test:0.6026337017899579
    #n_bins = 3 Train:0.7505449781471817 Test:0.6657945799812501
    #n_bins = 4 Train:0.6603524370993202 Test:0.5111082041554931
    #n_bins = 5 Train:0.6058331454098344 Test:0.4296117034033183
    #n_bins = 6 Train:0.5633409576581511 Test:0.3706368413020254
    
    # As I mentioned before, because i could not set a loop to show all results at once,
    # I decided to create a plot vizualization for that 
    binList = [2,3,4,5,6]
    trainingList =[0.7329982925807803, 0.7505449781471817, 0.6603524370993202, 0.6058331454098344, 0.5633409576581511] 
    testList = [0.6026337017899579, 0.6657945799812501, 0.5111082041554931, 0.4296117034033183, 0.3706368413020254]
        # create a vizualization 
    w=0.4
    bar1 = np.arange(len(binList))
    bar2 = [i+w for i in bar1]
    plt.bar(bar1, trainingList, w, label ="train")
    plt.bar(bar2, testList, w, label ="test")
    plt.xticks(bar1, binList)
    plt.ylabel('accuracy')
    plt.xlabel('number of bins')
    plt.title('Accuracy train and test for different  number of bins')
    plt.legend()
    plt.show()


# Question 3
# The best number of bins for discretization is 3 because has the best test accuracy = 0.6657
# and there is no overfitting, the gap is approximately 9% from train to test
Task3()
    
    
def Task4():
    df = pd.read_csv("weatherAUS.csv", encoding = 'utf8')
    # select only the necessary attributes
    flt = df [['Temp9am', 'Temp3pm', 'Humidity9am', 'Humidity3pm']].copy()
    # data cleansing
    flt = flt.dropna()
    # normalization
    # that step is necessary because otherwise one attribute can dominate another one
    # so when we normalizate all attributes contributes as same
    scalingObj = preprocessing.MinMaxScaler()
    newFLT = scalingObj.fit_transform(flt)

    # create a loop to run different number of bins
    for x in range (2,4):
        # all attributes descritized into two or three bins.
        enc = KBinsDiscretizer(n_bins=x, encode='ordinal', strategy='uniform')
        enc.fit(newFLT)
        newFLT_binned = enc.transform(newFLT)
        
        costs = []
        # create a loop to run different number of clusters 
        for i in range(2,9):
            kmeans = KMeans(n_clusters=i).fit(newFLT_binned)
            costs.append(kmeans.inertia_)
            
            print(f'With number of bins = {x}') # this line returns number of bins
            print(f'With number of cluster = {i} ') # this line number of cluster
            print(f'distortion cost: {kmeans.inertia_}') # this line returns cost
            print()
            
        # vizualization    
        if x ==2:
            indexs = np.arange(2, 9)
            plt.plot(indexs, costs,label='number of bins = 2')
            plt.ylabel('distortion cost')
            plt.xlabel('number of cluster')
            plt.title('Distortion cost for different number of clusters')
            plt.legend()
        if x ==3:
            indexs = np.arange(2, 9)
            plt.plot(indexs, costs,label='number of bins = 3')
            plt.ylabel('distortion cost')
            plt.xlabel('number of cluster')
            plt.title('Distortion cost for different number of clusters')
            plt.legend()

# Question 4
# When the number of bins are 2, looks like we have 3 elbows (when the number of cluster is around 3, 5 and 7)
# the last elbow is n_cluster around 7, after that there is relatively straight line with no major drop down

# When the number of bins are 3, looks like we have 2 elbows (when the number of cluster is around 4 and 6)
# the last elbow is n_cluster around 6, after that there is relatively straight line with no major drop down

# The distortion cost is lower when the number of bins is 2, compared with number of bin equals 3

Task4()
    


