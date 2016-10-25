#
# Lesson #1 challenge from Sirajology - Introduction - Learn Python for Data Science #1
# https://www.youtube.com/watch?v=T5pRlIbr6gg
# Added a few classifiers

# (C) 2016 - Shazz 
# Under MIT license

# classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# accuracy scoring, and data partitioning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import data_importer
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn import DNNClassifier
from tensorflow.contrib.learn import LinearClassifier
import importlib
from monitor import ExampleMonitor

tf.logging.set_verbosity(tf.logging.ERROR)
example_monitor = ExampleMonitor()

def getInstance(module, classname, params):
    my_module = importlib.import_module(module)
    my_class = getattr(my_module, classname)
    if params != None:
        instance = my_class(**params)
    else:
        instance = my_class()
    #print("class instanced:", instance)    
    return instance   

# Classifiers we'd like to compare
classifiers = {
    "Decision Tree":                                ["sklearn.tree", "DecisionTreeClassifier", None],
    "K-Nearest Neighbors":                          ["sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors": 3}],
    "Linear Support Vector Machine":                ["sklearn.svm", "SVC", {"kernel" : "linear"}],
    "Radial Basic Function Support Vector Machine": ["sklearn.svm", "SVC", { "kernel": "rbf" }],
    "Random Forest":                                ["sklearn.ensemble", "RandomForestClassifier", None],
    "Neural Network":                               ["sklearn.neural_network", "MLPClassifier", {'activation': 'logistic', 'solver': 'adam', 'learning_rate': 'invscaling', 'momentum': .9,'nesterovs_momentum': True, 'learning_rate_init': 0.02}], 
    "AdaBoost":                                     ["sklearn.ensemble", "AdaBoostClassifier", None],
    "Gaussian Naive Bayes":                         ["sklearn.naive_bayes", "GaussianNB", None],
    "TF Deep Neural Network":                       ["tensorflow.contrib.learn", "DNNClassifier", {"hidden_units": [5], "n_classes": 2, "feature_columns": [tf.contrib.layers.real_valued_column("", dimension=2)], "enable_centered_bias": None}],
    "TF Linear Classifier":                         ["tensorflow.contrib.learn", "LinearClassifier", {"n_classes": 2, "feature_columns": [tf.contrib.layers.real_valued_column("", dimension=2)], "enable_centered_bias": None}]
} 
    
# load data and convert Y into 1d vector    
X, Y = data_importer.load_data(2000)    
print("X", X.shape, "Y", Y.shape)
print(Y)

# train and check the model against the test data for each classifier
iterations = 10
results = {}
for itr in range(iterations):   
    
    # Resuffle training/testing datasets by sampling randomly 80/20% of the input data 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)    
    
    for classifier in classifiers:
        clf = getInstance(classifiers[classifier][0], classifiers[classifier][1], classifiers[classifier][2])
        if classifier == "TF Deep Neural Network" or classifier == "TF Linear Classifier": 
            clf.fit(X_train, Y_train, max_steps=200)
        else:
            clf.fit(X_train, Y_train)
            
        # predict on test data
        prediction = clf.predict(X_test)
        
        # work with sckit learn but not with TF.learn
        #print("a person of 100 kg and 185 cm is probably a", "male" if clf.predict([[100.0, 185.0]]) == 1 else "female")
        #print("a person of 55 kg and 158 cm is probably a", "male" if clf.predict([[55, 158]]) == 1 else "female")
        #print("a person of 40 kg and 148 cm is probably a", "male" if clf.predict([[40, 148]]) == 1 else "female")
        
        # compute accuracy and sum it to the previous ones
        accuracy = accuracy_score(Y_test, prediction)
        if classifier in results:
            results[classifier] = results[classifier] + accuracy
        else:
            results[classifier] = accuracy
        
        #print(itr, "- classifier", classifier, "says", X_test, "are", prediction,"=> accuracy", accuracy)
        print(itr, "- classifier", classifier, "accuracy", accuracy)
       
print("Classifiers average scores over", iterations, "iterations are:")
halloffame = [ (v,k) for k,v in results.items() ]
halloffame.sort(reverse=True)
for v,k in halloffame:
    print("\t-",k,"with",v/iterations*100,"% accuracy")     
    
#using the best to train the full dataset
X, Y = data_importer.load_data(0)    
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)    

print("Retraining with best classifier", halloffame[0])
bestclf = getInstance(classifiers[halloffame[0][1]][0], classifiers[halloffame[0][1]][1], classifiers[halloffame[0][1]][2])
print("fit")
bestclf.fit(X_train, Y_train)
print("pred")
prediction = bestclf.predict(X_test)
print("accuracy")
accuracy = accuracy_score(Y_test, prediction)

print(bestclf, "accuracy", accuracy)
print("a person of 100 kg and 185 cm is probably a", "male" if bestclf.predict([[100, 185]]) == 0 else "female")
print("a person of 55 kg and 158 cm is probably a", "male" if bestclf.predict([[55, 158]]) == 0 else "female")
print("a person of 40 kg and 148 cm is probably a", "male" if bestclf.predict([[40, 148]]) == 0 else "female")

    
