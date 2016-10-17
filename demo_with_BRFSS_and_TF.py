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


tf.logging.set_verbosity(tf.logging.ERROR)

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
    "Neural Network":                               ["sklearn.neural_network", "MLPClassifier", None], 
    "AdaBoost":                                     ["sklearn.ensemble", "AdaBoostClassifier", None],
    "Gaussian Naive Bayes":                         ["sklearn.naive_bayes", "GaussianNB", None],
    "TF Deep Neural Network":                       ["tensorflow.contrib.learn", "DNNClassifier", {"hidden_units": [5], "n_classes": 2, "enable_centered_bias": None}],
    "TF Linear Classifier":                         ["tensorflow.contrib.learn", "LinearClassifier", {"n_classes": 2, "enable_centered_bias": None}]
} 
    
# load data and convert Y into 1d vector    
X, Y = data_importer.load_data(500)    
print("X", X.shape, "Y", Y.shape)
    
# train and check the model against the test data for each classifier
iterations = 1
results = {}
for itr in range(iterations):   
    
    # Resuffle training/testing datasets by sampling randomly 80/20% of the input data 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)    
    
    for classifier in classifiers:
        if classifier == "TF Deep Neural Network":
            feature_columns = learn.infer_real_valued_columns_from_input(X_train)   
            clf = DNNClassifier(feature_columns=feature_columns, 
                               hidden_units=[10,10,10], 
                               n_classes=2, enable_centered_bias=None);
            clf.fit(X_train, Y_train, steps=200)
        elif classifier == "TF Linear Classifier":
            feature_columns = learn.infer_real_valued_columns_from_input(X_train)   
            clf = LinearClassifier(n_classes=2, feature_columns=feature_columns)     
            clf.fit(X_train, Y_train, steps=200)
        else:
            clf = getInstance(classifiers[classifier][0], classifiers[classifier][1], classifiers[classifier][2])
            clf.fit(X_train, Y_train)
        
        
        # predict on test data
        prediction = clf.predict(X_test)
        
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

#clf = classifiers[halloffame[0][1]]
clf = getInstance(classifiers[halloffame[0][1]][0], classifiers[halloffame[0][1]][1], classifiers[halloffame[0][1]][2])
clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
accuracy = accuracy_score(Y_test, prediction)

print(clf, "accuracy", accuracy)
print("a person of 100 kg and 185 cm is probably a", "male" if clf.predict([[100, 185]]) == 1 else "female")
print("a person of 55 kg and 158 cm is probably a", "male" if clf.predict([[55, 158]]) == 1 else "female")
print("a person of 40 kg and 148 cm is probably a", "male" if clf.predict([[40, 148]]) == 1 else "female")


    
