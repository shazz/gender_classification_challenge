#
# Lesson #1 challenge from Sirajology - Introduction - Learn Python for Data Science #1
# https://www.youtube.com/watch?v=T5pRlIbr6gg
# Added a few classifiers
# Using CDC BRFSS Annual Survey Data 2015 
# https://www.cdc.gov/brfss/annual_data/annual_2015.html

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

# Classifiers we'd like to compare
classifiers = {
        "Decision Tree": DecisionTreeClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(3),
        "Linear Support Vector Machine":  SVC(kernel="linear"),
        "Radial Basic Function Support Vector Machine": SVC(kernel="rbf"),
        "Random Forest": RandomForestClassifier(),
        "Neural Network": MLPClassifier(), 
        "AdaBoost": AdaBoostClassifier(),
        "Gaussian Naive Bayes": GaussianNB()
    }
    
# load data and convert Y into 1d vector    
X, Y = data_importer.load_data(2000)    
print("X", X.shape, "Y", Y.shape)
    
# train and check the model against the test data for each classifier
iterations = 100
results = {}
for itr in range(iterations):    
    # Resuffle training/testing datasets by sampling randomly 80/20% of the input data 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)    
    
    for classifier in classifiers:
        clf = classifiers[classifier]
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

clf = classifiers[halloffame[0][1]]
clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
accuracy = accuracy_score(Y_test, prediction)

print(clf, "accuracy", accuracy)
print("a person of 100 kg and 185 cm is probably a", "male" if clf.predict([[100, 185]]) == 0 else "female")
print("a person of 55 kg and 158 cm is probably a", "male" if clf.predict([[55, 158]]) == 0 else "female")
print("a person of 40 kg and 148 cm is probably a", "male" if clf.predict([[40, 148]]) == 0 else "female")
