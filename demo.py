#
# Lesson #1 challenge from Sirajology - Introduction - Learn Python for Data Science #1
# https://www.youtube.com/watch?v=T5pRlIbr6gg
# Added a few classifiers

# (C) 2016 - Shazz 
# Under MIT license

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

classifier_names = ["Decision Tree",
         "K-Nearest Neighbors", 
         "Linear Support Vector Machine", 
         "Radial Basic Function Support Vector Machine", 
         "Random Forest", 
         "Neural Network", 
         "AdaBoost",
         "Gaussian Naive Bayes"]

classifier_classes = [
    DecisionTreeClassifier(),
    KNeighborsClassifier(3),
    SVC(kernel="linear"),
    SVC(kernel="rbf"),
    RandomForestClassifier(),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB()]


#training data [height, weight, shoe_size]
X_train = [[181, 80, 44], [177, 70, 43], [160, 60, 38], 
     [154, 54, 37], [166, 65, 40], [190, 90, 47], 
     [175, 64, 39], [177, 70, 40], [159, 55, 37], 
     [171, 75, 42], [181, 85, 43]]

Y_train = ['male', 'male', 'female', 
     'female', 'male', 'male', 
     'female', 'female', 'female', 
     'male', 'male']

#testing data [height, weight, shoe_size]
X_test = [[183, 82, 45], [189, 92, 47], [175, 70, 41], [155, 47, 38],
          [144, 36, 36], [180, 60, 39], [195, 110, 45]]
Y_test = ['male', 'male', 'male', 'female',
          'female', 'female', 'male']

# train and check the model against the test data for each classifier
max_accuracy = 0
best_classifiers = []
for name, clf in zip(classifier_names, classifier_classes):     
    clf = clf.fit(X_train, Y_train)
    
    # predict on test data
    prediction = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    if accuracy >= max_accuracy:
        best_classifiers.append(name)
        max_accuracy = accuracy
    
    print("Using classifier", name, X_test, "are", prediction,"=> accuracy", accuracy)
   
print("The most accurate classifiers are", best_classifiers,"with",max_accuracy*100,"% accuracy")

