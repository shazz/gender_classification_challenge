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

# Classifiers we'd like to compare
classifiers = {
        "DecisionTree": DecisionTreeClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(6),
        "Linear Support Vector Machine":  SVC(kernel="linear"),
        "Radial Basic Function Support Vector Machine": SVC(kernel="rbf"),
        "Random Forest": RandomForestClassifier(n_estimators=50),
        "Neural Network": MLPClassifier(alpha=1), 
        "AdaBoost": AdaBoostClassifier(n_estimators=100),
        "Gaussian Naive Bayes": GaussianNB()
    }
    
# input data [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], 
     [154, 54, 37], [166, 65, 40], [190, 90, 47], 
     [175, 64, 39], [177, 70, 40], [159, 55, 37], 
     [171, 75, 42], [181, 85, 43]]
# labels
Y = ['male', 'male', 'female', 
     'female', 'male', 'male', 
     'female', 'female', 'female', 
     'male', 'male']

# Create training/testing datasets by sampling 'randomly' 80/20% of the input data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# train and check the model against the test data for each classifier
max_accuracy = 0
best_classifiers = []
for classifier in classifiers:
    clf = classifiers[classifier]
    clf.fit(X_train, Y_train)
    
    # predict on test data
    prediction = clf.predict(X_test)
    # compute accuracy
    accuracy = accuracy_score(Y_test, prediction)
    
    # check if this is the best
    if accuracy > max_accuracy:
        best_classifiers = [classifier]
        max_accuracy = accuracy
    elif accuracy == max_accuracy:
        best_classifiers.append(classifier)
    
    print("Using classifier", classifier, X_test, "are", prediction,"=> accuracy", accuracy)
   
print("The most accurate classifiers are", best_classifiers,"with",max_accuracy*100,"% accuracy")

