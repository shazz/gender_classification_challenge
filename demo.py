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
        "K-Nearest Neighbors": KNeighborsClassifier(3),
        "Linear Support Vector Machine":  SVC(kernel="linear"),
        "Radial Basic Function Support Vector Machine": SVC(kernel="rbf"),
        "Random Forest": RandomForestClassifier(),
        "Neural Network": MLPClassifier(alpha=1), 
        "AdaBoost": AdaBoostClassifier(),
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

# train and check the model against the test data for each classifier
iterations = 50
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
        
        print(itr, "- classifier", classifier, "says", X_test, "are", prediction,"=> accuracy", accuracy)
       
print("Classifiers average scores over", iterations, "iterations are:")
halloffame = [ (v,k) for k,v in results.items() ]
halloffame.sort(reverse=True)
for v,k in halloffame:
    print("\t-",k,"with",v/iterations*100,"% accuracy")     
