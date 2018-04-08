
# 3.1. Cross-validation: evaluating estimator performance
# http://scikit-learn.org/stable/modules/cross_validation.html

# Iris flower data set
# https://en.wikipedia.org/wiki/Iris_flower_data_set

# import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

def main():
    iris = datasets.load_iris()    
    print("Data Shape:", iris.data.shape)
    
#     X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)    
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5, stratify=iris.target, random_state=0)
    
#     define the SVC model
    classifier_model = SVC(C=1, kernel="linear", random_state=0) 
       
#     fit the model
    classifier_model.fit(X_train, y_train)
    
#     get y predictions    
    y_predicted = classifier_model.predict(X_test) 
         
#     calculate accuracy score
    predicted_score = accuracy_score(y_test, y_predicted) * 100
    print("Predicted Accuracy: %0.3f" % (predicted_score))
        
#     CROSS VALIDATION CODE
    cv_scores = cross_val_score(classifier_model, iris.data, iris.target, cv=5)    
    print("Cross Validation Accuracy: %0.3f (+/- %0.3f)" % (cv_scores.mean(), cv_scores.std() * 2))
     
    cv_y_predicted = cross_val_predict(classifier_model, iris.data, iris.target, cv=5)
    cv_predicted_score = accuracy_score(iris.target, cv_y_predicted) * 100
    print("Cross Validation Predicted Accuracy: %0.3f" % (cv_predicted_score))
    
if __name__ == '__main__':
    main()