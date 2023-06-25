"""
A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier.
"""

import mnist_loader

from sklearn import svm
from sklearn.metrics import accuracy_score

def mnist_svm():
    training_data, validation_data, test_data = mnist_loader.load_data()
    
    clf = svm.SVC(kernel="rbf")
    clf.fit(training_data[0],training_data[1])
    
    prediction = clf.predict(test_data[0])
    accuracy = accuracy_score(test_data[1],prediction)
    print(accuracy)
    return accuracy
    
if __name__ == "__main__":
    mnist_svm()
    