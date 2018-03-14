
from sklearn import linear_model, datasets
from sklearn import cross_validation

import pickle
import gzip

def load_data():
    with gzip.open("/home/ren/Desktop/machine_learning/data/1book/data/MNIST/mnist.pkl.gz")as fp:
        training_data, valid_data, test_data = pickle.load(fp, encoding="iso-8859-1")
    return training_data, valid_data, test_data

if __name__ =="__main__":
    training_data, valid_data, test_data = load_data()
    print("jj")
    x1, y1 = training_data
    x2, y2 = test_data
    logreg = linear_model.LogisticRegression(C=1e5)
    print("jj")
    logreg.fit(x1, y1)
    print("jj")
    print(cross_validation.cross_val_score(logreg, x2, y2, scoring="accuracy"))

