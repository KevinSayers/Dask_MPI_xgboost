from sklearn import datasets
import xgboost as xgb
import dask_xgboost as dxgb

from dask_mpi import initialize

from dask.distributed import Client, wait
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import dask
import dask.dataframe as dd
import dask.array as da
from dask_ml.model_selection import train_test_split
import time

def main():
    start = time.time()
    initialize(interface='ib0')
    client = Client()

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.05)
    D_test = xgb.DMatrix(X_test, label=Y_test)

    params = {'eta': 0.3, 
        'max_depth': 3,  
        'objective': 'multi:softprob',  
        'num_class': 3} 

    bst = dxgb.train(client, params, da.asarray(X_train), da.asarray(Y_train), num_boost_round=10)
    preds = bst.predict(D_test)
    best_preds = np.asarray([np.argmax(line) for line in preds])

    print("Precision = {}".format(precision_score(Y_test, best_preds, average='macro')))
    print("Recall = {}".format(recall_score(Y_test, best_preds, average='macro')))
    print("Accuracy = {}".format(accuracy_score(Y_test, best_preds)))
    elapsed = (time.time() - start)
    print (f"Elapsed time: {elapsed}")

if __name__ == '__main__':
    main()