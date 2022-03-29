import os
import scipy.io as sio
from pyod.models.lscp import LSCP
from pyod.models.lof import LOF
from sklearn.model_selection import train_test_split


def loaddata(filename):
    mat = sio.loadmat(os.path.join('datasets', filename + '.mat'))
    X_orig = mat['X']
    y_orig = mat['y'].ravel()

    return X_orig, y_orig


if __name__ == '__main__':
    data = 'cardio'
    X_orig, y_orig = loaddata(data)

    # X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig,
    #                                                     test_size=0.4,
    #                                                     random_state=42)
    detector_list = [LOF(), LOF()]
    clf = LSCP(detector_list)
    clf.fit(X_orig)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores
    print(y_train_pred)
    print(y_train_scores)

    # # get the prediction on the test data
    # y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    # y_test_scores = clf.decision_function(X_test)  # outlier scores
    #
    # # it is possible to get the prediction confidence as well
    # y_test_pred, y_test_pred_confidence = clf.predict(X_test, return_confidence=True)

