from pyod.models.knn import KNN   # kNN detector
from pyod.utils.data import generate_data
from pyod.utils.example import visualize
from pyod.models.cblof import CBLOF

# Generate sample data

contamination = 0.1  # percentage of outliers
n_train = 200  # number of training points
n_test = 100  # number of testing points
# X_train, y_train, X_test, y_test = generate_data(
#     n_train=n_train, n_test=n_test, contamination=contamination)
X_train, X_test, y_train, y_test = generate_data(n_train=n_train, n_test=n_test,
                                                 contamination=contamination,behaviour='new',
                                                 random_state=42)
# train kNN detector
# clf_name = 'KNN'
# clf = KNN()
# clf.fit(X_train)

clf_name = 'CBLOF'
clf = CBLOF()
clf.fit(X_train)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores

# it is possible to get the prediction confidence as well
# outlier labels (0 or 1) and confidence in the range of [0,1]
y_test_pred, y_test_pred_confidence = clf.predict(X_test, return_confidence=True)

# Evaluate the prediction using ROC and Precision
from pyod.utils.data import evaluate_print
# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)

# Generate the visualizations by visualize function included in all examples.
visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
          y_test_pred, show_figure=True, save_figure=False)