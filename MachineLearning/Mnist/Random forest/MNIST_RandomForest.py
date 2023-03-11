import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, \
    roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from mllib import *


def show_digit(data, n):
    digit = data.iloc[n].values.reshape(28, 28)
    plt.imshow(digit, cmap=mpl.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()


def plot_digits(array, a, b):
    array = array.values.reshape(5, 5, 784)
    plt.suptitle(f"{a} classified as {b}")
    for i in range(5):
        for j in range(5):
            plt.subplot(5, 5, (i * 5 + j + 1))
            digit = array[i][j].reshape(28, 28)
            plt.imshow(digit, cmap=mpl.cm.binary, interpolation='nearest')
            plt.axis('off')
    plt.show()


if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', version=1, data_home='datasets/mnist')
    X, y = mnist.data, mnist.target
    # show_digit(X, 0)
    y = y.astype(np.uint8)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    forest_clf = RandomForestClassifier(random_state=0)
    forest_clf.fit(X_train, y_train)

    print(f"cross validation score: "
          f"{cross_val_score(forest_clf, X_train, y_train, cv=3, scoring='accuracy')}")  # [0.96465 0.96225 0.9659 ]

    # # Confusion Matrix
    y_train_pred = cross_val_predict(forest_clf, X_train, y_train, cv=3)
    conf_mtrx = confusion_matrix(y_true=y_train, y_pred=y_train_pred)
    print(conf_mtrx)
    print(f"precision score: "
          f"{precision_score(y_true=y_train, y_pred=y_train_pred, average='macro')}")  # 0.9639309479436831
    print(f"recall score: "
          f"{recall_score(y_true=y_train, y_pred=y_train_pred, average='macro')}")  # 0.9639676757525134
    print(f"f1 score for: "
          f"{f1_score(y_true=y_train, y_pred=y_train_pred, average='macro')}")  # 0.9639345013736424

    y_train_pred_prob = cross_val_predict(forest_clf, X_train, y_train, cv=3, method='predict_proba')
    for i in range(len(set(y_train))):
        plot_multiclass_precision_recall_vs_threshold(y_train, y_train_pred_prob, class_no=i)
    plot_multiclass_precision_vs_recall(y_train, y_train_pred_prob)
    plot_multiclass_roc_curve(y_train, y_train_pred_prob)
    for i in range(len(set(y_train))):
        print(f"AUC class {i}: {roc_auc_score(y_true=(y_train==i), y_score=y_train_pred_prob[:, i])}")
    # AUC class 0: 0.9997312753094914
    # AUC class 1: 0.9997412365806215
    # AUC class 2: 0.9984773026422488
    # AUC class 3: 0.9974793505762705
    # AUC class 4: 0.9987551907127988
    # AUC class 5: 0.9985731264972588
    # AUC class 6: 0.9996136004106966
    # AUC class 7: 0.999184912569747
    # AUC class 8: 0.9977407616353869
    # AUC class 9: 0.9972536257173972

    print(f"cross validation score for Random forest classifier: "
          f"{cross_val_score(forest_clf, X_train, y_train, cv=3, scoring='accuracy')}")  # [0.96465 0.96225 0.9659 ]
    X_train_scaled = StandardScaler().fit_transform(X_train)
    print(f"cross validation score for Random forest classifier (Scales input): "
          f"{cross_val_score(forest_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')}")  # [0.9647 0.9621 0.9659]

    # # Error Analysis
    y_train_pred_forest = cross_val_predict(forest_clf, X_train_scaled, y_train, cv=3)
    conf_mtrx_forest = confusion_matrix(y_true=y_train, y_pred=y_train_pred_forest)
    print(conf_mtrx_forest)
    plt.matshow(conf_mtrx_forest, cmap=plt.cm.gray)
    plt.title('Random Forest')
    plt.show()
    row_sums = conf_mtrx_forest.sum(axis=1, keepdims=True)
    norm_conf_mtrx_forest = conf_mtrx_forest / row_sums
    np.fill_diagonal(norm_conf_mtrx_forest, 0)
    plt.matshow(norm_conf_mtrx_forest, cmap=plt.cm.gray)
    plt.title('Random Forest Normalized')
    plt.show()
    # Check some of errors
    cl_a, cl_b = 4, 9
    X_aa = X_train[(y_train == cl_a) & (y_train_pred_forest == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_train_pred_forest == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred_forest == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_train_pred_forest == cl_b)]
    plot_digits(X_aa[:25], cl_a, cl_a)
    plot_digits(X_ab[:25], cl_a, cl_b)
    plot_digits(X_ba[:25], cl_b, cl_a)
    plot_digits(X_bb[:25], cl_b, cl_b)

    # # Predict Test Set
    (forest_clf.predict(X_test) == y_test).sum() / 10000  # 0.9705
    test_cvs = cross_val_score(forest_clf, X_test, y_test, cv=3, scoring='accuracy')
    print(f'cross_val_score: {test_cvs}')  # [0.92261548 0.93639364 0.95619562]
    X_test_norm = StandardScaler().fit_transform(X_test)
    test_cvs_norm = cross_val_score(forest_clf, X_test_norm, y_test, cv=3, scoring='accuracy')
    print(f'cross_val_score (Scaled input): {test_cvs_norm}')  # [0.9220156  0.93639364 0.95649565]

    y_test_pred = cross_val_predict(forest_clf, X_test_norm, y_test, cv=3)
    precision_score(y_test, y_test_pred, average='macro')  # 0.937782784979021
    recall_score(y_test, y_test_pred, average='macro')  # 0.9375927290258886
    f1_score(y_test, y_test_pred, average='macro')  # 0.9376254310881338

    test_conf_mtrx = confusion_matrix(y_test, y_test_pred)
    print(test_conf_mtrx)
    plt.matshow(test_conf_mtrx, cmap=plt.cm.gray)
    plt.title('Random Forest')
    plt.show()
    row_sum = test_conf_mtrx.sum(axis=1, keepdims=True)
    norm_test_conf_mtrx = test_conf_mtrx / row_sum
    np.fill_diagonal(norm_test_conf_mtrx, 0)
    plt.matshow(norm_test_conf_mtrx, cmap=plt.cm.gray)
    plt.title('Random Forest Scaled')
    plt.show()

    y_test_score = cross_val_predict(forest_clf, X_test_norm, y_test, cv=3, method='predict_proba')
    for i in range(10):
        plot_multiclass_precision_recall_vs_threshold(y_test, y_test_score, class_no=i)
    plot_multiclass_precision_vs_recall(y_test, y_test_score)
    plot_multiclass_roc_curve(y_test, y_test_score)
    for i in range(10):
        print(f"AUC class {i}: {roc_auc_score(y_true=(y_test==i), y_score=y_test_score[:, i])}")
    # AUC class 0: 0.9991357640617222
    # AUC class 1: 0.9990999599971178
    # AUC class 2: 0.9967744378807681
    # AUC class 3: 0.9943236709655392
    # AUC class 4: 0.9974250412955488
    # AUC class 5: 0.9961506392645504
    # AUC class 6: 0.9980159857108488
    # AUC class 7: 0.9972919424200843
    # AUC class 8: 0.9948672691394379
    # AUC class 9: 0.9908299996946622
