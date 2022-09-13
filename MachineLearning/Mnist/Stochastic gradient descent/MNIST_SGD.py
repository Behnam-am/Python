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
    y = y.astype(np.uint8)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    # #################################################################
    # ######## Stochastic gradient descent OneVsRestClassifier ########
    # #################################################################
    sgd_clf = SGDClassifier(random_state=0)
    sgd_clf.fit(X_train, y_train)
    # (sgd_clf.predict(X_train) == y_train).sum() / 60000  # 0.8662333333333333
    cvs_sgd = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')
    print(f'cross validation score for sgd classifier: '
          f'{cvs_sgd}')
    # # prints: cross validation score for sgd classifier: [0.8485 0.8676 0.8621]
    X_train_scaled = StandardScaler().fit_transform(X_train)
    cvs_sgd_norm = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')
    print(f'cross validation score for sgd classifier (Scaled input): '
          f'{cvs_sgd_norm}')
    # prints: cross validation score for sgd classifier (Scaled input): [0.89935, 0.9016 , 0.9017 ]

    # Error Analysis
    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    conf_mtrx = confusion_matrix(y_true=y_train, y_pred=y_train_pred)
    print(conf_mtrx)
    plt.matshow(conf_mtrx, cmap=plt.cm.gray)
    plt.title('SGD OvR')
    plt.show()
    row_sums = conf_mtrx.sum(axis=1, keepdims=True)
    norm_conf_mtrx = conf_mtrx / row_sums
    np.fill_diagonal(norm_conf_mtrx, 0)
    plt.matshow(norm_conf_mtrx, cmap=plt.cm.gray)
    plt.title('SGD OvR Normalized')
    plt.show()
    # Check some of errors
    cl_a, cl_b = 5, 8
    X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
    plot_digits(X_aa[:25], cl_a, cl_a)
    plot_digits(X_ab[:25], cl_a, cl_b)
    plot_digits(X_ba[:25], cl_b, cl_a)
    plot_digits(X_bb[:25], cl_b, cl_b)

    # # Predict Test Set
    (sgd_clf.predict(X_test) == y_test).sum() / 10000  # 0.8592
    test_cvs_sgd = cross_val_score(sgd_clf, X_test, y_test, cv=3, scoring='accuracy')
    print(f'cross_val_score: {test_cvs_sgd}')  # [0.82633473 0.86258626 0.88178818]
    X_test_norm = StandardScaler().fit_transform(X_test)
    test_cvs_sgd_norm = cross_val_score(sgd_clf, X_test_norm, y_test, cv=3, scoring='accuracy')
    print(f'cross_val_score (Scaled input): {test_cvs_sgd_norm}')  # [0.86592681 0.89828983 0.8949895]

    y_test_pred = cross_val_predict(sgd_clf, X_test_norm, y_test, cv=3)
    precision_score(y_test, y_test_pred, average='macro')  # 0.8864533963702435
    recall_score(y_test, y_test_pred, average='macro')  # 0.885372274050377
    f1_score(y_test, y_test_pred, average='macro')  # 0.885372274050377

    test_conf_mtrx = confusion_matrix(y_test, y_test_pred)
    print(test_conf_mtrx)
    plt.matshow(test_conf_mtrx, cmap=plt.cm.gray)
    plt.title('SGD OvR')
    plt.show()
    row_sum = test_conf_mtrx.sum(axis=1, keepdims=True)
    norm_test_conf_mtrx = test_conf_mtrx / row_sum
    np.fill_diagonal(norm_test_conf_mtrx, 0)
    plt.matshow(norm_test_conf_mtrx, cmap=plt.cm.gray)
    plt.title('SGD OvR Scaled')
    plt.show()
    y_test_score = cross_val_predict(sgd_clf, X_test_norm, y_test, cv=3, method='decision_function')
    for i in range(10):
        plot_multiclass_precision_recall_vs_threshold(y_test, y_test_score, class_no=i)
        print(f"AUC class {i}: {roc_auc_score(y_true=(y_test==i), y_score=y_test_score[:, i])}")
    plot_multiclass_precision_vs_recall(y_test, y_test_score)
    plot_multiclass_roc_curve(y_test, y_test_score)

    # ################################################################
    # ######## Stochastic gradient descent OneVsOneClassifier ########
    # ################################################################
    ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=0))
    ovo_clf.fit(X_train, y_train)
    # (ovo_clf.predict(X_train) == y_train).sum() / 60000  # 0.9315666666666667
    cvs_ovo = cross_val_score(ovo_clf, X_train, y_train, cv=3, scoring='accuracy')
    print(f'cross validation score for ovo classifier: '
          f'{cvs_ovo}')
    # prints: cross validation score for ovo classifier: [0.9172 0.9125 0.9155]
    cvs_ovo_norm = cross_val_score(ovo_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')
    print(f'cross validation score for ovo classifier (Scales input): '
          f'{cvs_ovo_norm}')
    # prints: cross validation score for ovo classifier (Scales input): [0.9138 0.9125 0.9175]

    # Error Analysis
    try:
        y_train_pred_ovo = np.load('y_train_pred_ovo.npy')
    except FileNotFoundError:
        y_train_pred_ovo = cross_val_predict(ovo_clf, X_train_scaled, y_train, cv=3)
        np.save('y_train_pred_ovo.npy', y_train_pred_ovo)
    conf_mtrx_ovo = confusion_matrix(y_true=y_train, y_pred=y_train_pred_ovo)
    print(conf_mtrx_ovo)
    plt.matshow(conf_mtrx_ovo, cmap=plt.cm.gray)
    plt.title('SGD OvO')
    plt.show()
    row_sums = conf_mtrx_ovo.sum(axis=1, keepdims=True)
    norm_conf_mtrx_ovo = conf_mtrx_ovo / row_sums
    np.fill_diagonal(norm_conf_mtrx_ovo, 0)
    plt.matshow(norm_conf_mtrx_ovo, cmap=plt.cm.gray)
    plt.title('SGD OvO Normalized')
    plt.show()
    # Check some of errors
    cl_a, cl_b = 3, 5
    X_aa = X_train[(y_train == cl_a) & (y_train_pred_ovo == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_train_pred_ovo == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred_ovo == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_train_pred_ovo == cl_b)]
    plot_digits(X_aa[:25], cl_a, cl_a)
    plot_digits(X_ab[:25], cl_a, cl_b)
    plot_digits(X_ba[:25], cl_b, cl_a)
    plot_digits(X_bb[:25], cl_b, cl_b)

    # # Predict Test Set
    (ovo_clf.predict(X_test) == y_test).sum() / 10000  # 0.9103
    test_cvs_ovo = cross_val_score(ovo_clf, X_test, y_test, cv=3, scoring='accuracy')
    print(f'cross_val_score: {test_cvs_ovo}')  # [0.88662268 0.90939094 0.9219922 ]
    X_test_norm = StandardScaler().fit_transform(X_test)
    test_cvs_ovo_norm = cross_val_score(ovo_clf, X_test_norm, y_test, cv=3, scoring='accuracy')
    print(f'cross_val_score (Scaled input): {test_cvs_ovo_norm}')  # [0.88782244 0.90849085 0.91719172]

    y_test_pred = cross_val_predict(ovo_clf, X_test_norm, y_test, cv=3)
    precision_score(y_test, y_test_pred, average='macro')  # 0.9034242550928548
    recall_score(y_test, y_test_pred, average='macro')  # 0.9033898023503497

    test_conf_mtrx = confusion_matrix(y_test, y_test_pred)
    print(test_conf_mtrx)
    plt.matshow(test_conf_mtrx, cmap=plt.cm.gray)
    plt.title('SGD OvO')
    plt.show()
    row_sum = test_conf_mtrx.sum(axis=1, keepdims=True)
    norm_test_conf_mtrx = test_conf_mtrx / row_sum
    np.fill_diagonal(norm_test_conf_mtrx, 0)
    plt.matshow(norm_test_conf_mtrx, cmap=plt.cm.gray)
    plt.title('SGD OvO Scaled')
    plt.show()
    y_test_score = cross_val_predict(ovo_clf, X_test_norm, y_test, cv=3, method='decision_function')
    for i in range(len(set(y_test))):
        plot_multiclass_precision_recall_vs_threshold(y_test, y_test_score, class_no=i)
    plot_multiclass_precision_vs_recall(y_test, y_test_score)
    plot_multiclass_roc_curve(y_test, y_test_score)
    for i in range(len(set(y_test))):
        print(f"AUC class {i}: {roc_auc_score(y_true=(y_test==i), y_score=y_test_score[:, i])}")
