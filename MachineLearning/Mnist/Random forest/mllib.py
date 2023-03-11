import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, threshold_limit):
    point = np.where(thresholds <= threshold_limit)[0][-1]
    point_x = [thresholds[point], thresholds[point]]
    point_y = [recalls[point], precisions[point]]

    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-''', label='Recall')
    plt.legend()
    plt.grid(visible=True, which='both', axis='both')
    plt.xlabel('thresholds')
    plt.plot(point_x, point_y, 'ro')

    plt.vlines(point_x, 0, point_y, linestyles='dotted', colors='r')
    plt.hlines(point_y, thresholds[0], point_x, linestyles='dotted', colors='r')
    plt.annotate(f'({round(point_x[0], 2)}, {round(point_y[0], 2)})', (point_x[0], point_y[0]))
    plt.annotate(f'({round(point_x[1], 2)}, {round(point_y[1], 2)})', (point_x[1], point_y[1]))
    plt.show()


def plot_precision_vs_recall(precisions, recalls, recall_limit):
    point = np.where(recalls >= recall_limit)[0][-1]
    point_x = recalls[point]
    point_y = precisions[point]

    plt.plot(recalls[:-1], precisions[:-1], 'b-', label='Precision')
    plt.plot(point_x, point_y, 'ro')
    plt.grid(visible=True, which='both', axis='both')
    plt.xlabel('Recalls')
    plt.ylabel('Precisions')
    plt.vlines(point_x, 0, point_y, linestyles='dotted', colors='r')
    plt.hlines(point_y, 0, point_x, linestyles='dotted', colors='r')
    plt.annotate(f'({round(point_x, 2)}, {round(point_y, 2)})', (point_x, point_y))
    plt.show()


def plot_roc_curve(fpr, tpr, tpr_limit=0.0, label=None):
    if tpr_limit:
        point = np.argmax(tpr >= tpr_limit)
        point_x = fpr[point]
        point_y = tpr[point]
        plt.vlines(point_x, 0, point_y, linestyles='dotted', colors='r')
        plt.hlines(point_y, 0, point_x, linestyles='dotted', colors='r')
        plt.annotate(f'({round(point_x, 2)}, {round(point_y, 2)})', (point_x, point_y))

    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # dashed diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.grid(visible=True, which='both', axis='both')
    plt.legend()
    plt.show()


def plot_multiclass_precision_vs_recall(y_true, y_score):
    precisions = dict()
    recalls = dict()
    thresholds = dict()
    for i in range(len(set(y_true))):
        precisions[i], recalls[i], thresholds[i] = precision_recall_curve((y_true == i), y_score[:, i])
        plt.plot(recalls[i], precisions[i], label='class {}'.format(i))
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend()
    plt.title('precision vs. recall curve')
    plt.grid(visible=True, which='both', axis='both')
    plt.show()


def plot_multiclass_roc_curve(y_true, y_score):
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    for i in range(len(set(y_true))):
        fpr[i], tpr[i], thresholds[i] = roc_curve((y_true == i), y_score[:, i])
        plt.plot(fpr[i], tpr[i], label='class {}'.format(i))
    plt.xlabel('False Positive Range')
    plt.ylabel('True Positive Range (Recall)')
    plt.legend()
    plt.title('ROC curve')
    plt.grid(visible=True, which='both', axis='both')
    plt.show()


def plot_multiclass_precision_recall_vs_threshold(y_true, y_score, class_no=0):
    precisions = dict()
    recalls = dict()
    thresholds = dict()
    for i in range(len(set(y_true))):
        precisions[i], recalls[i], thresholds[i] = precision_recall_curve((y_true == i), y_score[:, i])

    plt.plot(thresholds[class_no], precisions[class_no][:-1], label=f'precision class {class_no}')
    plt.plot(thresholds[class_no], recalls[class_no][:-1], label=f'recall class {class_no}')
    plt.xlabel('threshold')
    plt.ylabel('precision')
    plt.legend()
    plt.title('precision-recall vs. threshold curve')
    plt.grid(visible=True, which='both', axis='both')
    plt.show()
