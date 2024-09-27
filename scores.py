import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf


def scores_cm(pred, y_test, le):
    ttlpred = []
    for i in pred:
        maxidx = i.argmax()
        ttlpred.append(maxidx)

    print('Accuracy: ', accuracy_score(y_test, ttlpred)) 
    print()
    print('Precision (macro): ', precision_score(y_test, ttlpred, average='macro'))
    print('Precision (micro): ', precision_score(y_test, ttlpred, average='micro'))
    print('Precision (weighted): ', precision_score(y_test, ttlpred, average='weighted'))
    print()
    print('Recall (macro): ', recall_score(y_test, ttlpred, average='macro'))
    print('Recall (micro): ', recall_score(y_test, ttlpred, average='micro'))
    print('Recall (weighted): ', recall_score(y_test, ttlpred, average='weighted'))
    print()
    print('F1 Score (macro): ', f1_score(y_test, ttlpred, average='macro'))
    print('F1 Score (micro): ', f1_score(y_test, ttlpred, average='micro'))
    print('F1 Score (weighted): ', f1_score(y_test, ttlpred, average='weighted'))

    cm = confusion_matrix(y_test, ttlpred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot()
    plt.xticks(rotation=45, ha='right')
    plt.show()