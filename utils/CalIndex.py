from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score, f1_score, roc_curve, roc_auc_score, auc
import numpy as np
def cal_index(y_true, y_pred):
    '''
    Calculate Accuracy, Recall, Precision, F1-Score
    https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
    '''
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro',labels=np.unique(y_pred))
    recall = recall_score(y_true, y_pred, average='macro',labels=np.unique(y_pred))
    F1_score = f1_score(y_true, y_pred, average='macro',labels=np.unique(y_pred))

    return acc, prec, recall, F1_score