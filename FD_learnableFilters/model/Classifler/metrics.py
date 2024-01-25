# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/051_metrics.ipynb (unless otherwise specified).

__all__ = ['MatthewsCorrCoefBinary', 'get_task_metrics', 'accuracy_multi', 'metrics_multi_common', 'precision_multi',
           'recall_multi', 'specificity_multi', 'balanced_accuracy_multi', 'Fbeta_multi', 'F1_multi', 'mae', 'mape',
           'recall_at_specificity', 'mean_per_class_accuracy']

# Cell
import sklearn.metrics as skm
from fastai.metrics import *
from .imports import *

# Cell
mk_class('ActivationType', **{o:o.lower() for o in ['No', 'Sigmoid', 'Softmax', 'BinarySoftmax']},
         doc="All possible activation classes for `AccumMetric")

# Cell
def MatthewsCorrCoefBinary(sample_weight=None):
    "Matthews correlation coefficient for single-label classification problems"
    return AccumMetric(skm.matthews_corrcoef, dim_argmax=-1, activation=ActivationType.BinarySoftmax, thresh=.5, sample_weight=sample_weight)

# Cell
def get_task_metrics(dls, binary_metrics=None, multi_class_metrics=None, regression_metrics=None, verbose=True):
    if dls.c == 2:
        pv('binary-classification task', verbose)
        return binary_metrics
    elif dls.c > 2:
        pv('multi-class task', verbose)
        return multi_class_metrics
    else:
        pv('regression task', verbose)
        return regression_metrics

# Cell
def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True, by_sample=False):
    "Computes accuracy when `inp` and `targ` are the same size."
    if sigmoid: inp = inp.sigmoid()
    correct = (inp>thresh)==targ.bool()
    if by_sample:
        return (correct.float().mean(-1) == 1).float().mean()
    else:
        inp,targ = flatten_check(inp,targ)
        return correct.float().mean()

def metrics_multi_common(inp, targ, thresh=0.5, sigmoid=True, by_sample=False):
    "Computes TP, TN, FP, FN when `inp` and `targ` are the same size."
    if not by_sample: inp,targ = flatten_check(inp,targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp>thresh

    correct = pred==targ.bool()
    TP = torch.logical_and(correct, (targ==1).bool()).sum()
    TN = torch.logical_and(correct, (targ==0).bool()).sum()

    incorrect = pred!=targ.bool()
    FN = torch.logical_and(incorrect, (targ==1).bool()).sum()
    FP = torch.logical_and(incorrect, (targ==0).bool()).sum()

    N =  targ.size()[0]
    return N, TP, TN, FP, FN

def precision_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Computes precision when `inp` and `targ` are the same size."

    inp,targ = flatten_check(inp,targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp>thresh

    correct = pred==targ.bool()
    TP = torch.logical_and(correct,  (targ==1).bool()).sum()
    FP = torch.logical_and(~correct, (targ==0).bool()).sum()

    precision = TP/(TP+FP)
    return precision

def recall_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Computes recall when `inp` and `targ` are the same size."

    inp,targ = flatten_check(inp,targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp>thresh

    correct = pred==targ.bool()
    TP = torch.logical_and(correct,  (targ==1).bool()).sum()
    FN = torch.logical_and(~correct, (targ==1).bool()).sum()

    recall = TP/(TP+FN)
    return recall

def specificity_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Computes specificity (true negative rate) when `inp` and `targ` are the same size."

    inp,targ = flatten_check(inp,targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp>thresh

    correct = pred==targ.bool()
    TN = torch.logical_and(correct,  (targ==0).bool()).sum()
    FP = torch.logical_and(~correct, (targ==0).bool()).sum()

    specificity = TN/(TN+FP)
    return specificity

def balanced_accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Computes balanced accuracy when `inp` and `targ` are the same size."

    inp,targ = flatten_check(inp,targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp>thresh

    correct = pred==targ.bool()
    TP = torch.logical_and(correct,  (targ==1).bool()).sum()
    TN = torch.logical_and(correct,  (targ==0).bool()).sum()
    FN = torch.logical_and(~correct, (targ==1).bool()).sum()
    FP = torch.logical_and(~correct, (targ==0).bool()).sum()

    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    balanced_accuracy = (TPR+TNR)/2
    return balanced_accuracy

def Fbeta_multi(inp, targ, beta=1.0, thresh=0.5, sigmoid=True):
    "Computes Fbeta when `inp` and `targ` are the same size."

    inp,targ = flatten_check(inp,targ)
    if sigmoid: inp = inp.sigmoid()
    pred = inp>thresh

    correct = pred==targ.bool()
    TP = torch.logical_and(correct,  (targ==1).bool()).sum()
    TN = torch.logical_and(correct,  (targ==0).bool()).sum()
    FN = torch.logical_and(~correct, (targ==1).bool()).sum()
    FP = torch.logical_and(~correct, (targ==0).bool()).sum()

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    beta2 = beta*beta

    if precision+recall > 0:
        Fbeta = (1+beta2)*precision*recall/(beta2*precision+recall)
    else:
        Fbeta = 0
    return Fbeta

def F1_multi(*args, **kwargs):
    return Fbeta_multi(*args, **kwargs)  # beta defaults to 1.0

# Cell
def mae(inp,targ):
    "Mean absolute error between `inp` and `targ`."
    inp,targ = flatten_check(inp,targ)
    return torch.abs(inp - targ).mean()

# Cell
def mape(inp,targ):
    "Mean absolute percentage error between `inp` and `targ`."
    inp,targ = flatten_check(inp, targ)
    return (torch.abs(inp - targ) / torch.clamp_min(targ, 1e-8)).mean()

# Cell
def _recall_at_specificity(inp, targ, specificity=.95, axis=-1):
    inp0 = inp[targ == 0]
    inp1 = inp[targ == 1]
    thr = torch.sort(inp0).values[-int(len(inp0) * (1 - specificity))]
    return (inp1 > thr).float().mean()

recall_at_specificity = AccumMetric(_recall_at_specificity, specificity=.95, activation=ActivationType.BinarySoftmax, flatten=False)

# Cell
def _mean_per_class_accuracy(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None):
    cm = skm.confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight, normalize=normalize)
    return (cm.diagonal() / cm.sum(1)).mean()

mean_per_class_accuracy = skm_to_fastai(_mean_per_class_accuracy)