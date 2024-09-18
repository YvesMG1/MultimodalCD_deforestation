import numpy as np

from sklearn.metrics import confusion_matrix

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False
    
    def reset(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class
        self.conf_matrix = np.zeros((n_class, n_class), dtype=int)

    def update(self, new_cm):
        self.conf_matrix += new_cm

    def get_scores(self):
        scores_dict = cm2score(self.conf_matrix)
        return scores_dict

    def reset(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.conf_matrix = np.zeros((self.n_class, self.n_class), dtype=int)



def cm2score(confusion_matrix):
    """ Compute the pixel-wise metrics from the confusion matrix """
    
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    # Accuracy
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # Recall
    recall = np.zeros(n_class)
    precision = np.zeros(n_class)
    F1 = np.zeros(n_class)
    iu = np.zeros(n_class)

    for i in range(n_class):
        if sum_a1[i] == 0 and sum_a0[i] == 0:
            recall[i] = 1.0
            precision[i] = 1.0
            F1[i] = 1.0
            iu[i] = 1.0
        else:
            recall[i] = tp[i] / (sum_a1[i] + np.finfo(np.float32).eps)
            precision[i] = tp[i] / (sum_a0[i] + np.finfo(np.float32).eps)
            F1[i] = 2 * recall[i] * precision[i] / (recall[i] + precision[i] + np.finfo(np.float32).eps)
            iu[i] = tp[i] / (sum_a1[i] + sum_a0[i] - tp[i] + np.finfo(np.float32).eps)

    mean_F1 = np.nanmean(F1)
    mean_iu = np.nanmean(iu)
    
    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    cls_iou = dict(zip(['iou_'+str(i) for i in range(n_class)], iu))
    cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(n_class)], F1))

    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1': mean_F1}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    
    return score_dict
