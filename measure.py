from __future__ import print_function, division

import numpy as np
from sklearn.metrics import confusion_matrix
import torch

def evaluate(predictions, gts, num_classes):
    conmatrix = np.zeros((num_classes, num_classes))
    labels = np.arange(num_classes).tolist()
    predictions = predictions.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
    # gts = gts.data.squeeze(0).cpu().numpy()
    gts = gts.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
    for idx in range(predictions.shape[0]):
        lp = predictions[idx,:,:]
        lt = gts[idx,:,:]
        # lp = torch.argmax(lp[ :, :, :].squeeze(), dim=0).detach().cpu().numpy()
        lp = lp[:, :] > 0.5
        lp[lt == 1] = 1
        # lt[lt < 0] = -1
        conmatrix += confusion_matrix(lt.flatten(), lp.flatten(), labels=labels)

    M, N = conmatrix.shape
    tp = np.zeros(M, dtype=np.uint)
    fp = np.zeros(M, dtype=np.uint)
    fn = np.zeros(M, dtype=np.uint)

    for i in range(M):
        tp[i] = conmatrix[i, i]
        fp[i] = np.sum(conmatrix[:, i]) - tp[i]
        fn[i] = np.sum(conmatrix[i, :]) - tp[i]

    precision = tp / (tp + fp+ 1e-10)  # = tp/col_sum
    recall = tp / (tp + fn+ 1e-10)
    f1_score = 2 * recall * precision / (recall + precision+ 1e-10)

    ax_p = 0  # column of confusion matrix
    # ax_t = 1  # row of confusion matrix
    acc = np.diag(conmatrix).sum() / conmatrix.sum()
    acc_cls = np.diag(conmatrix) / conmatrix.sum(axis=ax_p)
    acc_cls = np.nanmean(acc_cls)
    iu = tp / (tp + fp + fn+ 1e-10)
    mean_iu = np.nanmean(iu)
    freq = conmatrix.sum(axis=ax_p) / conmatrix.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, iu, np.nanmean(f1_score)



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count