import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import time


def confusion_matrix_segmentation(preds, targets, th):
    targets_binary = (targets > 0.5).view(targets.shape[0], -1).sum(dim=1).cpu() > 1
    preds_binary = (preds > th).view(preds.shape[0], -1).sum(dim=1).cpu() > 1

    return confusion_matrix(targets_binary.numpy(), preds_binary.numpy())/len(targets_binary)

def batch_dice_coeff(pred, target, th):
    return torch.tensor([dice_coeff(pred[i,:], target[i,:], th) for i in range(pred.shape[0])]).mean()

def dice_coeff(pred, target, th):
    smooth = 0.00001
    pred = pred > th
    target[target < 1] = 0
    target[target > 1] = 1
    
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def topk_accuracy(output : torch.Tensor, 
                  target : torch.Tensor, 
                  n_topk : int = 1):
    """
        Calculates top-k accuracy for pytorch tensors
    """
    topk = output.topk(n_topk)[1]
 
    correct = 0
    for k in range(n_topk):
        correct += topk[:,k].eq(target.view_as(topk[:,k])).sum().item()

    return correct/len(output)


def __transform_with_dict__(arr, d):
    res = []
    for i in range(len(arr)):
        if arr[i] in d:
            res.append(d[arr[i]])
        else:
            res.append(-1)
    return np.array(res)

def do_groupping(output, target, lookup_dict, n_topk=1):
    topk = output.topk(n_topk)[1]

    output_transformed = __transform_with_dict__(np.array([topk[i, n_topk-1].item() for i in range(len(topk))]), lookup_dict)
    target_transformed = __transform_with_dict__(np.array([target[i].item() for i in range(len(target))]), lookup_dict)

    return output_transformed, target_transformed

def grouped_accuracy(output, target, lookup_dict):
    """
    Calculates the accuracy, but first it replaces predicted and target values with one from the lookup table
    """
    topk = output.topk(1)[1]

    output_transformed = __transform_with_dict__(np.array([topk[i, 0].item() for i in range(len(topk))]), lookup_dict)
    target_transformed = __transform_with_dict__(np.array([target[i].item() for i in range(len(target))]), lookup_dict)

    return (output_transformed == target_transformed).mean()



class MetricsCallback:
    """
    Aggregator of different metrics, with it you can add metrics, 
    call update method, and get the aggregate mean result at the end
    """
    def __init__(self):
        self.names = []
        self.callbacks = []
        self.results = []
        self.start_time = 0
        self.reset() 

    def add_callback(self, name, callback):
        self.names.append(name)
        self.callbacks.append(callback)
        self.results.append([])

    def reset(self):
        self.total_len = 0
        self.start_time = time.time()

        for i in range(len(self.callbacks)):
            self.results[i] = []

    def update(self, output, target):
        assert output.shape[0] == target.shape[0]
        self.total_len += output.shape[0]

        for i in range(len(self.callbacks)):
            self.results[i].append(output.shape[0] * self.callbacks[i](output, target))

    def get(self):
        return [self.get_by_idx(i) for i in range(len(self.callbacks))]

    def get_by_name(self, name):
        idx = self.names.index(name)
        return self.get_by_idx(idx)
   
    def get_by_idx(self, idx):
        return np.sum(self.results[idx], axis=0) / self.total_len


    def __str__(self):
        s = ""
        results = self.get()

        for i in range(len(self.callbacks)):
            s += self.names[i] + ": " + str(results[i]) + "; "

        s += "time: " + str(time.time() - self.start_time)
        return s
