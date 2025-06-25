import random
import math
import numpy as np
import argparse
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torchvision import transforms
from dataloader.loading import *
import torch.nn.functional as F

def set_random_seed(seed):
    print(f"\n* Set seed {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def sizeof_fmt(num, suffix='B'):
    """
    https://stackoverflow.com/questions/24455615/python-how-to-display-size-of-all-variables
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


# print("Check memory usage of different variables:")
# for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
#                          key=lambda x: -x[1])[:10]:
#     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def get_optimizer(config_optim, parameters):
    if config_optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay,
                          betas=(config_optim.beta1, 0.999), amsgrad=config_optim.amsgrad,
                          eps=config_optim.eps)
    elif config_optim.optimizer == 'AdamW':
        return optim.AdamW(parameters, lr=1e-3, weight_decay=0.05,
                          betas=(config_optim.beta1, 0.999),
                          eps=1e-8)
    elif config_optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay)
    elif config_optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config_optim.lr, weight_decay=1e-4, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config_optim.optimizer))


def get_optimizer_and_scheduler(config, parameters, epochs, init_epoch):
    scheduler = None
    optimizer = get_optimizer(config, parameters)
    if hasattr(config, "T_0"):
        T_0 = config.T_0
    else:
        T_0 = epochs // (config.n_restarts + 1)
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   T_0=T_0,
                                                                   T_mult=config.T_mult,
                                                                   eta_min=config.eta_min,
                                                                   last_epoch=-1)
        scheduler.last_epoch = init_epoch - 1
    return optimizer, scheduler


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.training.warmup_epochs:
        lr = config.optim.lr * epoch / config.training.warmup_epochs
    else:
        lr = config.optim.min_lr + (config.optim.lr - config.optim.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - config.training.warmup_epochs) / (
                     config.training.n_epochs - config.training.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_dataset(config):
    data_object = None
    if config.data.dataset == "PLACENTAL":
        train_dataset = BUDataset(data_list=config.data.traindata, train=True)
        test_dataset = BUDataset(data_list=config.data.testdata, train=False)
    elif config.data.dataset == "APTOS":
        train_dataset = APTOSDataset(data_list=config.data.traindata, train=True)
        test_dataset = APTOSDataset(data_list=config.data.testdata, train=False)
    elif config.data.dataset == "ISIC":
        train_dataset = ISICDataset(data_list=config.data.traindata, train=True)
        test_dataset = ISICDataset(data_list=config.data.testdata, train=False)
    elif config.data.dataset == "CHEST":

        train_dataset = ChestXrayDataSet(image_list_file=config.data.traindata, train=True)
        test_dataset = ChestXrayDataSet(image_list_file=config.data.testdata, train=False)
    else:
        raise NotImplementedError(
            "Options: toy (classification of two Gaussian), MNIST, FashionMNIST, CIFAR10.")
    return data_object, train_dataset, test_dataset

from sklearn.metrics import cohen_kappa_score
# ------------------------------------------------------------------------------------
# Revised from timm == 0.3.2
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py
# output: the prediction from diffusion model (B x n_classes)
# target: label indices (B)
# ------------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    maxk = min(max(topk), output.size()[1])
    # output = torch.softmax(-(output - 1)**2,  dim=-1)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]



def cohen_kappa(output, target, topk=(1,)):
    maxk = min(max(topk), output.size()[1])
    _, pred = output.topk(maxk, 1, True, True)
    kappa = cohen_kappa_score(pred, target, weights='quadratic')
    return kappa


def cast_label_to_one_hot_and_prototype(y_labels_batch, config, return_prototype=True):
    """
    y_labels_batch: a vector of length batch_size.
    """
    y_one_hot_batch = nn.functional.one_hot(y_labels_batch, num_classes=config.data.num_classes).float()
    if return_prototype:
        label_min, label_max = config.data.label_min_max
        y_logits_batch = torch.logit(nn.functional.normalize(
            torch.clip(y_one_hot_batch, min=label_min, max=label_max), p=1.0, dim=1))
        return y_one_hot_batch, y_logits_batch
    else:
        return y_one_hot_batch




import numpy as np
import sklearn.metrics as metrics
#from imblearn.metrics import sensitivity_score, specificity_score
import pdb
# from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def compute_isic_metrics(gt, pred):
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()

    gt_class = np.argmax(gt_np, axis=1)
    pred_class = np.argmax(pred_np, axis=1)

    ACC = accuracy_score(gt_class, pred_class)
    BACC = balanced_accuracy_score(gt_class, pred_class) # balanced accuracy
    Prec = precision_score(gt_class, pred_class, average='macro')
    Rec = recall_score(gt_class, pred_class, average='macro')
    F1 = f1_score(gt_class, pred_class, average='macro')
    try:
        AUC_ovo = metrics.roc_auc_score(gt_np, pred_np)
    except:
        AUC_ovo=0.
    #AUC_macro = metrics.roc_auc_score(gt_class, pred_np, average='macro', multi_class='ovo')

    #SPEC = specificity_score(gt_class, pred_class, average='macro')

    kappa = cohen_kappa_score(gt_class, pred_class, weights='quadratic')

    # cm = confusion_matrix(gt_class, pred_class)

    # sns.heatmap(cm, 
    #         annot=True,
    #         fmt='g')
    # plt.ylabel('Prediction',fontsize=13)
    # plt.xlabel('Actual',fontsize=13)
    # plt.title('Confusion Matrix',fontsize=17)
    # plt.savefig('confusion_matrix_placental.png')

    # print(confusion_matrix(gt_class, pred_class))
    return ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa
    #return ACC, BACC, Prec, Rec, F1, AUC_ovo, AUC_macro, SPEC, kappa

def compute_f1_score(gt, pred):
    gt_class = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()

    #gt_class = np.argmax(gt_np, axis=1)
    pred_class = np.argmax(pred_np, axis=1)

    F1 = f1_score(gt_class, pred_class, average='macro')
    #AUC_ovo = metrics.roc_auc_score(gt_np, pred_np, average='macro', multi_class='ovo')
    #AUC_macro = metrics.roc_auc_score(gt_class, pred_np, average='macro', multi_class='ovo')

    #SPEC = specificity_score(gt_class, pred_class, average='macro')

    # print(confusion_matrix(gt_class, pred_class))
    return F1


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

from sklearn.metrics import roc_auc_score
def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    #AUROCs_mean = roc_auc_score(gt_np[:, 0], pred_np[:, 0])
    pred_np[np.where(np.isnan(pred_np))] = 0.
    for i in range(15):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i],average=None))
    AUROCs_mean = np.array(AUROCs[:-1]).mean()
    print(AUROCs)
    return AUROCs_mean



import scipy.special

# for left-multiplication for RGB -> Y'PbPr
RGB_TO_YUV = np.array([[0.29900, -0.16874, 0.50000],
                       [0.58700, -0.33126, -0.41869],
                       [0.11400, 0.50000, -0.08131]])


def normalize_data(x, mode=None):
    if mode is None or mode == 'rgb':
        return x / 127.5 - 1.
    elif mode == 'rgb_unit_var':
        return 2. * normalize_data(x, mode='rgb')
    elif mode == 'yuv':
        return (x / 127.5 - 1.).dot(RGB_TO_YUV)
    else:
        raise NotImplementedError(mode)


def log_min_exp(a, b, epsilon=1.e-6):
    """Computes the log(exp(a) - exp(b)) (b<a) in a numerically stable fashion."""
    y = a + torch.log1p(-torch.exp(b - a) + epsilon)
    return y


def categorical_kl_logits(logits1, logits2, eps=1.e-6):
    """KL divergence between categorical distributions.

    Distributions parameterized by logits.

    Args:
      logits1: logits of the first distribution. Last dim is class dim.
      logits2: logits of the second distribution. Last dim is class dim.
      eps: float small number to avoid numerical issues.

    Returns:
      KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
    """
    out = (F.softmax(logits1 + eps, dim=-1) * (
                F.log_softmax(logits1 + eps, dim=-1) - F.log_softmax(logits2 + eps, dim=-1)))
    return torch.sum(out, dim=-1)


def categorical_kl_probs(probs1, probs2, eps=1.e-6):
    """KL divergence between categorical distributions.

    Distributions parameterized by logits.

    Args:
      probs1: probs of the first distribution. Last dim is class dim.
      probs2: probs of the second distribution. Last dim is class dim.
      eps: float small number to avoid numerical issues.

    Returns:
      KL(C(probs) || C(logits2)): shape: logits1.shape[:-1]
    """
    out = probs1 * (torch.log(probs1 + eps) - torch.log(probs2 + eps))
    return torch.sum(out, dim=-1)


def categorical_log_likelihood(x, logits):
    """Log likelihood of a discretized Gaussian specialized for image data.

    Assumes data `x` consists of integers [0, num_classes-1].

    Args:
      x: where to evaluate the distribution. shape = (bs, ...), dtype=int32/int64
      logits: logits, shape = (bs, ..., num_classes)

    Returns:
      log likelihoods
    """
    log_probs = F.log_softmax(logits, dim=-1)
    x_onehot = F.one_hot(x.to(torch.int64), logits.shape[-1])
    return torch.sum(log_probs * x_onehot, dim=-1)


def meanflat(x):
    """Take the mean over all axes except the first batch dimension."""
    return x.mean(dim=tuple(range(1, len(x.shape))))