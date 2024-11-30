import sys, os
import numpy as np
from scipy import stats
import math

import torch as tc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages




def compute_error_per_label(ld, mdl, loss_fn, device):
    loss_vec, y_vec = [], []
    with tc.no_grad():
        for x, y in ld:
            loss_dict = loss_fn(x, y, mdl, reduction='none', device=device)
            loss_vec.append(loss_dict['loss'])
            y_vec.append(y)
    loss_vec, y_vec = tc.cat(loss_vec), tc.cat(y_vec)
    error_label = []
    for y in set(y_vec.tolist()):
        error_label.append(loss_vec[y_vec==y].mean().item())
    return error_label
        

def to_device(x, device):
    if tc.is_tensor(x):
        x = x.to(device)  
    elif isinstance(x, dict):
        x = {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        x = [to_device(x_i, device) for x_i in x]
    elif isinstance(x, tuple):
        x = (to_device(x_i, device) for x_i in x)
    else:
        try:
            x = x.to(device)
        except:
            print(f'the type of x = {type(x)}')
            raise NotImplementedError
    return x


# def bci_clopper_pearson(k, n, alpha):
#     lo = stats.beta.ppf(alpha/2, k, n-k+1)
#     hi = stats.beta.ppf(1 - alpha/2, k+1, n-k)
#     lo = 0.0 if math.isnan(lo) else lo
#     hi = 1.0 if math.isnan(hi) else hi
    
#     return lo, hi

# def estimate_bin_density(k, n, alpha):
#     lo, hi = bci_clopper_pearson(k, n, alpha)
#     return lo, hi

    
