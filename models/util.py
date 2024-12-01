import os, sys
import numpy as np

import torch as tc
import torch.nn as nn
# import torchvision.transforms.functional as TF

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from transformers.data.metrics.squad_metrics import compute_exact, compute_f1, normalize_answer



# class Dummy(nn.Module):

#     def forward(self, x, y=None):
#         if y is not None:
#             x['logph'] = x['logph_y'] ##TODO: better way?
#         return x

# class NoCal(nn.Module):
#     def __init__(self, mdl, cal_target=-1):
#         super().__init__()
#         self.mdl = mdl
#         self.cal_target = nn.Parameter(tc.tensor(cal_target).long(), requires_grad=False)

        
#     def forward(self, x, training=False):
#         assert(training==False)
#         self.eval() ##always

#         ## forward along the base model
#         out = self.mdl(x)
#         if self.cal_target == -1:
#             ph = out['ph_top']
#         elif self.cal_target in range(out['ph'].shape[1]):
#             ph = out['ph'][:, self.cal_target]
#         else:
#             raise NotImplementedError
        
#         ## return
#         return {'yh_top': out['yh_top'],
#                 'yh_cal': out['yh_top'] if self.cal_target == -1 else tc.ones_like(out['yh_top'])*self.cal_target,
#                 'ph_cal': ph,
#         }


def dist_mah(xs, cs, Ms, sqrt=True):
    diag = True if len(Ms.size()) == 2 else False
    assert(diag)
    assert(xs.size() == cs.size())
    assert(xs.size() == Ms.size())

    diff = xs - cs
    dist = diff.mul(Ms).mul(diff).sum(1)
    if sqrt:
        dist = dist.sqrt()
    return dist


def neg_log_prob(yhs, yhs_logvar, ys, var_min=1e-16):

    d = ys.size(1)
    yhs_var = tc.max(yhs_logvar.exp(), tc.tensor(var_min, device=yhs_logvar.device))
    loss_mah = 0.5 * dist_mah(ys, yhs, 1/yhs_var, sqrt=False)
    # if not all(loss_mah >= 0):
    #     print('loss_mah', loss_mah)
    #     print('ys', ys)
    #     print('yhs', yhs)
    #     print('yhs_var', yhs_var)
    #     print('yhs_logvar', yhs_logvar)
    assert(all(loss_mah >= 0))
    loss_const = 0.5 * np.log(2.0 * np.pi) * d
    loss_logdet = 0.5 * yhs_logvar.sum(1)
    loss = loss_mah + loss_logdet + loss_const

    return loss

# def g(grad):
#     print('grad_prev:', grad)
#     grad = -grad
#     print('grad_post:', grad)

#     return grad

# class GradReversalLayer(nn.Module):
#     def __init__(self):
#         super().__init__()

        
#     def forward(self, x, training=False):
#         x = x * 1.0
#         if training:
#             if x.requires_grad:
#                 x.register_hook(lambda grad: -grad)
#                 #x.register_hook(g)
#         return x

    
# class ExampleNormalizer(nn.Module):
#     def __init__(self, mdl, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]): ## imagenet mean and std
#         super().__init__()
#         self.mdl = mdl
#         self.mean = mean
#         self.std = std

        
#     def forward(self, x, **kwargs):
#         x = TF.normalize(tensor=x, mean=self.mean, std=self.std)
#         x = self.mdl(x, **kwargs)
#         return x

    
# def box_inclusion(bb_small, bb_large):
#     ## bb_small is included in bb_large
#     ## assume xyxy format
#     if bb_large[0] <= bb_small[0] and bb_large[1] <= bb_small[1] and bb_small[2] <= bb_large[2] and bb_small[3] <= bb_large[3]:
#         return True
#     else:
#         return False


def pad_left(token_ids, pad_id, concat=True):
    max_len = max([len(l) for l in token_ids])
    
    token_padded_ids = []
    mask = []
    for token_ids_i in token_ids:
        len_i = len(token_ids_i)
        token_padded_ids.append(
            tc.hstack((tc.tensor([pad_id]*(max_len - len_i), dtype=tc.long, device=token_ids_i.device),  token_ids_i))
        )
        mask.append(
            tc.tensor([0]*(max_len - len_i) + [1]*len(token_ids_i), dtype=tc.long, device=token_ids_i.device)
        )
    if concat:
        token_padded_ids = tc.vstack(token_padded_ids)
        mask = tc.vstack(mask)

    return token_padded_ids, mask



def compute_inclusion(a_gold, a_pred):
    return normalize_answer(a_pred).find(normalize_answer(a_gold)) >= 0

def compute_EM(a_gold, a_pred):
    return normalize_answer(a_pred) == normalize_answer(a_gold)
