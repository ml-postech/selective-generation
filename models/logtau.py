import os, sys
import numpy as np

import torch as tc
from torch import nn


class ScalarModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.logtau = nn.Parameter(tc.tensor([-np.inf], dtype=tc.float32))

        
    def set_tau(self, tau):
        self.logtau.data = tc.tensor([tau], dtype=tc.float32, device=self.logtau.data.device).log()
        

    def forward(self, hidden_states=None, logprobs=None):
        return {'logits': self.logtau.data}

    
    
    # def forward(self, x):
        
        # if len(x.shape) == 1:
        #     logits = self.logtau.data
        # else:
        #     assert(len(x.shape) >= 2)
        #     logits = self.logtau.data.expand(x.shape[0], 1)
        # return {'logits': logits}

    
# do not use
class EpsThresholdModel(nn.Module):
    
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        #print('eps =', self.eps)

        
    def forward(self, logprob):

        prob = logprob.exp()
        prob_sorted, prob_sorted_index = prob.sort(descending=False, dim=-1)

        prob_sorted_cumsum = tc.cumsum(prob_sorted, dim=-1)
        
        tau_index = tc.argmax((prob_sorted_cumsum > self.eps).int(), dim=-1, keepdim=True)
        tau = tc.gather(prob_sorted, 1, tc.gather(prob_sorted_index, 1, tau_index))
        logits = tau.log()
        
        # if len(x.shape) == 1:
        #     logits = self.logtau.data
        # else:
        #     assert(len(x.shape) >= 2)
        #     logits = self.logtau.data.expand(x.shape[0], 1)
        
        return {'logits': logits}
        
