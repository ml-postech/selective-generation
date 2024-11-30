import os, sys
import numpy as np
import warnings

import torch as tc
import torch.nn as nn

from .base import *
from .util import *
from .logtau import *

class PrecisionSG(ConformalSG):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logtau_model = ScalarModel()
        if 'init_tau' in kwargs:
            self.logtau_model.set_tau(kwargs['init_tau'])
        
        
    def eval_logtau(self, hidden_states=None, logprobs=None):
        return self.logtau_model()

    
    def generate(self, kwargs):
        output = self.G.generate(kwargs)
        scores = tc.hstack([lp.mean().exp() for lp in output['logprobs_answer_pred']])
        output['scores'] = scores

        # MJ: calibration?
        if 'cal_probs' in output:
            scores_cal = output['cal_probs']
            output['scores_cal'] = scores_cal
            
        # self.logtau_model = self.logtau_model.to(scores.device)    
        # output['selection'] = scores >= self.eval_logtau()['logits'].exp()
        return output


