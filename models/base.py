
import os, sys
import numpy as np
from abc import abstractmethod

import torch as tc
from torch import nn

from .util import *
from uncertainty import ECE

class BaseModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()


    def _unify_model_output(self, **kwargs):
        return {**kwargs, 'logprobs': nn.LogSoftmax(dim=-1)(kwargs['logits'])}


class ConformalSetModel(nn.Module):
    """
    A conformal predictor
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.base_model = kwargs.get('base_model', None)
        #self.logtau_model = kwargs.get('logtau_model', None)
        self.eps = nn.Parameter(tc.tensor(kwargs.get('eps', 0.0), dtype=tc.float32), requires_grad=False)
        self.delta = nn.Parameter(tc.tensor(kwargs.get('delta', 0.0), dtype=tc.float32), requires_grad=False)
        self.n = nn.Parameter(tc.tensor(kwargs.get('n', 0), dtype=tc.int64), requires_grad=False)

        
    # def state_dict(self):
    #     # drop a base model
    #     sd = super().state_dict()
    #     #sd = {k: v for k, v in sd.items() if 'base_model.' not in k} # to save temperature
    #     return sd

    
    def load_state_dict(self, state_dict):
        assert('n' in state_dict.keys())
        assert('eps' in state_dict.keys())
        assert('delta' in state_dict.keys())
        assert(any(['logtau_model.' for k in state_dict.keys()]))    
        super().load_state_dict(state_dict, strict=False)


    # parameters for optimization
    def parameters(self):
        return self.base_model.parameters()
    
        
    @abstractmethod
    def eval_logtau(self, hidden_states=None, logprobs=None):
        raise NotImplementedError

    
    @abstractmethod
    def forward(self, x, y=None):
        raise NotImplementedError

    
    @abstractmethod
    def set(self, x):
        raise NotImplementedError

    
    @abstractmethod
    def membership(self, x, y):
        raise NotImplementedError

    
    @abstractmethod
    def size(self, x):
        raise NotImplementedError

    
class ConformalClassificationModel(ConformalSetModel):
    
    def forward(self, x, y=None):
        output = self.base_model(x)
        if y is not None:
            logprobs = output['logprobs']
            assert(logprobs.shape[0] == y.shape[0])
            logprobs = logprobs.gather(1, y.view(-1, 1)).squeeze(1)
            output['logprobs'] = logprobs
            
        return output
        
    
    def set(self, x, y=None, output=None, ideal=False):
        if output is None:
            with tc.no_grad():
                output = self.forward(x)
        logprobs = output['logprobs']
        if 'hidden_states' in output:
            hidden_states = output['hidden_states']
        else:
            hidden_states = None
    
        if ideal:
            if 'logprobs_y' in output:
                logprobs_y = output['logprobs_y']
            else:
                assert(y)
                logprobs_y = logprobs.gather(1, y.view(-1, 1))
            s = logprobs >= logprobs_y
        else:
            logtaus = self.eval_logtau(hidden_states=hidden_states, logprobs=logprobs)['logits']
            s = logprobs >= logtaus.unsqueeze(-1)
        return s

    
    # def membership(self, x, y, output=None, ideal=False):
    #     s = self.set(x, output=output, ideal=ideal)
    #     assert(s.shape[0] == y.shape[0])
    #     membership = s.gather(1, y.view(-1, 1)).squeeze(1)
    #     return membership, s

    
    def size(self, x, y=None, output=None, ideal=False):
        raise NotImplementedError #TODO: check the following
        s = self.set(x, y=y, output=output, ideal=ideal)
        # summary
        sz = s.sum(-1).float()
        return sz


    def error(self, x, y, output=None):
        raise NotImplementedError #TODO: check the following
        s = self.set(x, output=output)
        er = (s.gather(1, y.view(-1, 1)).squeeze(1) == 0).float()
        return er
        

    def error_size(self, x=None, y=None, output=None):
        with tc.no_grad():
            if output is None:
                output = self.forward(x)

            # error
            er = self.error(x, y, output=output)

            # size
            sz = self.size(x, output=output)
            
            # ideal size
            sz_ideal = self.size(x, y=y, output=output, ideal=True)
            
        return {'error': er, 'size': sz, 'size_ideal': sz_ideal}


class ConformalLM(ConformalClassificationModel):
    
    def size(self, x, y=None, output=None, ideal=False):

        #TODO: compute logprobs* when x and y are given
        
        logprobs = output['logprobs']
        logprobs_y = output['logprobs_y']
        # only consider the size stats with true answer length
        len_answers = [len(v) for v in output['logprobs_y']]

        if ideal:
            logtaus = logprobs_y
        else:
        
            logtaus = [v for v in self.eval_logtau(logprobs=logprobs)['logits']]
            
        # normalized size
        sz = tc.cat([(logprobs_i[:len_i,:] >= logtaus_i.unsqueeze(-1)[:len_i,:]).float().mean(-1).mean().unsqueeze(0)
                     for logprobs_i, logtaus_i, len_i in zip(logprobs, logtaus, len_answers)])

        return sz


    def error(self, x, y, output=None):

        #TODO: compute logprobs_y when x and y are given
        
        logprobs = output['logprobs']
        logprobs_y = output['logprobs_y']

        logtaus = self.eval_logtau(logprobs=logprobs)['logits']

        er = tc.tensor([(logprobs_y_i < logtaus_i[:len(logprobs_y_i)]).any() for logprobs_y_i, logtaus_i in zip(logprobs_y, logtaus)])

        return er



class ConformalLG(ConformalLM):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generation_type = kwargs['generation_type']
        self.gen_len = kwargs['gen_len']


    def is_correct(self, a_gold, a_pred, metric='em'):
        
        return compute_EM(a_gold, a_pred) if metric == 'em' else compute_inclusion(a_gold, a_pred)

    
    def decode_inputs(self, kwargs):
        input_ids = kwargs['input_ids']
        attention_mask = kwargs['attention_mask']
        answer_mask = kwargs['answer_mask']

        question_mask = attention_mask & (~answer_mask)
        question_ids = [input_ids_i[question_mask_i.bool()] for input_ids_i, question_mask_i in zip(input_ids, question_mask)]
        answer_ids = [input_ids_i[answer_mask_i.bool()] for input_ids_i, answer_mask_i in zip(input_ids, answer_mask)]

        return {
            'question_mask': question_mask,
            'question_ids': question_ids,
            'answer_ids': answer_ids,
        }
        

    def generate_onestep(self, kwargs):

        
        # if ('past_key_values' in kwargs) and (kwargs['past_key_values'] is not None):
        #     input_ids = kwargs['input_ids']
        #     past_key_values = kwargs['past_key_values']
        #     past_length = past_key_values[0][0].shape[2]

        #     # Some generation methods already pass only the last input ID
        #     if input_ids.shape[1] > past_length:
        #         remove_prefix_length = past_length
        #     else:
        #         # Default to old behavior: keep only final ID
        #         remove_prefix_length = input_ids.shape[1] - 1

        #     input_ids = input_ids[:, remove_prefix_length:]
        #     kwargs['input_ids'] = input_ids

        
        
        # # create position_ids on the fly for batch generation
        # if 'position_ids' not in kwargs:
        #     position_ids = kwargs['attention_mask'].long().cumsum(-1) - 1
        #     position_ids.masked_fill_(kwargs['attention_mask'] == 0, 1)

        #     past_key_values = kwargs['past_key_values'] if 'past_key_values' in kwargs else None
        #     if past_key_values:
        #         position_ids = position_ids[:, -kwargs['input_ids'].shape[1] :]
                
        #     kwargs['position_ids'] = position_ids

        # forward
        # MJ: PEFT 0.7.1 does not fit to this code.
        kwargs_processed = self.base_model.model.prepare_inputs_for_generation(**kwargs)
        for k, v in kwargs_processed.items():
            kwargs[k] = v
        output = self.forward(kwargs)

        
        # decison
        probs = output['logprobs'].exp()
        if kwargs['generation_type'] == 'greedy':
            # greedy decoding
            gen_id = tc.argmax(probs, dim=-1, keepdim=True)
        elif kwargs['generation_type'] == 'random':
            
            raise NotImplementedError
            # get a prediction set
            set_membership = self.set(None, output=output)

            probs[~set_membership] = 0
            probs = probs / probs.sum(-1, keepdim=True)
            if self.sampling:
                gen_id = tc.multinomial(probs, num_samples=1)
            
        elif kwargs['generation_type'] == 'labeled':
            gen_id = kwargs['current_answer_ids'].unsqueeze(1)
        else:
            raise NotImplementedError
            
        # # get a prediction set
        # set_membership = self.set(None, output=output)
        
        # # generate the next token
        # probs[~set_membership] = 0
        # probs = probs / probs.sum(-1, keepdim=True)
        # if self.sampling:
        #     gen_id = tc.multinomial(probs, num_samples=1)
        # else:
        #     # greedy decoding
        #     gen_id = tc.argmax(probs, dim=-1, keepdim=True)

        return {'gen_id': gen_id, **output}


    def generate(self, kwargs):

        # init
        decoded_inputs = self.decode_inputs(kwargs)
        question_mask = decoded_inputs['question_mask']
        question_ids = decoded_inputs['question_ids']
        answer_ids = decoded_inputs['answer_ids']
            
        # generate answers for each batch
        question_leftpad_ids, question_leftpad_mask = pad_left(question_ids, pad_id=self.base_model.tokenizer.bos_token_id)
        input_ids = question_leftpad_ids
        attention_mask = question_leftpad_mask
        answer_ids_pred = []
        logprobs_list = []
        hidden_states_list = []
        terminated = tc.tensor([False]*input_ids.shape[0])
        past_key_values = None
        for i_token in range(self.gen_len):
            output = self.generate_onestep(
                {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'use_cache': True,
                    'past_key_values': past_key_values,
                    'generation_type': kwargs['generation_type'] if 'generation_type' in kwargs else self.generation_type,
                    'current_answer_ids': tc.hstack(
                        [a[i_token] if i_token < len(a) else tc.tensor(self.base_model.tokenizer.eos_token_id, device=a.device)
                         for a in answer_ids])
                })
            past_key_values = output['past_key_values']
            gen_id_i = output['gen_id']

            terminated[gen_id_i.squeeze() == self.base_model.tokenizer.eos_token_id] = True

            # append generated tokens
            input_ids = tc.hstack((input_ids, gen_id_i.to(input_ids.device)))            
            attention_mask = tc.hstack((attention_mask, tc.ones(attention_mask.shape[0], 1, device=attention_mask.device, dtype=attention_mask.dtype)))

            # keep generated tokens
            gen_id_i_keep = gen_id_i
            gen_id_i_keep[terminated] = self.base_model.tokenizer.eos_token_id
            answer_ids_pred.append(gen_id_i_keep)
            logprobs_list.append(output['logprobs'])
            if 'hidden_states' in output:
                hidden_states_list.append(output['hidden_states'])
            
            if all(terminated):
                break

        
        # truncate predicted answers
        answer_ids_pred = tc.hstack(answer_ids_pred)
        answer_ids_pred = [a[:(a==self.base_model.tokenizer.eos_token_id).long().argmax()]
                           if any(a==self.base_model.tokenizer.eos_token_id)
                           else
                           a
                           for a in answer_ids_pred]

        
        # get log-probs
        logprobs = tc.cat([v.unsqueeze(1) for v in logprobs_list], dim=1)
        logprobs_answer = [lp[:min(len(lp), len(a))].gather(-1, a[:min(len(lp), len(a))].view(-1, 1)).squeeze(1)
                           for lp, a in zip(logprobs, answer_ids)]
        logprobs_answer_pred = [lp[:min(len(lp), len(a))].gather(-1, a[:min(len(lp), len(a))].view(-1, 1)).squeeze(1)
                                for lp, a in zip(logprobs, answer_ids_pred)]

        #
        if hidden_states_list:
            hidden_states = tc.cat([v.unsqueeze(1) for v in hidden_states_list], dim=1)
        else:
            hidden_states = None
        
        out = {
            'question_ids': question_ids,
            'answer_ids': answer_ids,
            'answer_ids_pred': answer_ids_pred,
            'logprobs': logprobs,
            'logprobs_answer': logprobs_answer,
            'logprobs_answer_pred': logprobs_answer_pred,
        }
        if hidden_states:
            out['hidden_states'] = hidden_states
        
        return out

    
        
class ConformalSG(ConformalLM):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.G = kwargs['generator']
        

    def generate(self, kwargs):
        return self.G(kwargs)

        
        

