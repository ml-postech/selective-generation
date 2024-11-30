import os, sys
import warnings

import torch as tc
from torch import nn

from .base import *
from peft import PeftModel
class QnALLMWrapper(BaseModelWrapper):
    
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        
    def forward(self, kwargs):
        
        if 'training' in kwargs:
            assert(not kwargs['training'])

        input_ids = kwargs['input_ids']
        inputs_embeds = kwargs['inputs_embeds'] if 'inputs_embeds' in kwargs else None
        attention_mask = kwargs['attention_mask']
        #answer_mask = kwargs['answer_mask']
        use_cache = kwargs['use_cache'] if 'use_cache' in kwargs else False
        past_key_values = kwargs['past_key_values'] if 'past_key_values' in kwargs else None

        # PeftModelForCausalLM concatenates soft prompt
        # both when prepare_inputs_for_generation() and forward()
        model = self.model
        if isinstance(self.model, PeftModel):
            model = self.model.base_model
            
        #TODO: output returns all hidden layers, which is not efficient if we only need the last hidden layer
        with tc.no_grad():
            output = model(
                input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=use_cache, past_key_values=past_key_values,
                #output_hidden_states=True,
                return_dict=True)
        
        # logits and hidden states of all tokens
        logits = output['logits']
        if 'hidden_states' in output:
            hidden_states = output['hidden_states'][-1] # the hidden states from the last decoder layer

        # logits and hidden states of the last token
        logits = logits[:, -1, :]
        if 'hidden_states' in output:
            hidden_states = hidden_states[:, -1, :]

        # return
        if 'hidden_states' in output:
            return self._unify_model_output(
                logits=logits,
                hidden_states=hidden_states,
                past_key_values=output['past_key_values'] if 'past_key_values' in output else None
            )
        else:
            return self._unify_model_output(
                logits=logits,
                past_key_values=output['past_key_values'] if 'past_key_values' in output else None
            )


# TODO MJ
class NliLLMWrapper(BaseModelWrapper):
    
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        
    def forward(self, kwargs):
        
        if 'training' in kwargs:
            assert(not kwargs['training'])

        input_ids = kwargs['input_ids']
        inputs_embeds = kwargs['inputs_embeds'] if 'inputs_embeds' in kwargs else None
        attention_mask = kwargs['attention_mask']
        token_type_ids = kwargs['token_type_ids']
        # answer_mask = kwargs['answer_mask']
        # use_cache = kwargs['use_cache'] if 'use_cache' in kwargs else False
        # past_key_values = kwargs['past_key_values'] if 'past_key_values' in kwargs else None

        # # PeftModelForCausalLM concatenates soft prompt
        # # both when prepare_inputs_for_generation() and forward()
        model = self.model
        # if isinstance(self.model, PeftModel):
        #     model = self.model.base_model
            
        #TODO: output returns all hidden layers, which is not efficient if we only need the last hidden layer
        with tc.no_grad():
            output = model(
                input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids,
                #output_hidden_states=True,
                return_dict=True)
        
        # logits and hidden states of all tokens
        logits = output['logits']
        if 'hidden_states' in output:
            hidden_states = output['hidden_states'][-1] # the hidden states from the last decoder layer

        # # logits and hidden states of the last token
        # logits = logits[:, -1, :]
        # if 'hidden_states' in output:
        #     hidden_states = hidden_states[:, -1, :]

        # return
        if 'hidden_states' in output:
            return {
                'logits':logits,
                'hidden_states': hidden_states,
            }
        else:
            return {
                'logits': logits
            }

        # if 'hidden_states' in output:
        #     return self._unify_model_output(
        #         logits=logits,
        #         hidden_states=hidden_states,
        #         past_key_values=output['past_key_values'] if 'past_key_values' in output else None
        #     )
        # else:
        #     return self._unify_model_output(
        #         logits=logits,
        #         past_key_values=output['past_key_values'] if 'past_key_values' in output else None
        #     )


