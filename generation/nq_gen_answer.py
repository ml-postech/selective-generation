import os, sys
import json
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    AdamW,
    get_scheduler,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    GPT2LMHeadModel,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


import torch.nn as nn
import torch as tc
from tqdm import tqdm
from torch.nn.functional import cosine_similarity, normalize
from datasets import load_metric, Dataset
import numpy as np


import pickle
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path

# from models import ConformalLG, QnALLMWrapper
import models
from torch.utils.data import DataLoader
# Check if the minimal version of Transformers is not installed.
check_min_version("4.21.0.dev0")

class ModelWrapper(nn.Module):
    def __init__(self, model, tokenizer, max_length=50):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length=50

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
    def _unify_model_output(self, **kwargs):
        return {**kwargs, 'logprobs': nn.LogSoftmax(dim=-1)(kwargs['logits'])}



def main():
    torch.cuda.empty_cache()
    root_dir = os.path.join(Path.home(), 'sg-llm/data/nli')
    split_file_list = ['nq_paired_data_labeled_stanza.json', 'nq_paired_data_Z_un_stanza.json']
    model_name_or_path = 'data/models/llama/13B_hf'
    
    # Set seed before initializing model.
    set_seed(42)
    batch_size=8

    base_model = AutoModelForCausalLM.from_pretrained(f'{str(Path.home())}/{model_name_or_path}', device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(f'{str(Path.home())}/{model_name_or_path}', use_fast=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.ConformalLG(base_model=models.QnALLMWrapper(base_model, tokenizer), generation_type='greedy', gen_len=50)
    # model.to(device)
    model.eval()

    def tokenize_function(examples):

        # concat
        question_list = examples['question']
        # MJ: CAUTION!!
        # may not be used except for PEFT
        answer_list = examples['answer']#examples['transformed_answer'] if ('transformed_answer' in examples) and (training_args.prompt_model_path is None) else examples['answer']

        # tokenize
        tokenized_questions = tokenizer(question_list, add_special_tokens=False)
        tokenized_answers = tokenizer(answer_list, add_special_tokens=False)

        output = {
            'input_ids': [[tokenizer.bos_token_id] + q + a + [tokenizer.eos_token_id] for q, a in zip(tokenized_questions['input_ids'], tokenized_answers['input_ids'])],
            'attention_mask': [[1] + q + a + [1] for q, a in zip(tokenized_questions['attention_mask'], tokenized_answers['attention_mask'])],
            'answer_mask': [[0]*(len(q) + 1) + a + [1] for q, a in zip(tokenized_questions['attention_mask'], tokenized_answers['attention_mask'])],
        }

        return output
    
    def collate_fn(batch):
        max_length_batch = max(len(b['input_ids']) for b in batch)
        #max_length = tokenizer.max_length
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.bos_token_id
        # print(tokenizer.pad_token_id)
        input_ids = []
        attention_mask = []
        answer_mask = []
        for i in range(len(batch)):
            assert len(batch[i]['input_ids']) == len(batch[i]['attention_mask']) == len(batch[i]['answer_mask']), \
                f"{len(batch[i]['input_ids'])} == {len(batch[i]['attention_mask'])} == {len(batch[i]['answer_mask'])}"
            # MJ: left padding
            n_pad = max_length_batch - len(batch[i]['input_ids'])
            input_ids.append(tc.tensor([pad_token_id]*n_pad + batch[i]['input_ids']))
            attention_mask.append(tc.tensor([0]*n_pad + batch[i]['attention_mask']))
            answer_mask.append(tc.tensor([0]*n_pad + batch[i]['answer_mask'])) # 0
        input_ids = tc.vstack(input_ids)
        attention_mask = tc.vstack(attention_mask)

        answer_mask = tc.vstack(answer_mask)

        label = answer_mask

        # return
        batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'answer_mask': answer_mask}
        label = answer_mask
        
        return batch, label  #(x, y) format
    
    for split_file in split_file_list:
        data_json = json.load(open(os.path.join(root_dir, split_file), 'r'))
        temp = []
        for data in data_json:
            if data.pop('whether_transformed'):
                # data['question'] = 'when did the gupta family arrive'
                # data['question'] = 'Question: ' + data['question'] + '\nAnswer: '
                data['original_question'] = data['question']
                data['question'] = f"""Answer the question in a sentence:

when did dublin celebrate the millennium of its birth
Dublin celebrated the millennium of its birth in 1988.
{tokenizer.eos_token}
what's the first book in the pretty little liars series
The first book in the Pretty Little Liars series is \"Pretty Little Liars.\"
{tokenizer.eos_token}
who sings the song eye in the sky
The song \"Eye in the Sky\" is sung by The Alan Parsons Project.
{tokenizer.eos_token}
{data['question']}
"""
                # data['question'] = 'A single sentence to the question ' + f'"{data["question"]}" ' + 'is '
                temp.append(data)
        data_json = temp
        raw_dataset = Dataset.from_list(data_json)
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=64,
            remove_columns=raw_dataset.column_names,
            desc="running tokenization"
        )

        dataloader = DataLoader(
            tokenized_dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=False,
            num_workers=64
        )
        for idx, (batch, label) in enumerate(tqdm(dataloader)):
            with tc.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model.generate(batch)
                
                generated_sequences = tokenizer.batch_decode(output['answer_ids_pred'], skip_special_tokens=False)
                logprobs_answer_preds = output['logprobs_answer_pred']
                for jdx in range(len(output['logprobs_answer_pred'])):
                    data_json[idx*batch_size + jdx]['question'] = data_json[idx*batch_size + jdx].pop('original_question')
                    data_json[idx*batch_size + jdx]['generated_sequence'] = generated_sequences[jdx]
                    data_json[idx*batch_size + jdx]['transformed_sequence'] = data_json[idx*batch_size + jdx].pop('transformed')
                    data_json[idx*batch_size + jdx]['id'] = idx*batch_size + jdx
                    data_json[idx*batch_size + jdx]['label'] = None
                    data_json[idx*batch_size + jdx]['logprobs'] = logprobs_answer_preds[jdx].tolist()
                # print(j)
            # break

        json.dump(data_json, open(os.path.join(root_dir, 'nq_llama13B', split_file), 'w'), indent=4)
        # json.dump(nq_dataset, open(sys.argv[1], 'w'), indent=4)
        # write
        #data_jsonl = '\n'.join([json.dumps(d) for d in data])
        #open(os.path.join(root_dir, 'nq_udep.jsonl'), 'w').write(data_jsonl)

if __name__== '__main__':
    main()
