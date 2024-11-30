import os, sys
import numpy as np
import warnings

import torch as tc
import torch.nn as nn

from .base import *
from .util import *
from .logtau import *

from .llm import *

from datasets import load_dataset#, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from torch.utils.data import DataLoader

# TODO MJ: class
class EntailmentSet(ConformalSetModel):
    
    def __init__(self, **kwargs):
        self.model_args = kwargs.pop('model_args')
        self.training_args = kwargs.pop('training_args')
        self.data_args = kwargs.pop('data_args')
        self.rd = kwargs.pop('raw_datasets')
        super().__init__(**kwargs)

        #self.tokenizer, self.td, self.ld = self.init_dataset_nli(self.data_args, self.model_args, self.training_args)
        self.tokenizer = self.init_tokenizer(self.model_args)
        _, self.mdl = self.init_model_nli(self.training_args, self.model_args, self.tokenizer)

        self.logtau_model = ScalarModel()
        if 'init_tau' in kwargs:
            self.logtau_model.set_tau(kwargs['init_tau'])
        
        
    def eval_logtau(self, hidden_states=None, logprobs=None):
        return self.logtau_model()

    # TODO entailment check
    def classify(self, kwargs, labels):

        # logits, past_key_values, (hidden_states)
        output = self.mdl(kwargs)
        probs =  tc.softmax(output['logits'], dim=-1)
        output['probs'] = probs
        # MJ: contradict:0, neutral: 1, entail: 2
        output['labels'] = labels


        # scores = tc.hstack([lp.sum().exp() for lp in output['logprobs_answer_pred']])
        # output['scores'] = scores

        # # MJ: calibration?
        # if 'cal_probs' in output:
        #     scores_cal = output['cal_probs']
        #     output['scores_cal'] = scores_cal

        return output
    
    def init_tokenizer(self, model_args):
        # concat questions and answers and tokenize datasets
        tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir,
            "use_fast": model_args.use_fast_tokenizer,
            "revision": model_args.model_revision,
            # "additional_special_tokens": ["<|endofquestion|>"], # "end-of-question" token
        }
        if model_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        elif model_args.entail_model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(model_args.entail_model_name_or_path, **tokenizer_kwargs)
        else:
            raise NotImplementedError
        tokenizer.pad_token = tokenizer.eos_token


        # originally from init_dataset
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
        }
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        elif model_args.entail_model_name_or_path:
            config = AutoConfig.from_pretrained(model_args.entail_model_name_or_path, **config_kwargs)
        else:
            raise NotImplementedError
        
        # tokenizer max length
        if hasattr(config, 'max_position_embeddings'):
            tokenizer.max_length = config.max_position_embeddings
        else:
            tokenizer.max_length = tokenizer.model_max_length
        print('model max length = ', tokenizer.max_length)


        return tokenizer

    def init_model_nli(self, training_args, model_args, tokenizer):

        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
        }
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        elif model_args.entail_model_name_or_path:
            config = AutoConfig.from_pretrained(model_args.entail_model_name_or_path, **config_kwargs)
        else:
            raise NotImplementedError
        
        if model_args.entail_model_name_or_path:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.entail_model_name_or_path,
                from_tf=bool(".ckpt" in model_args.entail_model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                #device_map='auto',
                #trust_remote_code=model_args.trust_remote_code,
            )

        else: 
            raise NotImplementedError
        
        # to avoid index errors.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        # TODO deberta v2 does not support "device_map"
        model = model.to('cuda')
        wrapper_model = NliLLMWrapper(model, tokenizer)

        self.entail_model = wrapper_model
        return model, wrapper_model




    def init_dataset_nli(self, answer_preds=None, split=None):

        # for aesthetics
        data_args = self.data_args
        model_args = self.model_args
        training_args = self.training_args
        # raw dataset
        # For now, we use generated results, not from dataset
        # MJ: However, I checked those are same.

        raw_datasets = self.rd[split]
        # raw_datasets = rd.remove_columns("generated_answer")
        # raw_datasets = raw_datasets.add_column("generated_answer", answer_preds)
        
        tokenizer = self.tokenizer


        
        def tokenize_function(examples):
            tokenized_questions = tokenizer(examples['question'], add_special_tokens=False)
            tokenized_answers = tokenizer(examples['answer'], add_special_tokens=False)
            tokenized_a1 = tokenizer(examples['transformed_answer'], add_special_tokens=False)
            tokenized_a2 = tokenizer(examples['generated_answer'], add_special_tokens=False)

            labels = examples['labels']
            if not model_args.entail_model_name_or_path.startswith('potsawee'):
                tokenized_a1, tokenized_a2 = tokenized_a2, tokenized_a1

            # TODO MJ: added temporary value -1 to labels. Should be modified if the model will be trained.
            result = {
                'input_ids': [[tokenizer.bos_token_id] + s1 + [tokenizer.sep_token_id] + s2 + [tokenizer.eos_token_id] for s1, s2 in zip(tokenized_a1['input_ids'], tokenized_a2['input_ids'])],
                'attention_mask': [[1] + s1 +[1] + s2 + [1] for s1, s2 in zip(tokenized_a1['attention_mask'], tokenized_a2['attention_mask'])],
                'token_type_ids': [[0]*(1 + len(s1) +1) + [1]*(len(s2) + 1) for s1, s2 in zip(tokenized_a1['input_ids'], tokenized_a2['input_ids'])],
                'labels': [[label] if label is not None else [-1] for label in labels]
            }

            max_length = tokenizer.max_length
            input_ids_list = result['input_ids']
            attention_mask_list = result['attention_mask']
            token_type_ids_list = result['token_type_ids']

            # truncate
            input_ids_list = [i[-max_length:] for i in input_ids_list]
            attention_mask_list = [i[-max_length:] for i in attention_mask_list]
            token_type_ids_list = [i[-max_length:] for i in token_type_ids_list]

            return {
                'input_ids': input_ids_list,
                'attention_mask': attention_mask_list,
                'token_type_ids': token_type_ids_list,
                'labels': result['labels']
            }

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_datasets.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        
        # left_padding implementation from ncg-llm.
        def collate_fn(batch):
            max_length_batch = max(len(b['input_ids']) for b in batch)
            max_length = tokenizer.max_length
            pad_token_id = tokenizer.pad_token_id

            for i in range(len(batch)):
                assert len(batch[i]['input_ids']) == len(batch[i]['attention_mask']), \
                    f"{len(batch[i]['input_ids'])} == {len(batch[i]['attention_mask'])}"
                # right padding
                n_pad = max_length_batch - len(batch[i]['input_ids'])

                batch[i]['input_ids'] = tc.tensor(batch[i]['input_ids'] + [pad_token_id]*n_pad)
                batch[i]['attention_mask'] = tc.tensor(batch[i]['attention_mask'] + [0]*n_pad)
                batch[i]['token_type_ids'] = tc.tensor(batch[i]['token_type_ids'] + [pad_token_id]*n_pad)
                batch[i]['labels'] = tc.Tensor([batch[i]['labels']])

            labels = tc.tensor([b['labels'] for b in batch])
            batch = {
                'input_ids': tc.vstack([b['input_ids'] for b in batch]),
                'attention_mask': tc.vstack([b['attention_mask'] for b in batch]),
                'token_type_ids': tc.vstack([b['token_type_ids'] for b in batch]),
            }

            return batch, labels#, answer_mask, context_mask
        

        dataloader = DataLoader(
            tokenized_datasets,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=training_args.per_device_eval_batch_size if split == 'test' else training_args.per_device_train_batch_size,
            num_workers=training_args.dataloader_num_workers
        )
        
        return dataloader
        # return (
        #     #tokenizer,
        #     #raw_datasets,
        #     #tokenized_datasets,
        #     {
        #         #"train": train_dataloader,
        #         "validation": dataloader,
        #         #"test": test_dataloader,
        #     },
        # )
