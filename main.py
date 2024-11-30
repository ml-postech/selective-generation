import logging
#import math
import os
import sys
#from itertools import chain
import warnings
import pickle
import io
import numpy as np
import time
import types


import datasets
from datasets import concatenate_datasets, load_dataset#, load_metric

import transformers
from transformers import (
    #CONFIG_MAPPING,
    #MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    #TrainingArguments,
    default_data_collator,
    # is_torch_tpu_available,
    set_seed,
)
from transformers.utils import check_min_version

# prompt learning
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, PeftModelForCausalLM

import torch as tc
from torch.utils.data import DataLoader

import models
import uncertainty
import util

from nltk.corpus import wordnet

from transformers.testing_utils import CaptureLogger
# from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version, send_example_telemetry
# from transformers.utils.versions import require_version

from args import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.21.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

# logger = logging.getLogger(__name__)



            
def tokenize_function(tokenizer, examples):
    # tokenize
    output = tokenizer(examples['text'], truncation=True)

    return output


def init_tokenizer(model_args):
    # concat questions and answers and tokenize datasets
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        # "additional_special_tokens": ["<|endofquestion|>"], # "end-of-question" token
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise NotImplementedError
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def init_datasets_qna(data_args, model_args, training_args):
    
    # load datasets
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
    )

    # For the purpose of init_dataset_nli(). (also 'val2')
    # val1 for unlabeled train dataset, val2 for labeled train dataset.
    raw_datasets['val1'] = raw_datasets['train']

    raw_datasets['val2'] = raw_datasets['validation']
    raw_datasets['val1+2'] = concatenate_datasets([raw_datasets['val1'], raw_datasets['val2']])

    # if 'logprobs' in the dataset.
    # if 'logprobs' in raw_datasets['test'].column_names and raw_datasets['test'][0]['logprobs'] is not None or model_args.model_name_or_path.startswith('gpt'):
    #     return None, raw_datasets, None, None

    tokenizer = init_tokenizer(model_args)
    

    # model config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
    }
    if model_args.config_name:
        model_config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise NotImplementedError

    # set the max length of context
    if hasattr(model_config, 'max_position_embeddings'):
        tokenizer.max_length = model_config.max_position_embeddings
    else:
        tokenizer.max_length = tokenizer.model_max_length
    print(f'[dataset] model max length (from a tokenizer) = {tokenizer.max_length}')
    print(f'[dataset] voc size = {len(tokenizer)}')
    
    def tokenize_function(examples):

        question_list = examples['question']
        # CAUTION, this part may not be used except for PEFT
        answer_list = examples['answer']

        # tokenize
        tokenized_questions = tokenizer(question_list, add_special_tokens=False)
        tokenized_answers = tokenizer(answer_list, add_special_tokens=False)

        output = {
            'input_ids': [[tokenizer.bos_token_id] + q + a + [tokenizer.eos_token_id] for q, a in zip(tokenized_questions['input_ids'], tokenized_answers['input_ids'])],
            'attention_mask': [[1] + q + a + [1] for q, a in zip(tokenized_questions['attention_mask'], tokenized_answers['attention_mask'])],
            'answer_mask': [[0]*(len(q) + 1) + a + [1] for q, a in zip(tokenized_questions['attention_mask'], tokenized_answers['attention_mask'])],
        }

        return output

    with training_args.main_process_first(desc="dataset tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets["test"].column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running a tokenizer on datasets",
        )
    
    # build data loaders
    def collate_fn(batch):
        max_length_batch = max(len(b['input_ids']) for b in batch)
        #max_length = tokenizer.max_length
        pad_token_id = tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        answer_mask = []
        for i in range(len(batch)):
            assert len(batch[i]['input_ids']) == len(batch[i]['attention_mask']) == len(batch[i]['answer_mask']), \
                f"{len(batch[i]['input_ids'])} == {len(batch[i]['attention_mask'])} == {len(batch[i]['answer_mask'])}"
            # left padding
            n_pad = max_length_batch - len(batch[i]['input_ids'])
            input_ids.append(tc.tensor([pad_token_id]*n_pad + batch[i]['input_ids']))
            attention_mask.append(tc.tensor([0]*n_pad + batch[i]['attention_mask']))
            answer_mask.append(tc.tensor([0]*n_pad + batch[i]['answer_mask'])) # 0
        input_ids = tc.vstack(input_ids)
        attention_mask = tc.vstack(attention_mask)
        answer_mask = tc.vstack(answer_mask)


        input_ids = input_ids[:, -tokenizer.max_length:]
        attention_mask = attention_mask[:, -tokenizer.max_length:]
        answer_mask = answer_mask[:, -tokenizer.max_length:]

        label = answer_mask

        # return
        batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'answer_mask': answer_mask}
        label = answer_mask
        
        return batch, label  #(x, y) format
    

    # dataset stats
    # if training_args.n_train_max:
    #     tokenized_datasets['train'] = tokenized_datasets['train'].shuffle(seed=training_args.seed).select(
    #         range(min(len(tokenized_datasets['train']), training_args.n_train_max))
    #     )
    #     #raw too
    #     raw_datasets['train'] = raw_datasets['train'].shuffle(seed=training_args.seed).select(
    #         range(min(len(raw_datasets['train']), training_args.n_train_max))
    #     )
        
    # TODO do not split?
    tokenized_datasets['val1'] = tokenized_datasets['train']
    # train_split = tokenized_datasets['train'].train_test_split(test_size=0.5, shuffle=False)
    # tokenized_datasets['train'] = train_split['train']
    # tokenized_datasets['val1'] = train_split['test']


    tokenized_datasets['val2'] = tokenized_datasets['validation']

    # raw_datasets['test'] = raw_datasets['test'].select(range(min(len(raw_datasets['test']), training_args.n_test_max))) # Same reason as in n_cal.
    # tokenized_datasets['test'] = tokenized_datasets['test'].select(range(min(len(tokenized_datasets['test']), training_args.n_test_max)))

    tokenized_datasets['val1+2'] = concatenate_datasets([tokenized_datasets['val1'], tokenized_datasets['val2']])
    
    
    print(
        f'# train examples = {len(tokenized_datasets["train"])}, '
        #f'# validation examples = {len(tokenized_datasets["validation"])}, '
        f'# val1 examples = {len(tokenized_datasets["val1"])}, '
        f'# val2 examples = {len(tokenized_datasets["val2"])}, '
        f'# val1+2 examples = {len(tokenized_datasets["val1+2"])}, '
        f'# test examples = {len(tokenized_datasets["test"])}'
    )

    dataloaders = { 
        'train': DataLoader(tokenized_datasets['train'],
                            #collate_fn=transformers.DataCollatorWithPadding(tokenizer=tokenizer),
                            collate_fn=collate_fn,
                            batch_size=training_args.per_device_train_batch_size,
                            shuffle=True,
                            num_workers=training_args.dataloader_num_workers),
        'val1': DataLoader(tokenized_datasets['val1'],
                           collate_fn=collate_fn,
                           batch_size=training_args.per_device_train_batch_size,
                           shuffle=False,
                           num_workers=training_args.dataloader_num_workers),
        'val2': DataLoader(tokenized_datasets['val2'],
                           collate_fn=collate_fn,
                           batch_size=training_args.per_device_train_batch_size,
                           shuffle=False,
                           num_workers=training_args.dataloader_num_workers),
        'val1+2': DataLoader(tokenized_datasets['val1+2'],
                           collate_fn=collate_fn,
                           batch_size=training_args.per_device_train_batch_size,
                           shuffle=False,
                           num_workers=training_args.dataloader_num_workers),        
        'test': DataLoader(tokenized_datasets['test'],
                           #collate_fn=transformers.DataCollatorWithPadding(tokenizer=tokenizer),
                           collate_fn=collate_fn,
                           batch_size=training_args.per_device_eval_batch_size,
                           shuffle=False,
                           num_workers=training_args.dataloader_num_workers),
    }
    
    return tokenizer, raw_datasets, tokenized_datasets, dataloaders


def init_model_qna(model_args, tokenizer):

    
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise NotImplementedError

    assert(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        device_map='auto',
    )
    model.resize_token_embeddings(len(tokenizer))

    n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    print(f"Model total size={n_params/2**20:.2f}M params")

    # init a wrapper model
    wrapper_model = models.QnALLMWrapper(model, tokenizer)
    
    return config, model, wrapper_model


                   

#==================================================
# training and evaluation code
#==================================================
def main():


    # # read training_args and model_args first
    # parser = HfArgumentParser((ModelArguments, UncertaintyTrainingArguments))
    # model_args, training_args = parser.parse_args_into_dataclasses()
    # if training_args.gen != 'gen_interactive':
    #     parser = HfArgumentParser((DataTrainingArguments))
    #     data_args = parser.parse_args_into_dataclasses()
        
    
    # read args
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UncertaintyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # setup logger
    os.makedirs(os.path.join(training_args.snapshot_root, training_args.exp_name), exist_ok=True)
    sys.stdout = util.Logger(os.path.join(training_args.snapshot_root, training_args.exp_name, 'out'))    
    sys.stderr = util.Logger(os.path.join(training_args.snapshot_root, training_args.exp_name, 'out_err'))

    print('==================================================')
    print(model_args)
    print(training_args)
    if 'data_args' in locals():
        print(data_args)
    
    
    logger = logging.getLogger(__name__)

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    

    # init datasets
    tokenizer, raw_datasets, tokenized_datasets, dataloaders = init_datasets_qna(data_args, model_args, training_args)
    # init a base model
    if tokenizer is None:
        base_model_config, base_model, wrapper_base_model = None, None, None        
    else:
        base_model_config, base_model, wrapper_base_model = init_model_qna(model_args, tokenizer)

    
    if training_args.method == 'GreedyGen-SG':
        # Our algorithm
        
        G = models.ConformalLG(
            base_model=wrapper_base_model,
            generation_type=training_args.gen_generation_type,
            gen_len=training_args.gen_len,
        )

        EG = models.EntailmentSet(
            model_args=model_args,
            training_args=training_args,
            data_args=data_args,
            raw_datasets=raw_datasets,
            #entail_model=wrapper_nli_model,
            #generation_type=training_args.gen_generation_type,
            #gen_len=training_args.gen_len,
        ) 

        # precision CP
        SG = models.PrecisionSG(generator=G)
        l = uncertainty.SGLearner(
            SG,
            EG,
            params=training_args,
            name_postfix='ncgprec'
        )
        l.train(
            dataloaders['val1'] if dataloaders is not None else None, # deprecated
            dataloaders['val2'] if dataloaders is not None else None, # deprecated
            updated_params=types.SimpleNamespace(n=len(raw_datasets['val1']), n_e=len(raw_datasets['val2']))
        )
    elif training_args.method == 'GreedyGen-SGPlot':
        
        # Our algorithm
        
        G = models.ConformalLG(
            base_model=wrapper_base_model,
            generation_type=training_args.gen_generation_type,
            gen_len=training_args.gen_len,
        )

        EG = models.EntailmentSet(
            model_args=model_args,
            training_args=training_args,
            data_args=data_args,
            raw_datasets=raw_datasets,
            #entail_model=wrapper_nli_model,
            #generation_type=training_args.gen_generation_type,
            #gen_len=training_args.gen_len,
        ) 

        # precision CP
        SG = models.PrecisionSG(generator=G)
        l = uncertainty.SGLearner(
            SG,
            EG,
            params=training_args,
            name_postfix='ncgprec'
        )
        l.plot(
            dataloaders['val1'] if dataloaders is not None else None, # deprecated
            dataloaders['val2'] if dataloaders is not None else None, # deprecated
            updated_params=types.SimpleNamespace(n=len(raw_datasets['val1']), n_e=len(raw_datasets['val2']))
        )
    elif training_args.method == 'GreedyGen-SGQuanPlot':
        
        # Our algorithm
        
        G = models.ConformalLG(
            base_model=wrapper_base_model,
            generation_type=training_args.gen_generation_type,
            gen_len=training_args.gen_len,
        )

        EG = models.EntailmentSet(
            model_args=model_args,
            training_args=training_args,
            data_args=data_args,
            raw_datasets=raw_datasets,
            #entail_model=wrapper_nli_model,
            #generation_type=training_args.gen_generation_type,
            #gen_len=training_args.gen_len,
        ) 

        # precision CP
        SG = models.PrecisionSG(generator=G)
        l = uncertainty.SGLearner(
            SG,
            EG,
            params=training_args,
            name_postfix='ncgprec'
        )
        l.quan_plot(
            dataloaders['val1'] if dataloaders is not None else None, # deprecated
            dataloaders['val2'] if dataloaders is not None else None, # deprecated
            updated_params=types.SimpleNamespace(n=len(raw_datasets['val1']), n_e=len(raw_datasets['val2']))
        )

    else:
        raise NotImplementedError
    

    
if __name__ == "__main__":
    t_start = time.time()
    main()
    print(f'[total running time] {time.time() - t_start:.2f} sec')
