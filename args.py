import os, sys
from enum import Enum
from dataclasses import asdict, dataclass, field, fields
import json
from typing import Optional
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    TrainingArguments,
)

import torch as tc


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

class Arguments:
    def __str__(self):
        self_as_dict = asdict(self)

        # Remove deprecated arguments. That code should be removed once
        # those deprecated arguments are removed from TrainingArguments. (TODO: v5)
        # del self_as_dict["per_gpu_train_batch_size"]
        # del self_as_dict["per_gpu_eval_batch_size"]

        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"    {k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__


@dataclass
class ModelArguments(Arguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    entail_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

    # def __str__(self):
    #     self_as_dict = asdict(self)

    #     # Remove deprecated arguments. That code should be removed once
    #     # those deprecated arguments are removed from TrainingArguments. (TODO: v5)
    #     # del self_as_dict["per_gpu_train_batch_size"]
    #     # del self_as_dict["per_gpu_eval_batch_size"]

    #     self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

    #     attrs_as_str = [f"    {k}={v},\n" for k, v in sorted(self_as_dict.items())]
    #     return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    # __repr__ = __str__

    
    # def to_dict(self):
    #     """
    #     Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
    #     the token values by removing their value.
    #     """
    #     # filter out fields that are defined as field(init=False)
    #     d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

    #     for k, v in d.items():
    #         if isinstance(v, Enum):
    #             d[k] = v.value
    #         if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
    #             d[k] = [x.value for x in v]
    #         if k.endswith("_token"):
    #             d[k] = f"<{k.upper()}>"
    #     return d

    
    # def to_json_string(self):
    #     """
    #     Serializes this instance to a JSON string.
    #     """
    #     return json.dumps(self.to_dict(), indent=2)


@dataclass
class DataTrainingArguments(Arguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

                
@dataclass
class UncertaintyTrainingArguments(Arguments, TrainingArguments):
    distributed_state: Optional[str] = field(default=None)

    dbg: Optional[bool] = field(default=False)

    tag: Optional[str] = field(default='')

    
    method: Optional[str] = field(default='Baseline', metadata={"help": "uncertainty learning"})
    snapshot_root: Optional[str] = field(default='snapshots')
    prompt_model_path: Optional[str] = field(default=None)
    
    cache_root: Optional[str] = field(default='snapshots/cache')
    cache_cal_fn: Optional[str] = field(default=None)
    cache_eval_fn: Optional[str] = field(default=None)
    cache_ent_fn: Optional[str] = field(default=None)
    cache_ent_eval_fn: Optional[str] = field(default=None)
    
    exp_name: Optional[str] = field(default='DBG')
    
    rerun: Optional[bool] = field(default=False)
    resume: Optional[bool] = field(default=False)
    load_final: Optional[bool] = field(default=True)
    verbose: Optional[bool] = field(default=True)
    device: Optional[str] = field(default='cuda')
    
    # n_cal: Optional[int] = field(default=None) #1_000_000
    eps: Optional[float] = field(default=0.1)
    eps_e: Optional[float] = field(default=0.1)
    delta: Optional[float] = field(default=1e-5)
    delta_p: Optional[float] = field(default=1e-5)
    # n_train_max: Optional[int] = field(default=20_000) #100_000
    # n_test_max: Optional[int] = field(default=1_000) #10_000
    # topk: Optional[int] = field(default=50)
    exp_method: Optional[str] = field(default='SSL')
    entail_model: Optional[str] = field(default=None)

    # prompt learning
    # prompt_learning: Optional[str] = field(default='prompt_tuning')
    # n_virtual_tokens: Optional[int] = field(default=50)
    
    # histogram binning
    # n_bins: Optional[int] = field(default=20)

    # optimizer: Optional[str] = field(default='Adam') ##TODO
    # n_epochs: Optional[int] = field(default=1)
    # lr: Optional[float] = field(default=1e-4)
    # #momentum: Optional[float] = field(default=0.0)
    # weight_decay: Optional[float] = field(default=0.0)
    # lr_decay_epoch: Optional[int] = field(default=1)
    # lr_decay_rate: Optional[float] = field(default=0.5)
    # lr_gamma: Optional[float] = field(default=0.99)
    # n_hidden_neurons: Optional[int] = field(default=4000)
    # dropout_prob: Optional[float] = field(default=0.5)
    # freeze_bias: Optional[bool] = field(default=False)
    # use_logsigmoid: Optional[bool] = field(default=False)
    # use_logspace: Optional[bool] = field(default=True)
    
    
    # tau_step: Optional[float] = field(default=1e-16) 
    # tau_end: Optional[float] = field(default=1.0) # 1.0: assume classification
    # eps_tol: Optional[float] = field(default=1.25)
    # tau_tol: Optional[float] = field(default=1e-16)
    
    num_workers: Optional[int] = field(default=64)

    # #TODO: create gen args
    # gen: Optional[str] = field(default='learn_and_gen')
    # gen_keywords: Optional[str] = field(default='')
    gen_generation_type: Optional[str] = field(default='greedy')
    gen_len: Optional[int] = field(default=50)
    # gen_samples: Optional[int] = field(default=0)

    z_u: Optional[int] = field(default=30000)
    z_e: Optional[int] = field(default=10000)
    md_name: Optional[str] = field(default=None)
    fer: Optional[bool] = field(default=False)
    K: Optional[int] = field(default=5)
    pl: Optional[float] = field(default=0.9)

    seed: Optional[int] = field(default=42)
    model: Optional[str] = field(default=None)
    def __post_init__(self):
        if self.device == 'cpu':
            self.device = tc.device('cpu')
        elif self.device == 'cuda':
            self.device = tc.device('cuda')
        else:
            raise NotImplementedError
