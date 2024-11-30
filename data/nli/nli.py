import os

import datasets
import json
import random

_CITATION = """\
    bib
"""

_DESCRIPTION ="""\
    dataset description
"""

_HOMEPAGE = "URL"

_LICENCE = "Licence for the dataset"


dataset_list =[
    {"name": "nq", "data_dir": 'data/nli/nq', "description": "processed nq dataset with generated answers"},
    {"name": "nq_alpaca7B", "data_dir": 'data/nli/nq_alpaca7B', "description": "processed nq dataset with alpaca7B"},
    {"name": "nq_gpt4", "data_dir": 'data/nli/nq_gpt4', "description": "processed nq dataset with gpt4"},
    {"name": "nq_gpt3.5", "data_dir": 'data/nli/nq_gpt3.5', "description": "processed nq dataset with gpt3.5"},
    {"name": "qa2d_gpt3.5", "data_dir": 'data/nli/qa2d_gpt3.5', "description": "processed qa2d dataset with gpt3.5"},
    {"name": "qa2d_alpaca7B", "data_dir": 'data/nli/qa2d_alpaca7B', "description": "processed qa2d dataset with alpaca7B"},
]
class nliConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)

class nli(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        nliConfig(
            name=dataset["name"],
            data_dir=dataset["data_dir"],
            description=dataset["description"],
        ) for dataset in dataset_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                'question': datasets.Value("string"),
                'answer': datasets.Value("string"),
                'generated_answer': datasets.Value("string"),
                'transformed_answer': datasets.Value("string"),
                'labels': datasets.Value("int8"),
                'logprobs': datasets.Sequence(datasets.Value("float32")),
                'entail_scores': datasets.Sequence(datasets.Value("float32")),
                'samples': datasets.Sequence(datasets.Value("string")),
                'samples_scores': datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
            }
        )
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENCE,
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager):
        seed = 42
        random.seed(seed)

        # load jsonl dataset, nli.jsonl was pre-processed by preprocess_{dataset_name}.py in ncg_llm
        data_jsonl = open(os.path.join(self.config.data_dir, 'nli.jsonl')).read().splitlines()

        #random.shuffle(data_jsonl)

        if self.config.name == 'nq_alpaca7B' or self.config.name == 'nq_gpt3.5' or self.config.name == 'qa2d_gpt3.5' or self.config.name == 'qa2d_alpaca7B':

            data_labeled_jsonl = open(os.path.join(self.config.data_dir, 'nli_labeled.jsonl')).read().splitlines()
            
            random.shuffle(data_jsonl)
            random.shuffle(data_labeled_jsonl)
            train_split = int(len(data_jsonl))
            test_split = int(len(data_labeled_jsonl) * 0.8)

            train_jsonl = data_jsonl[:train_split]
            valid_split = data_labeled_jsonl[:test_split]
            test_jsonl = data_labeled_jsonl[test_split:]

            # MJ: temp
            # is_upsample = False
            # if is_upsample:
            #     valid_split = random.choices(valid_split, k=train_split//2)
            
            # valid_split = random.choices(valid_split, k=5000)

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "nli_data": train_jsonl,
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "nli_data": valid_split,
                        "split": "val",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "nli_data": test_jsonl,
                        "split": "test"
                    },
                ),
            ]
        

        # 8:1:1
        train_split = int(len(data_jsonl)*0.8)
        valid_split = int(len(data_jsonl)*0.9)

        train_jsonl = data_jsonl[:train_split]
        validation_jsonl = data_jsonl[train_split:valid_split]
        test_jsonl = data_jsonl[valid_split:]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "nli_data": train_jsonl,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "nli_data": validation_jsonl,
                    "split": "val",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "nli_data": test_jsonl,
                    "split": "test"
                },
            ),
        ]
    
    # parameters from gen_kwargs.
    # TODO Maybe two methods could be modified flexibly for custom dataset
    def _generate_examples(self, nli_data, split):
        for key, nli in enumerate(nli_data):
            nli_dict = json.loads(nli)

            if self.config.name == 'nq' or self.config.name == 'nq_alpaca7B' or self.config.name == 'nq_gpt3.5' or self.config.name == 'qa2d_gpt3.5' or self.config.name == 'qa2d_alpaca7B':
                question = 'Question:\n' + nli_dict['question'] + '\n\nAnswer:\n'
                value = {
                    'question': question,
                    'answer': nli_dict['answer'],
                    'generated_answer': nli_dict['generated_answer'],
                    'transformed_answer': nli_dict['transformed_answer'],
                    'labels': nli_dict['labels'],
                    'logprobs': nli_dict['logprobs'] if 'logprobs' in nli_dict else None,
                    'entail_scores': nli_dict['entail_scores'],
                    'samples': nli_dict['samples'],
                    'samples_scores': nli_dict['samples_scores'] if 'samples_scores' in nli_dict else None,
                    # 'left_entail_scores': nli_dict['left_entail_scores'] if 'left_entail_scores' in nli_dict else None,
                    # 'left_samples_scores': nli_dict['left_samples_scores'] if 'left_samples_scores' in nli_dict else None,
                }
            else:
                raise NotImplementedError
            
            yield key, value

