# selective-generation

This repository contains the PyTorch implementation, which outputs evaluation results for the Neurips 2024 paper "[Selective Generation for Controllable Language Models](https://arxiv.org/abs/2307.09254)".

## Enviroment Setup

```
# If you use conda
conda create -n sg python=3.8
pip install -r requirements.txt
```

You might install extra dependency to run the model by your hands.

## Models

If you are only going to use the model and dataset in the paper, there is no need to load the models because both logprobs and implication scores are stored in the dataset.

Alpaca7B and GPT3.5-Turbo API were used as generators, and [deberta-v2-xxlarge fintuned on mnli dataset](https://github.com/microsoft/DeBERTa) was used as the entailment model.

To access the Alpaca7B model, simply gain access to [llama1](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form) and load the stanford alpaca weight.
To use GPT-3.5, use openai API.

If you want to change the entailment model, simply change model_name_or_path in the shell file.

Note that this code implementation supports for using greedy generation and logprobs of another model on-demand.
However, if you want to use a different model, labeling must be done manually. (+ To use other APIs, you must manually configure the dataset.)


## Data

NQ dataset and QA2D dataset

## How to run

```
# gpt 3.5, nq dataset example.
./run_nq_gpt3.5.sh
```

```
# This draws box plots.
./run_nq_gpt3.5_plot.sh
```

```
# This draws plots over different numbers of unlabeled samples, Figure 3.
./run_nq_gpt3.5_quan_plot.sh
```
