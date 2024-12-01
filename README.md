# selective-generation

This repository contains the PyTorch implementation, which outputs evaluation results for the Neurips 2024 paper "[Selective Generation for Controllable Language Models](https://arxiv.org/abs/2307.09254)".

## Environment Setup

If you use **conda**,
```
conda create -n sg python=3.8
conda activate sg
pip install -r requirements.txt
```

Additional dependencies (e.g., PyTorch) may need to be installed depending on your system setup and hardware.

## Models

If you are only going to use the models and datasets as provided in the paper, you do not need to load the models manually, as both log probabilities and entailment scores have been **precomputed and stored in the dataset**.

We used [Alpaca7B](https://crfm.stanford.edu/2023/03/13/alpaca.html) and the [GPT-3.5-Turbo API](https://platform.openai.com/docs/models#gpt-3-5-turbo) as generators, and [DeBERTa-v2-xxlarge](https://github.com/microsoft/DeBERTa), fine-tuned on the MNLI dataset, as the entailment model.

- To use **Alpaca7B**, request access to [LLaMA](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form) and load the Stanford Alpaca weights.
- To use **GPT-3.5**, set up the OpenAI API.

If you wish to use a different entailment model, modify the `EMDLPATH` variable in the shell script accordingly.

This implementation supports greedy generation and obtaining log probabilities from other models on demand. However, if you want to use a different model, labeling must be done manually. To use other APIs, you must manually configure the dataset.
(You can do this by referring to `/generation/`.)

## Data

In this paper, the [Natural Questions (NQ)](https://github.com/google-research-datasets/natural-questions) dataset and the QA2D dataset, filtered with only SQuAD, are sampled and used.
Since NQ dataset has no transformed answers, we use [Transforming Question Answering Datasets Into Natural Language Inference Datasets](https://github.com/kelvinguu/qanli) to obtain rule-based transformed sequences.
The QA2D dataset, also available via this repository, contains human-annotated answers from Turk.

## How to run

The following commands generate the figures and tables presented in the paper.

To get results (in `/snapshots/`) for a given model and dataset, GPT-3.5 & NQ dataset for example,
```
./run_nq_gpt3.5.sh
```

To draw box plots, GPT-3.5 & NQ dataset for example,
```
# This draws box plots, Figure 4.
./run_nq_gpt3.5_plot.sh
```

To draw bar plots,
```
# This draws bar plots over different numbers of unlabeled samples, Figure 3.
./run_nq_gpt3.5_quan_plot.sh
./run_nq_alpaca7B_quan_plot.sh
```
