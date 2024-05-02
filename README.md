# Language Modeling: TP Transformer vs Standard Transformer

This repository contains code and experiments comparing the performance of TP Transformer and Standard Transformer models on language modeling tasks.

## Setup

To set up the environment and install the dependencies, follow these steps:

1. Create and activate a new conda environment:

```
conda create --name myenv python=3.9
```
```
conda activate myenv
```

2. Install the required packages:

```
pip install -U pip setuptools wheel
pip install torch==2.1.0
pip install transformers
pip install higher
pip install matplotlib
pip install chardet
pip install torchtext
```

## Dataset
We will use the dataset from [BabyLM Challenge](https://babylm.github.io/). The preprocessed data can be found at [Huggingface](https://huggingface.co/datasets/vesteinn/babylm). 

Place the `train.txt`, `valid.txt` and `test.txt` in the babylm_preprocessed_10mtrain/ directory. 

To generate smaller datasets from the original dataset, run the `generate.py` after adjusting the target directory and word count. 

## Running the Experiments

There are 2 architectures - TP Transformer and standard Transformer, which are defined in `models.py`. 

Training is done with `lm_train.py`. To see all the training options, run `python lm_train.py -h`. 

This is an example of Slurm script for training the TP transformer. 
```
#!/bin/bash

#BATCH --job-name=TP_2m
#SBATCH --output=./logs/TP_2m.out
#SBATCH --error=./logs/TP_2m.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yining.wang@yale.edu

module load miniconda
conda activate myenv

python3 lm_train.py --batch_size 32 --directory babylm_2m/ --shuffle --architecture TP --n_embd 200 --n_positions 128 --n_head 8 --n_layer 6 --n_epochs 8 --eval_every 500 --learning_rate 0.001 --lr_scheduler_type cosine --warmup_proportion 0.05 --model_name TP_2m
```
This is an example of Slurm script for training the standard transformer. 

```
#!/bin/bash

#BATCH --job-name=STD_2m
#SBATCH --output=./logs/STD_2m.out
#SBATCH --error=./logs/STD_2m.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yining.wang@yale.edu

module load miniconda
conda activate myenv

python3 lm_train.py --batch_size 32 --directory babylm_2m/ --shuffle --architecture STD --n_embd 200 --n_positions 128 --n_head 8 --n_layer 6 --n_epochs 8 --eval_every 500 --learning_rate 0.001 --lr_scheduler_type cosine --warmup_proportion 0.05 --model_name STD_2m
```

