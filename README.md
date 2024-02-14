

This repository contains code for training language models (i.e., next-word prediction models). You can either train a language model with standard training or with meta-training (MAML).

# Setting up a Python venv

1. Load a Python module. On Adroit, this can be done with:
```
module load anaconda3/2022.5
```

2. Create a python venv to run code in:
```
python -m venv .venv
```

3. Activate the venv
```
source .venv/bin/activate
```

4. Install requirements (there might be others you need beyond this).
```
pip install -U pip setuptools wheel
pip install torch
pip install transformers
pip install higher
```

5. Whenever you run code, you will first need to load Python (step 1) and then activate the venv (step 3), and it should automatically load all the packages you've installed. For example, when launching a Slurm job on Adroit, you will need to include those lines of code before the command that launches your script. Here is an example Slurm script:

```
#!/bin/bash
#SBATCH --job-name=expt_adapt/adapt_besthyps_3_1
#SBATCH --time=13:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tm4633@princeton.edu
#SBATCH --output=expt_adapt/adapt_besthyps_3_1.log
#SBATCH --error=expt_adapt/adapt_besthyps_3_1.err

module purge
module load anaconda3/2022.5
source ../mamlized_seq2seq/.venv/bin/activate

python lm_train.py --directory CHILDES/pretraining_full_3/ --batch_size 10 --architecture LSTM --n_embd 1024 --n_positions 256 --n_layer 2 --dropout 0.4 --n_epochs 32 --eval_every 100 --weight_decay 0.01 --learning_rate 0.001 --lr_scheduler_type cosine --warmup_proportion 0.06 --model_name adapt_params0_hidden1024_pretraining_full_nopre --model_index 3 --weight_dir /scratch/gpfs/tm4633/inductive_bias_distillation/weights/ --log_dir /scratch/gpfs/tm4633/inductive_bias_distillation/logs/
```



# General overview

1. Requirements: I don't remember exactly what packages are required; I believe you will need numpy, pytorch, and transformers, which can be installed with the following command, but there might be others I'm missing:
```
pip install numpy
pip install torch
pip install transformers
```

2. Data: Your dataset should be in a directory of its own, containing 3 files called `train.txt`, `valid.txt`, and `test.txt`. The code is currently set up for word-level language modeling, so the files are set up with spaces separating the tokens.
- Standard dataset: Each line in the file is one "sentence." You can see an example in `standard_dataset`, which was generated with `python create_simple_standard.py`.
- Meta dataset: Each line is a mini-dataset of its own (one "language"). The line contains 2 strings separated by tabs. The first string is the training set for that language, and the second string is the test set. (Although this is stored as a single string, the model code will break the string down into chunks for you). You can see an example in `meta_dataset`, which was generated with `python create_simple_meta.py`.

3. Architectures: The code accommodates 2 types of architectures: LSTMs (which are defined in `models.py`) and Transformers (specifically, the GPT-2 architecture, which is imported from HuggingFace - not defined within this repository).

4. Training a standard language model: This is done with the file `lm_train.py`. To see all the training options, run `python lm_train.py -h`. Here is an example of training the GPT-2 architecture on our simple standard dataset (you may need to first create directories called `logs/` and `weights/`):
```
python lm_train.py --batch_size 10 --directory standard_dataset/ --shuffle --architecture GPT2 --n_embd 12 --n_positions 50 --n_head 4 --n_layer 2 --n_epochs 1 --eval_every 10 --learning_rate 5e-4 --lr_scheduler_type cosine --warmup_proportion 0.05 --model_name standard_lm_gpt2
```

5. Training a meta language model: This is done with the file `meta_train.py`. To see all the training options, run `python meta_train.py -h`. Here is an example of meta-training the GPT-2 architecture on our simple meta dataset (you may need to first create directories called `logs/` and `weights/`):
```
python meta_train.py --meta_train_batch_size 10 --dataset meta_dataset/ --architecture GPT2 --n_embd 12 --n_positions 50 --n_head 4 --n_layer 2 --n_epochs 2 --eval_every 10 --learning_rate 0.0005 --inner_lr 0.005 --warmup_proportion 0.05 --model_name meta_lm_gpt2
```

6. Note that it is possible to first meta-train a language model with `meta_train.py` and to then further train it (using standard training) on one specific dataset using `lm_train.py`.

7. I am happy to be consulted about what would be reasonable settings for all the parameters.


