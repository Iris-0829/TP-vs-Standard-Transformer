#!/bin/bash

# Run training with GPT2
echo "Training with GPT2..."
python3 lm_train.py --batch_size 10 --directory standard_dataset/ --shuffle --architecture GPT2 --n_embd 12 --n_positions 50 --n_head 4 --n_layer 2 --n_epochs 1 --eval_every 10 --learning_rate 5e-4 --lr_scheduler_type cosine --warmup_proportion 0.05 --model_name standard_lm_gpt2_0

# Evaluate GPT2 and save results
echo "Evaluating GPT2 and saving results..."
python3 lm_train.py --eval --batch_size 10 --directory standard_dataset/ --shuffle --architecture GPT2 --n_embd 12 --n_positions 50 --n_head 4 --n_layer 2 --n_epochs 1 --eval_every 10 --learning_rate 5e-4 --lr_scheduler_type cosine --warmup_proportion 0.05 --model_name standard_lm_gpt2_0 > eval_results_gpt2.txt 2>&1

# Run training with BERT
echo "Training with BERT..."
python3 lm_train.py --batch_size 10 --directory standard_dataset/ --shuffle --architecture BERT --n_embd 12 --n_positions 50 --n_head 4 --n_layer 2 --n_epochs 1 --eval_every 10 --learning_rate 5e-4 --lr_scheduler_type cosine --warmup_proportion 0.05 --model_name standard_lm_bert_0

# Evaluate BERT and save results
echo "Evaluating BERT and saving results..."
python3 lm_train.py --eval --batch_size 10 --directory standard_dataset/ --shuffle --architecture BERT --n_embd 12 --n_positions 50 --n_head 4 --n_layer 2 --n_epochs 1 --eval_every 10 --learning_rate 5e-4 --lr_scheduler_type cosine --warmup_proportion 0.05 --model_name standard_lm_bert_0 > eval_results_bert.txt 2>&1
