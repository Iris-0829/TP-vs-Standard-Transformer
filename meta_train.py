
import math
import os
import logging
from collections import Counter

import torch

from transformers import GPT2LMHeadModel, AutoConfig

from training import *
from dataloading import *
from models import *

import argparse

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


parser = argparse.ArgumentParser()

# Corpus arguments
parser.add_argument("--meta_train_batch_size", help="Sequences per episode of meta training", type=int, default=100)
parser.add_argument("--meta_eval_batch_size", help="Batch size during meta evaluation", type=int, default=None)
parser.add_argument("--meta_test_size", help="Evaluation sequences per episode of meta training", type=int, default=10)
parser.add_argument("--dataset", help="Directory containing the meta-dataset", type=str, default="simple")


# Architecture arguments
parser.add_argument("--architecture", help="Type of architecture. Options: GPT2, LSTM", type=str, default="GPT2")
parser.add_argument("--n_embd", help="Embedding size", type=int, default=768)
parser.add_argument("--n_positions", help="Max context length the model can take", type=int, default=128)
parser.add_argument("--n_head", help="Number of attention heads", type=int, default=12)
parser.add_argument("--n_layer", help="Number of layers", type=int, default=12)

# Training arguments
parser.add_argument("--n_epochs", help="Number of training epochs", type=int, default=10)
parser.add_argument("--eval_every", help="Number of training steps to go between evaluations", type=int, default=100)
parser.add_argument("--weight_decay", help="Weight decay", type=float, default=1e-1)
parser.add_argument("--learning_rate", help="Outer-loop learning rate", type=float, default=5e-4)
parser.add_argument("--inner_lr", help="Inner-loop learning rate", type=float, default=1e-1)
parser.add_argument("--warmup_proportion", help="Proportion of total steps that are warmup", type=float, default=0.05)
parser.add_argument("--multi_step_loss", help="use multi-step loss", action='store_true')

# Saving arguments
parser.add_argument("--model_name", help="Model name prefix", type=str, default=None)
parser.add_argument("--weight_dir", help="Directory to save model weights in", type=str, default="weights/")
parser.add_argument("--log_dir", help="Directory to save logs in", type=str, default="logs/")

# Evaluation arguments
parser.add_argument("--eval_testset", help="evaluate on the test set", action='store_true')



args = parser.parse_args()

if args.meta_eval_batch_size is None:
    args.meta_eval_batch_size = args.meta_train_batch_size

if args.eval_testset:
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, handlers=[logging.StreamHandler(),logging.FileHandler(args.log_dir + args.model_name + "_eval.log")])

else:
    model_name = args.model_name
    model_index = 0
    args.model_name = model_name + "_" + str(model_index)
    while args.model_name + ".log" in os.listdir(args.log_dir):
        model_index += 1
        args.model_name = model_name + "_" + str(model_index)

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, handlers=[logging.StreamHandler(),logging.FileHandler(args.log_dir + args.model_name + ".log")])

logging.info(args)





corpus = MetaCorpus(directory=args.dataset, context_size=args.n_positions, batches_per_buffer=1000, stream=True, shuffle=True)
data_collator = LMDataCollator(corpus.tokenizer, mlm=False, meta=True)

if args.architecture == "GPT2":
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(corpus.tokenizer),
        n_embd=args.n_embd,
        n_ctx=args.n_positions,
        n_layer=args.n_layer,
        n_head=args.n_head,
        bos_token_id=corpus.tokenizer.bos_token_id,
        #eos_token_id=corpus.tokenizer.eos_token_id,
    )

    model = GPT2LMHeadModel(config).to(device)

elif args.architecture == "LSTM":
    model = RNNLM(rnn_type="LSTM", vocab_size=len(corpus.tokenizer)+1, emb_size=args.n_embd, hidden_size=args.n_embd, n_layers=args.n_layer, dropout=0.1, tie_weights=True).to(device)
    model.rnn.flatten_parameters()



model_size = sum(t.numel() for t in model.parameters())
logging.info(f"Parameter count: {model_size/1000**2:.1f}M parameters")


trainer = MetaTrainer(
        model=model,
        train_dataset=corpus.train,
        eval_dataset=corpus.valid,
        data_collator=data_collator,
        n_epochs=args.n_epochs,
        eval_every=args.eval_every,
        log_every=args.eval_every,
        save_dir=args.weight_dir,
        model_name=args.model_name,
        weight_decay=args.weight_decay,   
        warmup_steps=math.ceil(args.warmup_proportion*args.n_epochs*len(corpus.train)),
        lr_scheduler_type="cosine",
        inner_lr=args.inner_lr,
        learning_rate=args.learning_rate,
        fp16=False,
        train_batch_size=args.meta_train_batch_size,
        eval_batch_size=args.meta_eval_batch_size,
        multi_step_loss=args.multi_step_loss,
        )

if not args.eval_testset: 
    trainer.meta_train()

if not args.model_name.startswith("random"):
    trainer.load_best()

if not args.eval_testset or args.eval_valid:
    trainer.meta_evaluate(eval_dataset=corpus.valid, name="Validation")

if args.eval_testset:
    trainer.meta_evaluate(eval_dataset=corpus.test, name="Test")


