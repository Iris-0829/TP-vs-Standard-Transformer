

import math
import os
import logging

import torch

from transformers import GPT2LMHeadModel, TransfoXLLMHeadModel, AutoConfig, BertModel, BertConfig, BertForMaskedLM

from training import *
from dataloading import *
from models import *

import argparse
import importlib
from utils.data_loader import *
from utils.lib import setup_logger

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


parser = argparse.ArgumentParser()

# Corpus arguments
parser.add_argument("--batch_size", help="Sequences per batch", type=int, default=128)
parser.add_argument("--directory", help="Directory where the datasets are found", type=str, default="CHILDES_final/pretraining/")
parser.add_argument("--add_eos", help="Add EOS at the end of each line", action='store_true')
parser.add_argument("--shuffle", help="Shuffle batches within buffer", action='store_true')

# Architecture arguments
parser.add_argument("--architecture", help="Type of architecture. Options: GPT2, LSTM", type=str, default="GPT2")
parser.add_argument("--n_embd", help="Embedding size", type=int, default=768)
parser.add_argument("--n_positions", help="Max context length the model can take", type=int, default=128)
parser.add_argument("--n_head", help="Number of attention heads", type=int, default=12)
parser.add_argument("--n_layer", help="Number of layers", type=int, default=12)
parser.add_argument("--pretrained_name", help="Name for a pretrained model to load", type=str, default=None)
parser.add_argument("--pretrained_vocab_size", help="Vocab size in pretrained model", type=int, default=None)

# Training arguments
parser.add_argument("--n_epochs", help="Number of training epochs", type=int, default=10)
parser.add_argument("--eval_every", help="Number of training steps to go between evaluations", type=int, default=100)
parser.add_argument("--weight_decay", help="Weight decay", type=float, default=1e-1)
parser.add_argument("--learning_rate", help="Learning rate", type=float, default=5e-4)
parser.add_argument("--lr_scheduler_type", help="Learning rate scheduler type (cosine or constant)", type=str, default="cosine")
parser.add_argument("--warmup_proportion", help="Proportion of total steps that are warmup", type=float, default=0.05)
parser.add_argument("--warmup_steps", help="Number of steps to warm up for", type=int, default=None)

# Saving arguments
parser.add_argument("--model_name", help="Model name prefix", type=str, default=None)
parser.add_argument("--weight_dir", help="Directory to save model weights in", type=str, default="weights/")
parser.add_argument("--log_dir", help="Directory to save logs in", type=str, default="logs/")

parser.add_argument("--eval", help="Just evaluating, no training", action='store_true')


# TP
parser.add_argument('--seed', type=int,
                    default=0xBADB1A5, metavar='SEED',
                    help='random seed (default: 0xBADB1A5)')
parser.add_argument('--module_name', type=str,
                    default="numbers__place_value", metavar='NAME',
                    help='module name (default: numbers__place_value)')
parser.add_argument('--load_model', type=str, default="", metavar='S',
                    help='Model to load (default: "")')
parser.add_argument('--eval_mode', action='store_true',
                    help="Don't write logs. (Default: False)")
parser.add_argument('--n_steps', type=int,
                    default=10000, metavar='N',
                    help='maximum number of steps to train (default: 10000)')
parser.add_argument('--max_strikes', type=int,
                    default=1000, metavar='N',
                    help='number of steps without eval loss improvement '
                         'before exiting (default: 1000)')
parser.add_argument('--log_every', type=int,
                    default=50, metavar='N',
                    help='after how many steps to log to terminal '
                         'and tensorboard (default: 50)')
parser.add_argument('--full_loader', action='store_true',
                    help="Use full data loader instead of JIT loader "
                         "(default: False)")
parser.add_argument('--force_remove', action='store_true',
                    help="Removes pre-existing log folders (default: False)")
parser.add_argument('--force_reload', action='store_true',
                    help="Load previous model if available. (Default: False)")
parser.add_argument('--no_train', action='store_true',
                    help="Don't start training. (Default: False)")
parser.add_argument('--log_folder', type=str, default="log", metavar='S',
                    help='Log folder (default: "")')
parser.add_argument('-s', '--log_suffix', type=str,
                    default="", metavar='S',
                    help='Additional log suffix (default: "")')
# optimizer parameters
parser.add_argument('-opt', '--optimizer', type=str,
                    default="Adam", metavar='S',
                    help='the sgd optimizer (default: "Adam")')
parser.add_argument('--beta1', type=float,
                    default=0.9, metavar='F',
                    help='adam beta1 (default: 0.9)')
parser.add_argument('--beta2', type=float,
                    default=0.995, metavar='F',
                    help='adam beta2 (default: 0.995)')
parser.add_argument('-bs', type=int,
                    default=256, metavar='N',
                    help='batch size for train and test (default: 256)')
parser.add_argument('--max_abs_grad_norm', type=float,
                    default=0.1, metavar='F',
                    help='max absolute gradient norm clip (default: 0.1)')
parser.add_argument('--grad_accum_steps', type=int,
                    default=1, metavar='N',
                    help='gradient accumulation steps (default: 1)')
# model parameters
parser.add_argument('--dropout', type=float,
                    default=0.0, metavar='PROB',
                    help='dropout (default: 0.0)')
parser.add_argument('--hidden', type=int,
                    default=512, metavar='N',
                    help='hidden size (default: 512)')
parser.add_argument('-l', '--n_layers', type=int,
                    default=6, metavar='N',
                    help='number of transformer layers (default: 6)')
parser.add_argument('-nh', '--n_heads', type=int,
                    default=8, metavar='N',
                    help='number of attention heads (default: 8)')
parser.add_argument('-f', '--filter', type=int,
                    default=2048, metavar='N',
                    help='filter size (default: 2048)')
parser.add_argument('-d_r', type=int,
                    default=0, metavar='N',
                    help='role size (default: 0)')

args = parser.parse_args()
log = setup_logger(args.log_folder)
module = DataLoader(module_name=args.module_name,
                    train_bs=args.batch_size,
                    eval_bs=args.batch_size,
                    device=device,
                    log=log)

args.PAD = module.source.vocab.stoi['<pad>']

if not args.eval:
    model_name = args.model_name
    model_index = 0
    args.model_name = model_name + "_" + str(model_index)
    while args.model_name + ".log" in os.listdir(args.log_dir):
        model_index += 1
        args.model_name = model_name + "_" + str(model_index)


if args.eval:
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, handlers=[logging.StreamHandler(),logging.FileHandler(args.log_dir + args.model_name + "_eval.log")])
else:
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, handlers=[logging.StreamHandler(),logging.FileHandler(args.log_dir + args.model_name + ".log")])

logging.info(args)

# Load dataset
corpus = Corpus(directory=args.directory, batch_size=args.batch_size, context_size=args.n_positions, batches_per_buffer=10, stream=True, shuffle=args.shuffle, add_eos=args.add_eos) 


logging.info("Vocab size: " + str(len(corpus.tokenizer)+1))

if args.pretrained_vocab_size is not None:
    vocab_size = args.pretrained_vocab_size
else:
    vocab_size = len(corpus.tokenizer)+1

# Create model
if args.architecture == "GPT2":

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=vocab_size, # Add 1 for padding token
        n_embd=args.n_embd,
        n_positions=args.n_positions,
        n_ctx=args.n_positions,
        n_layer=args.n_layer,
        n_head=args.n_head,
    )

    model = GPT2LMHeadModel(config).to(device)

elif args.architecture == "LSTM":
    model = RNNLM(rnn_type="LSTM", vocab_size=vocab_size, emb_size=args.n_embd, hidden_size=args.n_embd, n_layers=args.n_layer, dropout=0.1, tie_weights=True).to(device)

elif args.architecture == "BERT":
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=args.n_embd,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=args.n_embd * 4,  # 4x hidden_size
        max_position_embeddings=args.n_positions,
    )

    model = BertForMaskedLM(config).to(device)

elif args.architecture == "TP":
    imp_module = importlib.import_module("tp-transformer")
    model = imp_module.build_transformer(params=args, pad_idx=args.PAD).to(device)
else:
    logging.info("Architecture not recognized")


model_size = sum(t.numel() for t in model.parameters())
logging.info(f"Model size: {model_size/1000**2:.1f}M parameters")

data_collator = LMDataCollator(corpus.tokenizer, mlm=False)

if args.warmup_steps is None:
    args.warmup_steps = math.ceil(args.warmup_proportion*args.n_epochs*len(corpus.train))

# Create trainer
trainer = Trainer(
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
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        fp16=False,
        )

# Load pretrained weights, if applicable
if args.pretrained_name is not None:
    trainer.load(args.pretrained_name)

# Train model
if not args.eval:
    trainer.train()

# Load best saved weights
trainer.load_best()


# Print some text generated by the model
input_ids = corpus.tokenizer(["the"], add_special_tokens=False, return_tensors="pt").to(device)["input_ids"]

# Pure sampling
text = trainer.model.generate(input_ids, do_sample=True, max_length=min(args.n_positions-10,100), top_p=1.00, top_k=0, early_stopping=True, pad_token_id=corpus.tokenizer.pad_token_id)
text = corpus.tokenizer.decode(text[0], skip_special_tokens=True)
logging.info("Pure sampling:")
logging.info(text)

# Top-40
text = trainer.model.generate(input_ids, do_sample=True, max_length=min(args.n_positions-10,100), top_p=1.00, top_k=40, early_stopping=True, pad_token_id=corpus.tokenizer.pad_token_id)
text = corpus.tokenizer.decode(text[0], skip_special_tokens=True)
logging.info("Top-40:")
logging.info(text)


# Evaluate the model on the validation and/or test set
trainer.evaluate(eval_dataset=corpus.valid, name="Validation")
trainer.evaluate(eval_dataset=corpus.stride_valid, name="Validation (strided)")

if args.eval:
    trainer.evaluate(eval_dataset=corpus.test, name="Test")
    trainer.evaluate(eval_dataset=corpus.stride_test, name="Test (strided)")







