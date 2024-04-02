

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

parser.add_argument('--log_folder', type=str, default="log", metavar='S',
                    help='Log folder (default: "")')

args = parser.parse_args()
log = setup_logger(args.log_folder)

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
    model = RNNLM(
        rnn_type="LSTM",
        vocab_size=vocab_size,
        emb_size=args.n_embd,
        hidden_size=args.n_embd,
        n_layers=args.n_layer,
        dropout=0.1,
        tie_weights=True
    ).to(device)

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
    # TP transformer
    model = TPDecoder(
        vocab_size=vocab_size,
        hidden_size=args.n_embd,
        num_layers=args.n_layer,
        num_heads=args.n_head,
        max_length=min(args.n_positions-10,100),
        dropout=0.1
    ).to(device)

elif args.architecture == "STD":
    # Standard transformer
    model = TransformerDecoder(
        vocab_size=vocab_size,
        hidden_size=args.n_embd,
        num_layers=args.n_layer,
        num_heads=args.n_head,
        max_length=min(args.n_positions-10,100),
        dropout=0.1
    ).to(device)

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







