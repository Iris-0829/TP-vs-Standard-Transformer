# TP-Transformer
# See https://arxiv.org/abs/1910.06611
# See *github link to readme*
#
#
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lib import HyperParams

LOG_TERMINAL = False  # slow down if enabled

def is_nan_string(x):
  if not LOG_TERMINAL:
    return "skip"
  return "NaN" if torch.isnan(x).any() else "ok"

def is_nan(d):
  if LOG_TERMINAL:
    for t in d:
      if torch.isnan(t).any():
        return True
  return False

def debug(*args):
  if LOG_TERMINAL:
    print(*args)

def build_transformer(params, pad_idx):
  p = HyperParams()
  p.d_vocab = params.input_dim
  p.d_pos = 200  # max input size

  p.d_f = params.filter

  p.n_L = params.n_layers
  p.n_I = params.n_heads

  p.d_x = params.hidden  # token embedding dimension
  p.d_p = params.hidden  # position embedding dimension

  p.d_v = p.d_x // p.n_I  # value dimension
  p.d_r = p.d_x // p.n_I  # role dimension

  p.d_k = p.d_x // p.n_I  # key dimension
  p.d_q = p.d_x // p.n_I  # query dimension

  p.dropout = params.dropout

  embedding = EmbeddingMultilinearSinusoidal(d_vocab=params.input_dim,
                                             d_x=p.d_x,
                                             dropout=params.dropout,
                                             max_length=200)
  encoder = Encoder(p=p)
  decoder = Decoder(p=p)
  model = Seq2Seq(p=p,
                  embedding=embedding,
                  encoder=encoder,
                  decoder=decoder,
                  pad_idx=pad_idx)

  return model


class Encoder(nn.Module):
  def __init__(self, p):
    super().__init__()

    layers = [EncoderLayer(p)]
    for _ in range(p.n_L - 1):
      layers.append(EncoderLayer(p))
    self.layers = nn.ModuleList(layers)

  def forward(self, src, src_mask):
    # src = [batch_size, src_seq_len]
    # src_mask = [batch_size, 1, attn_size]
    debug("\nencoder")
    debug("src: ", src.shape, is_nan_string(src))
    debug("src_mask: ", src_mask.shape, is_nan_string(src_mask))

    for layer in self.layers:
      src = layer(src, src_mask)

    return src


class EncoderLayer(nn.Module):
  def __init__(self, p):
    super().__init__()
    # d_in list of input dimension of z e.g. [p.d_x, p.d_p] or [p.d_v, p.d_r]
    d_h = p.d_x

    # sublayer 1
    self.layernorm1 = nn.LayerNorm(d_h)
    self.MHA = SelfAttention(p)
    self.dropout1 = nn.Dropout(p.dropout)
    # sublayer 2
    self.layernorm2 = nn.LayerNorm(d_h)
    self.densefilter = PositionwiseFeedforward(hid_dim=d_h,
                                               pf_dim=p.d_f,
                                               dropout=p.dropout)
    self.dropout2 = nn.Dropout(p.dropout)
    # output
    self.layernorm3 = nn.LayerNorm(d_h)

  def forward(self, src, src_mask):
    # src = [batch_size, src_seq_size, hid_dim]
    # src_mask = [batch_size, src_seq_size]

    # sublayer 1
    z = self.layernorm1(src)
    z = self.MHA(z, z, z, src_mask)
    z = self.dropout1(z)
    src = src + z

    # sublayer 2
    z = self.layernorm2(src)
    z = self.densefilter(z)
    z = self.dropout2(z)
    src = src + z

    return self.layernorm3(src)


class EmbeddingMultilinearSinusoidal(nn.Module):
  def __init__(self, d_vocab, d_x, dropout, max_length):
    super(EmbeddingMultilinearSinusoidal, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.max_length = max_length
    self.d_x = d_x

    # token encodings
    self.tok_embedding = nn.Embedding(d_vocab, d_x)
    self.scale = torch.sqrt(torch.FloatTensor([d_x]))

    # sinusoidal encoding
    pe = torch.zeros(max_length, d_x)
    position = torch.arange(0., max_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., d_x, 2) *
                         -(math.log(10000.0) / d_x))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)
    # pe = [1, seq_len, d_p]

    # x -> r
    self.linear = nn.Linear(d_x, d_x)
    self.mul_scale = torch.FloatTensor([1. / math.sqrt(math.sqrt(2) - 1)])

    self.reset_parameters()

  def forward(self, src):
    # src = [batch_size, src_seq_len]
    tok_emb = self.tok_embedding(src) * self.scale.to(src.device)

    # sinusoidal pos embedding
    pos_sin_emb = torch.autograd.Variable(self.pe[:, :src.size(1)],
                                          requires_grad=False)
    x = tok_emb + pos_sin_emb
    # x = [batch_size, src_seq_len, d_x]

    r = self.linear(x) + 1  # such that initially ~N(1,1)
    # r = [batch_size, src_seq_len, d_r]

    z = x * r
    z = self.dropout(z)
    # src = [batch_size, src_seq_len, d_x*d_r]
    return z

  def transpose_forward(self, trg):
    # trg = [batch_size, trg_seq_len, d_v, d_r]

    logits = torch.matmul(trg, torch.transpose(self.tok_embedding.weight, 0, 1))
    # logits = [batch_size, trg_seq_len, d_vocab
    return logits

  def reset_parameters(self):
    nn.init.normal_(self.tok_embedding.weight,
                    mean=0,
                    std=1./math.sqrt(self.d_x))
    nn.init.normal_(self.linear.weight,
                    mean=0,
                    std=1./math.sqrt(self.d_x))


class SelfAttention(nn.Module):
  def __init__(self, p):
    super().__init__()
    self.p = p
    self.d_h = p.d_x
    self.n_I = p.n_I

    self.W_q = nn.Linear(self.d_h, p.d_q * p.n_I)
    self.W_k = nn.Linear(self.d_h, p.d_k * p.n_I)
    self.W_v = nn.Linear(self.d_h, p.d_v * p.n_I)
    self.W_r = nn.Linear(self.d_h, p.d_r * p.n_I)

    self.W_o = nn.Linear(p.d_v * p.n_I, p.d_x)

    self.dropout = nn.Dropout(p.dropout)
    self.dot_scale = torch.FloatTensor([math.sqrt(p.d_k)])
    self.mul_scale = torch.FloatTensor([1./math.sqrt(math.sqrt(2) - 1)])

  def forward(self, query, key, value, mask=None):
    # query = key = value = [batch_size, seq_len, hid_dim]
    # src_mask = [batch_size, 1, 1, pad_seq]
    # trg_mask = [batch_size, 1, pad_seq, past_seq]

    bsz = query.shape[0]

    Q = self.W_q(query)
    K = self.W_k(key)
    V = self.W_v(value)
    R = self.W_r(query)
    # Q, K, V, R = [batch_size, seq_len, n_heads * d_*]

    Q = Q.view(bsz, -1, self.n_I, self.p.d_q).permute(0,2,1,3)
    K = K.view(bsz, -1, self.n_I, self.p.d_k).permute(0,2,1,3)
    V = V.view(bsz, -1, self.n_I, self.p.d_v).permute(0,2,1,3)
    R = R.view(bsz, -1, self.n_I, self.p.d_r).permute(0,2,1,3)
    # Q, K, V, R = [batch_size, n_heads, seq_size, d_*]

    dot = torch.einsum('bhid,bhjd->bhij', Q, K) / self.dot_scale.to(key.device)
    # energy   = [batch_size, n_heads, query_pos     , key_pos]
    # src_mask = [batch_size, 1      , 1             , attn]
    # trg_mask = [batch_size, 1      , query_specific, attn]

    if mask is not None:
      dot = dot.masked_fill(mask == 0, -1e10)

    attention = self.dropout(F.softmax(dot, dim=-1))
    # attention = [batch_size, n_heads, seq_size, seq_size]

    v_bar = torch.einsum('bhjd,bhij->bhid', V, attention)
    # v_bar = [batch_size, n_heads, seq_size, d_v]

    # bind
    new_v = v_bar * R
    new_v = new_v.permute(0,2,1,3).contiguous()
    # v_bar = [batch_size, seq_size, n_heads, d_v]

    new_v = new_v.view(bsz, -1, self.n_I * self.p.d_v)
    # new_v = [batch_size, src_seq_size, n_heads * d_v]

    x = self.W_o(new_v)
    # x = [batch_size, seq_size, d_x]

    return x

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.W_q.weight)
    nn.init.xavier_uniform_(self.W_k.weight)
    nn.init.xavier_uniform_(self.W_v.weight)
    nn.init.xavier_uniform_(self.W_o.weight)

    nn.init.normal_(self.W_r.weight,
                    mean=0,
                    std=1./math.sqrt(self.p.d_r))


class PositionwiseFeedforward(nn.Module):
  def __init__(self, hid_dim, pf_dim, dropout):
    super().__init__()

    self.hid_dim = hid_dim
    self.pf_dim = pf_dim

    self.linear1 = nn.Linear(hid_dim, pf_dim)
    self.linear2 = nn.Linear(pf_dim, hid_dim)
    self.dropout = nn.Dropout(dropout)

    self.reset_parameters()

  def forward(self, x):
    # x = [batch_size, seq_size, hid_dim]

    x = self.linear1(x)
    x = self.dropout(F.relu(x))
    x = self.linear2(x)

    # x = [batch_size, seq_size, hid_dim]
    return x

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.linear1.weight)
    nn.init.xavier_uniform_(self.linear2.weight)


class Decoder(nn.Module):
  def __init__(self, p):
    super().__init__()

    self.layers = nn.ModuleList([DecoderLayer(p) for _ in range(p.n_L)])

  def forward(self, trg, trg_mask):
    # trg = [batch_size, trg_seq_size, hid_dim]
    # src = [batch_size, src_seq_size, hid_dim]
    # trg_mask = [batch_size, trg_seq_size]
    # src_mask = [batch_size, src_seq_size]

    for layer in self.layers:
      trg = layer(trg, trg_mask)

    return trg


class DecoderLayer(nn.Module):
  def __init__(self, p):
    super().__init__()
    d_h = p.d_x
    # sublayer 1
    self.layernorm1 = nn.LayerNorm(d_h)
    self.selfAttn = SelfAttention(p)
    self.dropout1 = nn.Dropout(p.dropout)
    # sublayer 2
    self.layernorm2 = nn.LayerNorm(d_h)
    self.encAttn = SelfAttention(p)
    self.dropout2 = nn.Dropout(p.dropout)
    # sublayer 3
    self.layernorm3 = nn.LayerNorm(d_h)
    self.densefilter = PositionwiseFeedforward(hid_dim=d_h,
                                               pf_dim=p.d_f,
                                               dropout=p.dropout)
    self.dropout3 = nn.Dropout(p.dropout)

    # output
    self.layernorm4 = nn.LayerNorm(d_h)


  def forward(self, trg, trg_mask):
    # trg = [batch_size, trg_seq_size, hid_dim]
    # src = [batch_size, src_seq_size, hid_dim]
    # trg_mask = [batch_size, trg_seq_size]
    # src_mask = [batch_size, src_seq_size]

    # self attention
    z = self.layernorm1(trg)
    z = self.selfAttn(z, z, z, trg_mask)
    z = self.dropout1(z)
    trg = trg + z

    # encoder attention
    z = self.layernorm2(trg)
    # z = self.encAttn(z, src, src, src_mask)
    z = self.dropout2(z)
    trg = trg + z

    # dense filter
    z = self.layernorm3(trg)
    z = self.densefilter(z)
    z = self.dropout3(z)
    trg = trg + z

    return self.layernorm4(trg)


class Seq2Seq(nn.Module):
  def __init__(self, p, embedding, encoder, decoder, pad_idx):
    super().__init__()

    self.embedding = embedding
    # self.encoder = encoder
    self.decoder = decoder
    self.pad_idx = pad_idx
    self.p = p

  def make_masks(self, trg):
    # src = [batch_size, src_seq_size]
    # trg = [batch_size, trg_seq_size]

    # src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
    # trg_mask = [batch_size, 1, trg_seq_size, 1]
    trg_len = trg.shape[1]

    trg_sub_mask = torch.tril(
      torch.ones((trg_len, trg_len), dtype=torch.uint8, device=trg.device))

    trg_mask = trg_pad_mask.type(torch.ByteTensor) & trg_sub_mask.type(torch.ByteTensor)

    # src_mask = [batch_size, 1, 1, pad_seq]
    # trg_mask = [batch_size, 1, pad_seq, past_seq]
    return trg_mask

  def forward(self, trg):
    # src = [batch_size, src_seq_size]
    # trg = [batch_size, trg_seq_size]

    # src_mask, trg_mask = self.make_masks(src, trg)
    # src_mask = [batch_size, 1, 1, pad_seq]
    # trg_mask = [batch_size, 1, pad_seq, past_seq]

    # src = self.embedding(src)
    # trg = self.embedding(trg)
    # src = [batch_size, src_seq_size, hid_dim]

    # enc_src = self.encoder(src, src_mask)
    # enc_src = [batch_size, src_seq_size, hid_dim]

    # out = self.decoder(trg, enc_src, trg_mask, src_mask)
    # out = [batch_size, trg_seq_size, d_x]

    # logits = self.embedding.transpose_forward(out)
    # logits = [batch_size, trg_seq_size, d_vocab]

    # return logits
    trg_mask = self.make_masks(trg)
    trg = self.embedding(trg)
    out = self.decoder(trg, trg_mask)
    logits = self.embedding.transpose_forward(out)
    return logits

  def make_src_mask(self, src):
    # src = [batch size, src sent len]
    src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask.to(src.device)

  def make_trg_mask(self, trg):
    # trg = [batch size, trg sent len]
    trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)

    trg_len = trg.shape[1]

    trg_sub_mask = torch.tril(
      torch.ones((trg_len, trg_len), dtype=torch.uint8, device=trg.device))

    trg_mask = trg_pad_mask.type(torch.ByteTensor) & trg_sub_mask.type(torch.ByteTensor)

    return trg_mask.to(trg.device)

  def greedy_inference(model, src, sos_idx, eos_idx, max_length, device):
    model.eval()
    src = src.to(device)
    src_mask = model.make_src_mask(src)
    src_emb = model.embedding(src)

    # run encoder
    # enc_src = model.encoder(src_emb, src_mask)
    trg = torch.ones(src.shape[0], 1).fill_(sos_idx).type_as(src).to(device)

    done = torch.zeros(src.shape[0]).type(torch.uint8).to(device)
    for _ in range(max_length):
      trg_emb = model.embedding(trg)
      trg_mask = model.make_trg_mask(trg)
      # run decoder
      output = model.decoder(trg=trg_emb, trg_mask=trg_mask)
      logits = model.embedding.transpose_forward(output)
      pred = torch.argmax(logits[:,[-1],:], dim=-1)
      trg = torch.cat([trg, pred], dim=1)

      eos_match = (pred.squeeze(1) == eos_idx)
      done = done | eos_match

      if done.sum() == src.shape[0]:
        break

    return trg


if __name__ == "__main__":
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
  parser.add_argument("--directory", help="Directory where the datasets are found", type=str,
                      default="CHILDES_final/pretraining/")
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
  parser.add_argument("--lr_scheduler_type", help="Learning rate scheduler type (cosine or constant)", type=str,
                      default="cosine")
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
  # module = DataLoader(module_name=args.module_name,
  #                     train_bs=args.batch_size,
  #                     eval_bs=args.batch_size,
  #                     device=device,
  #                     log=log)
  args.input_dim = 256
  args.PAD = 0
  model = build_transformer(params=args, pad_idx=args.PAD)
  model_size = sum(t.numel() for t in model.parameters())
  print(f"Model size: {model_size / 1000 ** 2:.1f}M parameters")