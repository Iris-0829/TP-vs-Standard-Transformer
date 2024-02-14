
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



class RNNLM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type="LSTM", vocab_size=None, emb_size=None, hidden_size=None, n_layers=None, dropout=0.5, tie_weights=False, pad_token=None):
        super(RNNLM, self).__init__()

        self.pad_token = pad_token

        self.ntoken = vocab_size
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, emb_size)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(emb_size, hidden_size, n_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(emb_size, hidden_size, n_layers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if hidden_size != emb_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = hidden_size
        self.nlayers = n_layers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, max_loss=False, **inputs):

        emb = self.drop(self.encoder(inputs["input_ids"].transpose(0,1)))
        hidden = self.init_hidden(len(inputs["input_ids"]))

        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.transpose(0,1)

        logits = decoded

        loss = None
        if "labels" in inputs:
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            if max_loss:
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).max()
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"logits" : logits, "loss" : loss}


    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


    def generate(self, input_ids, do_sample=True, max_length=500, top_p=1.0, top_k=0, early_stopping=True, pad_token_id=None, eos_token_id=3):
        batch_size = len(input_ids)
        done = torch.zeros(batch_size).type(torch.uint8).to(device)
        sentence = input_ids
        for _ in range(60):
            logits = self.forward(**{"input_ids" : sentence})["logits"]
            logits = logits[:, -1, :]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
          
            cumulative_probs = torch.cumsum(nn.Softmax(dim=-1)(sorted_logits), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, -1000000)
          
            probabilities = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probabilities, 1)
            pred[done != 0] = pad_token_id

            sentence = torch.cat([sentence, pred], dim=1)
            eos_match = (pred.squeeze(1) == 3)
          
            done = done | eos_match
            if done.sum() == batch_size:
                break

        return sentence









