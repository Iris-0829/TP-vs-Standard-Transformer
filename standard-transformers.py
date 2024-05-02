import math

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class TPDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, max_length, dropout=0.1):
        super(TPDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length

        self.word_embeddings = EmbeddingMultilinearSinusoidal(vocab_size, hidden_size, dropout)
        self.layers = nn.ModuleList([TPDecoderLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None):
        # Embedding
        embedded = self.word_embeddings(input_ids)

        # Create attention mask
        attn_mask = self.create_attn_mask(input_ids)

        # Pass through decoder layers
        for layer in self.layers:
            embedded = layer(embedded, attn_mask)

        # Projection to output vocabulary
        output = self.output_projection(self.norm(embedded))

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = output[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss
        return output

    def create_attn_mask(self, input_ids):
        batch_size, seq_len = input_ids.size()
        attn_mask = torch.tril(torch.ones(seq_len, seq_len)).expand(batch_size, 1, seq_len, seq_len)
        return attn_mask.to(input_ids.device)

    def generate(self, input_ids, do_sample=True, max_length=500, top_p=1.0, top_k=0, early_stopping=True,
                 pad_token_id=None, eos_token_id=3):
        with torch.no_grad():
            generated_ids = input_ids

            for _ in range(max_length - input_ids.size(1)):
                # Forward pass
                logits = self.forward(generated_ids)

                # Get the last token logits
                next_token_logits = logits[:, -1, :]

                # Apply top-k and top-p filtering
                if top_k > 0:
                    top_k = min(top_k,
                                next_token_logits.size(-1))  # Adjust top_k if it's greater than the vocabulary size
                    topk_logits, topk_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    next_token_logits[next_token_logits < topk_logits[..., -1, None]] = -float('inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices,
                                                                         sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')

                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

                # Append the generated token to the sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)

                # Stop generation if the end-of-sequence token is encountered
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

            return generated_ids


class TPDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(TPDecoderLayer, self).__init__()

        self.self_attn = SelfAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = PositionwiseFeedforward(hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        # Self-attention with Transformer Product
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, attn_mask)))

        # Feedforward
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x


# Transformer Product Self-Attention
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.r_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, q, k, v, attn_mask):
        batch_size, seq_len, _ = q.size()

        q = self.q_proj(q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        r = self.r_proj(q.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size))
        r = r.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.einsum('bnqd,bnkd->bnqk', q, k) / self.scale
        attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        x = torch.einsum('bnqk,bnkd->bnqd', attn_probs, v)
        x = x * r  # Bind
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        x = self.o_proj(x)

        return x


# Embedding with Multilinear and Sinusoidal positional encoding
class EmbeddingMultilinearSinusoidal(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size]))

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.mul_scale = torch.FloatTensor([1 / math.sqrt(math.sqrt(2) - 1)])

        self.reset_parameters()

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        seq_len = input_ids.size(1)

        token_emb = self.token_embedding(input_ids) * self.scale

        position = torch.arange(0, seq_len, dtype=torch.float, device=input_ids.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2, device=input_ids.device).float() * (
                    -math.log(10000.0) / self.hidden_size))
        pe = torch.zeros(seq_len, self.hidden_size, device=input_ids.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        x = token_emb + pe
        r = self.linear(x) + 1
        z = x * r
        z = self.dropout(z)
        return z

    def reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, mean=0, std=1 / math.sqrt(self.hidden_size))
        nn.init.normal_(self.linear.weight, mean=0, std=1 / math.sqrt(self.hidden_size))


# Feedforward layer
class PositionwiseFeedforward(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


