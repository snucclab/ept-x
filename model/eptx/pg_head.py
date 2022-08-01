from typing import Tuple

import torch
from torch import nn

from common.const.model import DEF_Q_HIDDEN, MDL_Q_HIDDEN, MDL_Q_HEAD, DEF_Q_EMBED
from common.const.operand import VOCAB_MAX
from common.const.pad import NEG_INF, PAD_ID
from common.data import Encoded, Label
from model.base.util import init_weights, logsoftmax
from model.ept.attention import MultiheadAttentionWeights


class PointerGeneratorHead(nn.Module):
    def __init__(self, hidden_dim: int = DEF_Q_HIDDEN, embed_dim: int = DEF_Q_EMBED, vocab_size: int = VOCAB_MAX,
                 init_factor: float = 0.01):
        super().__init__()

        # Single-head attention score layer
        self.encoder_attention = MultiheadAttentionWeights(**{MDL_Q_HIDDEN: hidden_dim, MDL_Q_HEAD: 1})
        if hidden_dim != embed_dim:
            self.hidden_to_embed = torch.nn.Linear(hidden_dim, embed_dim)

        # Generation distribution layer
        self.generation_dist = torch.nn.Linear(embed_dim, vocab_size)
        self.vocab_size = vocab_size

        # W_h h^*_t + W_s s_t + W_x x_t + b
        self.generation_prob_linear = nn.Linear(in_features=hidden_dim * 2 + embed_dim, out_features=1)
        self.apply(lambda module: init_weights(module, init_factor))

        self.log_sigmoid = nn.LogSigmoid()

    def _compute_attention(self, text: Encoded, decoded: Encoded,
                           prev_key: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        # Text: [B, S, H]
        # Decoded: [B, T, H]

        # Attention score: [B, T, S, N=1]
        attn_score, new_key = self.encoder_attention.forward(query=decoded.vector, key=text.vector,
                                                             key_ignorance_mask=text.pad, prev_key=prev_key,
                                                             head_at_last=True, is_self=False)

        # Apply softmax
        attn_score = logsoftmax(attn_score.squeeze(-1))

        # Set score as zero on padding in decoded.
        attn_score = attn_score.masked_fill(decoded.pad.unsqueeze(-1), NEG_INF)

        # Compute attented vector h_t^* [B, T, H] = [B, T, S] * [B, S, H]
        attented_vector = torch.bmm(attn_score.exp(), text.vector)

        return attented_vector, new_key, attn_score

    def _generation_probability(self, text_attn: torch.Tensor, decoded: torch.Tensor,
                                embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Text: [B, T, H]
        # Decoded: [B, T, H]
        # Embedding: [B, T, H]

        # [B, T, 3H]
        concatenated = torch.cat([text_attn, decoded, embedding], dim=-1)
        before_sigmoid = self.generation_prob_linear(concatenated)
        after_logsig = self.log_sigmoid(before_sigmoid)

        # Return transformed result [B, T, 1]
        return after_logsig, after_logsig - before_sigmoid

    def forward(self, text: Encoded, text_label: Label, decoded: Encoded, decoder_embedding: Encoded,
                prev_key: torch.Tensor = None, pad_value: int = 0) -> Tuple[torch.Tensor, tuple]:
        # Compute distribution of token generation
        if hasattr(self, 'hidden_to_embed'):
            decoded_vector = self.hidden_to_embed(decoded.vector)
        else:
            decoded_vector = decoded.vector
        gen_dist = logsoftmax(self.generation_dist(decoded_vector))

        # if text is None, then we cannot use copying method.
        if text is None:
            return gen_dist, (tuple(),)  # We need to return tuple to pass type checking.

        # Compute attention.
        # [B, T, H], ?, [B, T, S].
        text_attented, new_key, attn_score = self._compute_attention(text, decoded, prev_key=prev_key)

        # Compute generation probability
        gen_prob, copy_prob = self._generation_probability(text_attented, decoded.vector, decoder_embedding.vector)

        # Expand index to [B, T, S]
        text_label = text_label.indices

        # Copying probability
        copy_attn = (attn_score + copy_prob).exp()
        
        if torch.are_deterministic_algorithms_enabled():
            # Use manual but deterministic algorithm (scatter_add is non-deterministic on CUDA)
            copy_dist = torch.zeros_like(gen_dist)  # [B, T, V]
            batch_sz, text_len = text_label.shape
            for b in range(batch_sz):
                for s in range(text_len):
                    text_bs = text_label[b, s].item()
                    if text_bs == PAD_ID:
                        continue
                    copy_dist[b, :, text_bs] += copy_attn[b, :, s]
        else:
            copy_dist = torch.zeros_like(gen_dist).scatter_add(dim=-1, index=text_label, src=copy_attn)

        # Generating probability (P_gen * Vocab)
        gen_dist = (gen_dist + gen_prob).exp()

        # Add copying to generation & return as log-probability
        logprob = (copy_dist + gen_dist).log()
        logprob = logprob.masked_fill(torch.isfinite(logprob).logical_not(), NEG_INF)
        return logprob, (new_key,)
