from typing import Tuple, Optional

import torch

from common.const.model import *
from common.const.operand import VAR_MAX
from common.const.pad import PAD_ID, UNEXPLAINED_NUMBER
from common.data import Encoded, Label
from model.base.chkpt import *


class ExplanationDecoder(CheckpointingModule):
    """
    Base model for equation generation/classification (Abstract class)
    """

    def __init__(self, encoder: str = DEF_ENCODER, **kwargs):
        """
        Initiate Equation Builder instance

        :param dict config: Configuration of this model
        """
        super().__init__(encoder=encoder)

        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(encoder, add_cross_attention=True, is_decoder=True)

        tokenizer = AutoTokenizer.from_pretrained(encoder)
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id
        self.tokens_ignored = {PAD_ID, self.eos_id}
        self.empty_sequence = tokenizer.encode(UNEXPLAINED_NUMBER, add_special_tokens=False)

        self.tokenizer = tokenizer
        self.embed_dim = model.config.embedding_size

        # Copy encoder and embeddings
        self.embeddings = model.embeddings
        self.encoder = model.encoder
        self.extended_attention_mask = model.get_extended_attention_mask
        self.invert_attention_mask = model.invert_attention_mask
        if hasattr(model, 'embeddings_project'):
            self.embeddings_project = model.embeddings_project

        # Register prefix for explaining number (exclude [SEP] at the end)
        # Shape [P]
        self.register_buffer('_prefix_number', torch.LongTensor(tokenizer.encode('explain:')[:-1]))

        # Store explanation labels for variables
        self.var_labels = Label.from_list([tokenizer.encode('variable x%s' % i, add_special_tokens=False)
                                           for i in range(VAR_MAX)])

    @property
    def prefix_length(self) -> int:
        return self._prefix_number.shape[0]

    def forward(self, text: Encoded, target: Label, **kwargs) -> Tuple[Encoded, Encoded, Optional[tuple], int]:
        # text: [B,S]
        # target: [B,D]
        # out: [B,D]

        # expl_label?: [B,T]
        expl_label: Label = kwargs.get('expl_label', None)

        # Whether key-value pair is cached or not
        cached = kwargs.get('cached', None)
        is_cached = (not self.training) and (cached is not None)

        # Compute input embedding
        word_emb, full_emb, prefix_len = self.build_decoder_input(expl_label, target)

        # Compute hidden state vectors
        encoded, cached = self.build_decoder_context(full_emb, text, cached)

        if is_cached:
            # Cached: we need only the last token (encoded has already been cut)
            word_emb = word_emb[:, -1:]

        return encoded, word_emb, cached, (0 if is_cached else prefix_len)

    def build_decoder_context(self, embedding: Encoded, text: Encoded = None,
                              prev_key_value: tuple = None) -> Tuple[Encoded, tuple]:
        if (not self.training) and (prev_key_value is not None):
            # Cached: we need only the last token
            embedding = embedding[:, -1:]

        # Build attention masks
        # Note: we need full mask (raw_input_ids.attn_mask_float) even if we cached
        extended_attention_mask = self.extended_attention_mask(embedding.attn_mask_float,
                                                               embedding.shape, embedding.device)
        extended_text_mask = self.invert_attention_mask(text.attn_mask_float) if text is not None else None

        # Compute hidden states [B, P+D, H]
        outputs = self.encoder.forward(
            embedding.vector,
            attention_mask=extended_attention_mask,  # [B, H, P+D, P+D]
            head_mask=[None] * self.encoder.config.num_hidden_layers,
            encoder_hidden_states=text.vector if text is not None else None,
            encoder_attention_mask=extended_text_mask,  # [B, ?, ?, S]
            output_attentions=False,
            output_hidden_states=False,
            past_key_values=prev_key_value,
            return_dict=True,
            use_cache=not self.training  # Use caching if this is for evaluation
        )
        # Truncate the prefix [B, D]
        encoded = Encoded(outputs.last_hidden_state, embedding.pad)
        # On evaluation, return cached output. otherwise None
        next_key_value = None if self.training else outputs.past_key_values

        return encoded, next_key_value

    def build_decoder_input(self, context, explanation, prefix: Label = None):
        # Add prefix
        # We want: "explain a number <VECTOR VALUES> <TARGET TOKENS>"
        # Concatenate prefix and snippet labels.  [P] + [B, T] -> [B, P+T]
        context_input = context.prepend(self._prefix_number)
        context_len = context_input.shape[-1]
        # Extend target with prefix. [B, D] -> [B, P+T+D]
        input_ids = Label.concat(prefix, context_input, explanation, dim=1)

        # Build token-type indices. [T] -> [1, T]
        token_type = torch.arange(input_ids.shape[-1]).ge(context_len).long().unsqueeze(0).to(input_ids.device)

        # As we may add 'expl_label' vector and do want to apply it after adding the vector,
        # we will explicitly call word_embedding here.
        word = self.embeddings.word_embeddings(input_ids.pad_fill(self.pad_id))

        # Compute entire embedding [B, P+D, H] or [B, 1, H]
        embeddings = self.embeddings(inputs_embeds=word, token_type_ids=token_type)
        if hasattr(self, 'embeddings_project'):
            embeddings = self.embeddings_project(embeddings)

        # Wrap as Encoded instance
        word = Encoded(word, input_ids.pad)
        embeddings = Encoded(embeddings, input_ids.pad)

        return word, embeddings, context_len


__all__ = ['ExplanationDecoder']
