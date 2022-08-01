from typing import Tuple

import torch

from common.const.model import DEF_ENCODER
from common.data import Encoded, Text, Label
from model.base.chkpt import CheckpointingModule


def _gather_number_vectors(hidden: torch.Tensor, mask: torch.Tensor) -> Encoded:
    # Compute the maximum number of indicated positions in the text
    batch_size, seq_len, hidden_size = hidden.shape

    batched_items = []
    for b in range(batch_size):
        row_items = []
        row_number_max = mask[b].max().item()
        for n in range(row_number_max + 1):  # Including the last number
            indices = mask[b].eq(n).nonzero(as_tuple=False).view(-1).tolist()
            if len(indices) == 0:  # Cut because the text is too long.
                row_items.append(Encoded.zero(1, hidden_size, device=hidden.device))
                continue

            # Copy masked positions. Shape [1, H].
            mean_vec = hidden[b, indices].mean(dim=0, keepdim=True)
            row_items.append(Encoded(mean_vec, None))

        # Add batched vectors. Shape [N, H].
        if len(row_items):
            batched_items.append(Encoded.concat(*row_items, dim=0))
        else:
            batched_items.append(Encoded.empty(0, hidden_size, device=hidden.device))

    # Return [B, N, H].
    return Encoded.build_batch(*batched_items)


class TextEncoder(CheckpointingModule):
    """
    Model for encoding text.
    """

    def __init__(self, encoder: str = DEF_ENCODER):
        """
        Initiate Text Model instance.

        :param ModelConfig config: Model configuration instance
        """
        super().__init__(encoder=encoder)
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(encoder)
        self.pad_id = AutoTokenizer.from_pretrained(encoder).pad_token_id

    def _encode(self, label: Label) -> Encoded:
        # Replace PAD_ID (-1) with pad of tokenizer
        model_out = self.model.forward(input_ids=label.pad_fill(self.pad_id),
                                       attention_mask=label.attn_mask_float)[0]

        return Encoded(model_out, label.pad)

    def forward(self, text: Text) -> Tuple[Encoded, Encoded]:
        with torch.no_grad():
            # Find the last non-pad position
            text_length = text.sequence_lengths.max().item()
            # Cut off padded items to reduce GPU usage
            text: Text = text[:, :text_length]

        # Encode text
        # Form an encoded output
        encoded = self._encode(text.tokens)

        # Gather numbers
        number_out = _gather_number_vectors(encoded.vector, text.numbers.indices)

        return encoded, number_out


__all__ = ['TextEncoder']
