from typing import Callable, Optional

import torch

from common.torch.util import stack_tensors, concat_tensors
from .base import TypeTensorBatchable, TypeSelectable


class Encoded(TypeTensorBatchable, TypeSelectable):
    vector: torch.Tensor
    pad: torch.Tensor

    def __init__(self, vector: torch.Tensor, pad: Optional[torch.Tensor]):
        super().__init__()
        if pad is None:
            pad = torch.zeros(vector.shape[:-1], dtype=torch.bool, device=vector.device)

        assert vector.shape[:-1] == pad.shape
        self.vector = vector
        self.pad = pad
    
    def __add__(self, other: 'Encoded') -> 'Encoded':
        assert self.vector.shape == other.vector.shape
        return Encoded(self.vector + other.vector, self.pad)

    def __mul__(self, other: float) -> 'Encoded':
        return Encoded(self.vector * other, self.pad)

    @property
    def shape(self) -> torch.Size:
        return self.pad.shape

    @property
    def device(self) -> torch.device:
        return self.vector.device

    @property
    def sequence_lengths(self) -> torch.LongTensor:
        return self.pad.logical_not().sum(dim=-1)

    @property
    def attn_mask_float(self) -> torch.Tensor:
        return self.pad.logical_not().float()

    @property
    def pooled_state(self) -> 'Encoded':
        sum_of_states = self.vector.masked_fill(self.pad.unsqueeze(-1), 0).sum(dim=-2)
        len_of_states = self.sequence_lengths
        pooled = sum_of_states / len_of_states.unsqueeze(-1)
        pooled_pad = len_of_states.eq(0)
        return Encoded(pooled, pooled_pad)

    @classmethod
    def build_batch(cls, *items: 'Encoded') -> 'Encoded':
        vectors = stack_tensors([item.vector for item in items], pad_value=0.0)
        pads = stack_tensors([item.pad for item in items], pad_value=True)
        return Encoded(vectors, pads)

    @classmethod
    def concat(cls, *items: 'Encoded', dim: int = 0) -> 'Encoded':
        vectors = concat_tensors([item.vector for item in items], dim=dim, pad_value=0.0)
        pads = concat_tensors([item.pad for item in items], dim=dim, pad_value=True)
        return Encoded(vectors, pads)

    @classmethod
    def empty(cls, *shape: int, device='cpu') -> 'Encoded':
        return Encoded(torch.empty(*shape, device=device),
                       torch.empty(*shape[:-1], dtype=torch.bool, device=device))

    @classmethod
    def zero(cls, *shape: int, device='cpu') -> 'Encoded':
        return Encoded(torch.zeros(*shape, device=device),
                       torch.ones(*shape[:-1], dtype=torch.bool, device=device))

    def as_dict(self) -> dict:
        return dict(vector=self.vector, pad=self.pad)

    def repeat(self, n: int) -> 'Encoded':
        return Encoded(vector=self.vector.expand((n,) + self.vector.shape[1:]),
                       pad=self.pad.expand((n,) + self.pad.shape[1:]))

    def detach(self) -> 'Encoded':
        return Encoded(self.vector.detach(), self.pad.detach())

    def unsqueeze(self, dim: int) -> 'Encoded':
        return Encoded(self.vector.unsqueeze(dim), self.pad.unsqueeze(dim))

    def pad_fill(self, fill_value: float) -> torch.Tensor:
        return self.vector.masked_fill(self.pad.unsqueeze(-1), fill_value)

    def apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> 'Encoded':
        new_vector = fn(self.vector)
        assert new_vector.shape[:-1] == self.shape

        return Encoded(new_vector, self.pad)

    def to_human_readable(self, **kwargs) -> dict:
        return {
            'shape': self.shape
        }
