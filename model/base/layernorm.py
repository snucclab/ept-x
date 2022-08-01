import torch
from torch.nn import LayerNorm, Module


class MaskedLayerNorm(LayerNorm):
    def forward(self, vector: torch.Tensor, ignorance_mask: torch.Tensor = None) -> torch.Tensor:
        if ignorance_mask is None or not ignorance_mask.any().item():
            return super().forward(vector)

        normalizing_dims = len(self.normalized_shape)
        last_flat_dim = - normalizing_dims - 1

        # Flatten vectors :: [F, ...]
        flat_vector = vector.flatten(start_dim=0, end_dim=last_flat_dim)
        flat_mask = ignorance_mask.flatten()  # [F]
        assert flat_vector.shape[0] == flat_mask.shape[0]

        # Fliter masked value
        masked_indices = flat_mask.logical_not().nonzero(as_tuple=False).view(-1).tolist()
        masked_vector = flat_vector[masked_indices]

        # Normalize vectors
        normalized = super().forward(masked_vector)

        # Pad with zeros for masked position
        new_flat_vector = torch.zeros_like(flat_vector)
        new_flat_vector[masked_indices] = normalized

        # Unflatten the padded vector
        return new_flat_vector.view(vector.shape)
