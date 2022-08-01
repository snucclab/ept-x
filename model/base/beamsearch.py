from typing import Callable, Tuple, List

import torch

from common.const.operand import RES_MAX


def _length_penalty(score: float, seq_len: int, alpha: float):
    if alpha <= 0:
        return score

    # Following:
    # Wu et al (2016) Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation
    penalty = ((5 + seq_len) / (5 + 1)) ** alpha
    return score * penalty


def beam_search(initialize_fn: Callable[[], Tuple[List[dict], torch.Tensor]],
                compute_score_fn: Callable[[int, dict, int], List[Tuple[float, int, dict]]],
                concat_next_fn: Callable[[dict, List[int], dict], dict],
                is_item_finished: Callable[[dict], bool],
                max_len: int = RES_MAX, beam_size: int = 3,
                len_penalty_alpha: float = 0):
    # List of beams. B * [M, ...] and Tensor of [B, M].
    batch, beamscores = initialize_fn()
    finished = False

    # From 1 to HORIZON.
    for seq_len in range(1, max_len):
        if finished:
            break

        next_beamscores = torch.zeros(beamscores.shape[0], beam_size)
        next_batch = []
        finished = True
        for i, item in enumerate(batch):
            if seq_len > 1 and is_item_finished(item):
                # If all beams of this item is done, this item will not be computed anymore.
                next_batch.append(item)
                continue

            # Compute scores
            score_i = [(_length_penalty(score + beamscores[i, m_prev], seq_len, len_penalty_alpha), m_prev, predicted)
                       for score, m_prev, predicted in compute_score_fn(seq_len, item, beam_size)]
            score_i = sorted(score_i, key=lambda t: t[0], reverse=True)[:beam_size]

            # Construct the next beams
            prev_beam_order = [m_prev for _, m_prev, _ in score_i]
            predictions = [prediction for _, _, prediction in score_i]
            next_tokens = {key: torch.LongTensor([prediction[key]
                                                  for prediction in predictions])  # Shape: [M, ?]
                           for key in predictions[0]}

            next_batch.append(concat_next_fn(item, prev_beam_order, next_tokens))
            finished = finished and is_item_finished(next_batch[-1])
            for m_new, (score, m_prev, predicted) in enumerate(score_i):
                next_beamscores[i, m_new] = score

        batch = next_batch
        beamscores = next_beamscores

    return batch