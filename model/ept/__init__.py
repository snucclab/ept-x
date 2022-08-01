from typing import Dict, Tuple, List

import torch
from torch import nn

from common.const.model import *
from common.const.operand import *
from common.const.operator import *
from common.const.pad import PAD_ID, NEG_INF
from common.data import *
from common.data.base import move_to
from model.base.chkpt import CheckpointingModule
from model.base.util import Squeeze, init_weights, mask_forward, logsoftmax, apply_module_dict
from model.base.beamsearch import beam_search
from .attention import MultiheadAttentionWeights
from .decoder import EquationDecoder
from .encoder import TextEncoder

MODEL_CLS = 'model'
OPR_EXCLUDED = {OPR_NEW_EQN_ID}


class EPT(CheckpointingModule):
    def __init__(self, **config):
        super().__init__(**config)
        # Encoder: [B, S] -> [B, S, H]
        self.encoder = TextEncoder.create_or_load(encoder=self.config[MDL_ENCODER])

        # Decoder
        self.equation = EquationDecoder.create_or_load(**self.config[MDL_EQUATION],
                                                       encoder_config=self.encoder.model.config)

        # Action output
        hidden_dim = self.equation.hidden_dim
        self.operator = nn.Linear(hidden_dim, OPR_SZ)
        self.operands = nn.ModuleList([nn.ModuleDict({
            '0_attn': MultiheadAttentionWeights(**{MDL_Q_HIDDEN: hidden_dim, MDL_Q_HEAD: 1}),
            '1_mean': Squeeze(dim=-1)
        }) for _ in range(OPR_MAX_ARITY)])

        # Initialize
        factor = self.equation.init_factor
        init_weights(self.operator, factor)
        self.operands.apply(lambda w: init_weights(w, factor))

    @property
    def constant_embedding(self):
        # [1, C, H]
        return self.equation.constant_word_embedding.weight.unsqueeze(0)

    @property
    def operator_embedding(self):
        # [1, V, H]
        return self.equation.operator_word_embedding.weight.unsqueeze(0)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _build_attention_keys(self, variable: Encoded, number: Encoded = None) -> Dict[str, torch.Tensor]:
        # Retrieve size information
        batch_sz, res_len, hidden_dim = variable.vector.shape
        operand_sz = NUM_END + res_len if number is not None else CON_END + res_len

        key = torch.zeros(batch_sz, operand_sz, hidden_dim, device=variable.device)
        key_ignorance_mask = torch.ones(batch_sz, operand_sz, dtype=torch.bool, device=variable.device)
        attention_mask = torch.zeros(res_len, operand_sz, dtype=torch.bool, device=variable.device)

        # Assign constant weights
        key[:, :CON_END] = self.constant_embedding
        key_ignorance_mask[:, :CON_END] = False
        offset = CON_END

        # Add numbers
        if number is not None:
            num_count = number.vector.shape[1]
            num_end = offset + num_count
            key[:, offset:num_end] = number.vector
            key_ignorance_mask[:, offset:num_end] = number.pad
            offset = offset + NUM_MAX

        # Add results
        res_end = offset + res_len
        key[:, offset:res_end] = variable.vector
        key_ignorance_mask[:, offset:res_end] = variable.pad

        attention_mask[:, offset:res_end] = mask_forward(res_len, diagonal=0).to(attention_mask.device)
        attention_mask[:, res_end:] = True

        return dict(key=key, key_ignorance_mask=key_ignorance_mask, attention_mask=attention_mask)

    def _encode(self, text: Text) -> Tuple[Encoded, Encoded]:
        return self.encoder(text)

    def _decode_equation(self, **kwargs) -> Tuple[Encoded, tuple]:
        cached = kwargs.pop('cached') if 'cached' in kwargs else None
        is_cached = (not self.training) and (cached is not None)
        if is_cached:
            prev_out, prev_pad = cached[-1]
            cached = cached[:-1]

        output, cached = self.equation.forward(**kwargs, cached=cached)

        # If cached, concatenate with the previous output
        if is_cached:
            # [B, T+1, H]
            vector = torch.cat([prev_out, output.vector], dim=1).contiguous()
            # [B, T+1]
            pad = torch.cat([prev_pad, output.pad], dim=1).contiguous()
            output = Encoded(vector, pad)

        if not self.training:
            # Reconstruct output and register cache
            cached += ((output.vector, output.pad),)

        # Ignore the result of equality at the function output
        output_not_usable = output.pad.clone()

        previous_op = kwargs['target'].operator.shifted_indices
        output_not_usable[:, :-1].masked_fill_((previous_op >= OPR_EQ_SGN_ID) & (previous_op < OPR_PLUS_ID), True)
        # We need offset '1' because 'function_word' is input and output_not_usable is 1-step shifted output.

        output = output.copy(pad=output_not_usable)
        return output, cached

    def _equation_for_train(self, predict_last: bool = False, operator_excluded: set = None,
                            **kwargs) -> Tuple[tuple, EquationPrediction]:
        if operator_excluded is None:
            operator_excluded = OPR_EXCLUDED

        assert 'text' in kwargs
        assert 'number' in kwargs
        assert 'target' in kwargs
        # text: [B,S]
        # number: [B,N]
        # target: [B,T]
        # decoded: [B,T]
        # ignore the cached result
        decoded, new_cache = self._decode_equation(**kwargs)

        # Prepare values for attention
        attention_input = self._build_attention_keys(variable=decoded, number=kwargs['number'])

        if predict_last:
            decoded = decoded[:, -1:]
            attention_input['attention_mask'] = attention_input['attention_mask'][-1:]

        # Compute operator
        operator = self.operator(decoded.vector)
        for excluded in operator_excluded:
            operator[:, :, excluded] = NEG_INF
        operator = logsoftmax(operator)

        # Compute operands: List of [B, T, N+T]
        operands = [logsoftmax(apply_module_dict(layer, encoded=decoded.vector, **attention_input))
                    for layer in self.operands]

        return new_cache, EquationPrediction.from_tensors(operator, operands)

    def _equation_for_eval(self, max_len: int = RES_MAX, beam_size: int = 3, excluded_operators: set = None,
                           **kwargs) -> Equation:
        assert 'text' in kwargs
        assert 'number' in kwargs
        text: Encoded = kwargs['text']
        number: Encoded = kwargs['number']

        text_label: Label = kwargs.get('text_label', None)
        assert text is not None or text_label is not None

        if excluded_operators is None:
            excluded_operators = set()

        def initialize_fn():
            # Initially we start with a single beam.
            batch_sz = text.shape[0] if text is not None else text_label.shape[0]
            beamscores = torch.zeros((batch_sz, 1))

            return [dict(text=text[b:b + 1] if text is not None else None,  # [1, S]
                         number=number[b:b + 1] if number is not None else None,  # [1, N]
                         text_label=text_label[b:b + 1] if text_label is not None else None,  # [1, S]
                         target=Equation.get_generation_base(),  # [1, T=1]
                         cached=None
                         )
                    for b in range(batch_sz)], beamscores

        def compute_next_score_of_beam(seq_len: int, beams: dict, k: int):
            # Shape [M]
            cached, last_pred = self._equation_for_train(**move_to(beams, self.device), predict_last=True)
            last_pred = last_pred[:, 0].to('cpu')

            # Store cache
            beams['cached'] = move_to(cached, 'cpu')

            # Compute score
            scores = []
            for m_prev in range(last_pred.operator.shape[0]):
                if seq_len > 1 and beams['target'].operator.indices[m_prev, -1].item() in {OPR_DONE_ID, PAD_ID}:
                    scores += [(0, m_prev, dict(operator=[PAD_ID], operands=[PAD_ID] * OPR_MAX_ARITY))]
                    continue

                # Take top-K position for each j-th operand
                operands = [list(zip(*[tensor.tolist()
                                       for tensor in operand_j.log_prob[m_prev].topk(k=k, dim=-1)]))
                            for operand_j in last_pred.operands]

                score_beam = []
                for f, f_info in enumerate(OPR_VALUES):
                    if (f == OPR_DONE_ID and seq_len == 1) or f in excluded_operators:
                        continue

                    arity = f_info[KEY_ARITY]
                    score_f = [(last_pred.operator.log_prob[m_prev, f], (f,))]
                    for operand_j in operands[:arity]:
                        score_f = [(score_aj + score_prev, tuple_prev + (aj,))
                                   for score_aj, aj in operand_j
                                   for score_prev, tuple_prev in score_f]

                    score_beam += [(score, m_prev, dict(operator=[f],
                                                        operands=list(a) + [PAD_ID] * (OPR_MAX_ARITY - len(a))))
                                   for score, (f, *a) in score_f]

                scores += sorted(score_beam, key=lambda t: t[0], reverse=True)[:k]

            return scores

        def concat_next_fn(prev_beams: dict, beam_selected: List[int], list_of_next: dict):
            if prev_beams['target'].shape[0] == 1:
                # Before expanding beams.
                beamsz = len(beam_selected)
                for key in prev_beams:
                    if key in {'target', 'cached'} or prev_beams[key] is None:
                        continue
                    prev_beams[key] = prev_beams[key].repeat(beamsz)

            # Extend beams
            prev_beams['target'] = prev_beams['target'][beam_selected] \
                .extends_to(next_operator=list_of_next['operator'],
                            next_operands=[list_of_next['operands'][:, j:j + 1]
                                           for j in range(OPR_MAX_ARITY)])

            # Select cache of selected beams. All have shape [M, ...], so we will shuffle only the first dim.
            prev_beams['cached'] = tuple(tuple(tensor[beam_selected] for tensor in pair)
                                         for pair in prev_beams['cached'])

            return prev_beams

        def is_all_finished(beams: dict):
            return all(f in {OPR_DONE_ID, PAD_ID}
                       for f in beams['target'].operator.indices[:, -1].tolist())

        with torch.no_grad():
            # Execute beam search. List[Dict[str, ?]]
            batched_beams = beam_search(initialize_fn, compute_next_score_of_beam,
                                        concat_next_fn, is_all_finished, max_len, beam_size)

            # Select top-scored beam
            return Equation.build_batch(*[item['target'][0] for item in batched_beams])

    def forward(self, text: Text, **kwargs):
        # Forward the encoder
        text, num_enc = self._encode(text.to(self.device))
        return_value = dict(text=text, num_enc=num_enc)

        if self.training:
            target: Equation = kwargs['equation']

            # Compute hidden states & predictions
            _, equation = self._equation_for_train(text=text, number=num_enc, target=target)

            return_value['equation'] = equation
        else:
            beam: int = kwargs.get('beam', 3)

            # Generate equation
            equation = self._equation_for_eval(text=text, number=num_enc, beam_size=beam)
            return_value['equation'] = equation

        return return_value


__all__ = ['EPT', 'MODEL_CLS']
