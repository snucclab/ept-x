from typing import Tuple, List, Optional

import torch
from numpy.random import Generator, PCG64, randint

from common.const.model import *
from common.data import *
from common.const.operand import *
from common.const.operator import OPR_NEW_VAR_ID, OPR_NEW_EQN_ID
from common.const.pad import PAD_ID
from common.data import Text, Equation, Explanation, Encoded, EquationPrediction, Label
from common.data.base import move_to
from model.base.util import init_weights, tie_lm_head_with_embed, logsoftmax
from model.base.beamsearch import beam_search
from model.ept import *
from .explain import ExplanationDecoder
from .pg_head import PointerGeneratorHead

EPTX_OPERATOR_EXCLUDED = {OPR_NEW_EQN_ID, OPR_NEW_VAR_ID}


class EPTXBase(EPT):
    def __init__(self, **config):
        super().__init__(**config)

        # Explanation Decoder
        self.explanation = ExplanationDecoder.create_or_load(**self.config[MDL_EXPLANATION])

        # Head for predicting explanation
        self.explanation_pghead = PointerGeneratorHead(hidden_dim=self.equation.hidden_dim,
                                                       embed_dim=self.explanation.embed_dim,
                                                       vocab_size=self.encoder.model.config.vocab_size,
                                                       init_factor=self.equation.init_factor)
        tie_lm_head_with_embed(self.explanation_pghead.generation_dist, self.explanation.embeddings.word_embeddings)

        # Variable counts (as regression problem)
        self.var_count_expand = torch.nn.Linear(self.equation.hidden_dim, self.equation.intermediate_dim)
        self.var_count_predict = torch.nn.Linear(self.equation.intermediate_dim, VAR_MAX)

        init_weights(self.var_count_expand, self.equation.init_factor)
        init_weights(self.var_count_predict, self.equation.init_factor)

        self._what_is = self.explanation.tokenizer.encode('what is ?', add_special_tokens=False)
        self._is_a_number = self.explanation.tokenizer.encode('is a number.', add_special_tokens=False)
        self._recombine_policy = self._recombine_policy_default = None
        self._rng = Generator(PCG64(1))

    @property
    def _sep_token(self) -> int:
        return self.explanation.eos_id

    @property
    def _cls_token(self) -> int:
        return self.explanation.bos_id

    @property
    def _pad_token(self) -> int:
        return self.explanation.pad_id

    @property
    def _mask_token(self) -> int:
        return self.explanation.mask_id

    @property
    def _shuffle_on_training(self) -> bool:
        return self.config[MDL_EXPLANATION].get(MDL_X_SHUFFLE_ON_TRAIN, True)

    def _get_recombine_policy(self, size: int) -> List[Tuple[bool, bool]]:
        policy = self._recombine_policy
        if policy == MDL_X_R_BOTH:
            # Return (text, reconstructed)
            return [(True, True)] * size
        if policy == MDL_X_R_COMP:
            # Ensuring Comprehensiveness requires always using the problem text
            return [(True, False)] * size
        if policy == MDL_X_R_SUFF:
            # Ensuring Sufficiency requires always using the reconstructed text
            return [(False, True)] * size
        # Randomized policy (training only)
        if self.training:
            return [(r < policy, r >= policy)
                    for r in torch.rand(size).tolist()]
        else:
            return [(False, True)] * size

    def _predict_var_count(self, encoded: Encoded, **kwargs) -> torch.Tensor:
        # Value should be at least 1.0
        return self.var_count_predict(self.var_count_expand(encoded.vector[:, 0]).relu())

    def _equation_for_train(self, predict_last: bool = False,
                            **kwargs) -> Tuple[tuple, EquationPrediction]:
        # Exclude NEW_VAR operator
        return super()._equation_for_train(predict_last=predict_last, operator_excluded=EPTX_OPERATOR_EXCLUDED,
                                           **kwargs)

    def _equation_for_eval(self, **kwargs) -> Equation:
        # Exclude NEW_VAR operator
        return super()._equation_for_eval(**kwargs, excluded_operators={OPR_NEW_VAR_ID})

    def _decode_explanation(self, **kwargs) -> Tuple[Encoded, Encoded, tuple, int]:
        return self.explanation.forward(**kwargs)

    def _explanation_for_train(self, **kwargs) -> Tuple[Encoded, tuple, Optional[Prediction]]:
        if 'cached' in kwargs and kwargs['cached'] is not None:
            # Reset cached keys
            cached = kwargs.pop('cached')
            kwargs['cached'] = cached[:-1]
            head_cache = cached[-1][0]
        else:
            head_cache = None

        # out: [B,D]
        expl_enc, expl_emb, key_value_cache, prefix_len = self._decode_explanation(**kwargs)

        if kwargs.get('no_pred', False):
            return expl_enc, key_value_cache, None
        else:
            predicted, head_cache = \
                self.explanation_pghead.forward(text=kwargs['text'], text_label=kwargs['text_label'],
                                                prev_key=head_cache,
                                                pad_value=self._pad_token, decoded=expl_enc[:, prefix_len:],
                                                decoder_embedding=expl_emb[:, prefix_len:])
            # Append cache
            if key_value_cache is not None:
                key_value_cache = key_value_cache + (head_cache,)

            return expl_enc, key_value_cache, Prediction(predicted)

    def _explanation_for_eval(self, max_len: int = EXPL_MAX, beam_size: int = 5, **kwargs) -> List[Label]:
        assert 'text' in kwargs
        # text: [B,S]
        text: Encoded = kwargs['text']
        # text_label: [B,S]
        text_label: Label = kwargs['text_label']
        # expl_label: B-list of [N,X]
        expl_label: List[Label] = kwargs['expl_label']

        batch_sz = len(expl_label)

        # out: [B,N,D]
        # beam: [B,N,M,D]. <-- [BN, M, D]
        lengths = [item.shape[0] for item in expl_label]

        def initialize_fn():
            # Initially we start with a single beam.
            flattened_items = []
            for b in range(batch_sz):
                text_b = text[b:b + 1] if text is not None else None  # [1, S]
                text_label_b = text_label[b:b + 1]  # [1, S]

                for n in range(lengths[b]):
                    expl_for_bn = expl_label[b][n:n + 1]
                    flattened_items.append(dict(text=text_b,  # [1, S]
                                                text_label=text_label_b,  # [1, S]
                                                expl_label=expl_for_bn,  # [1, 1] or [1, T]
                                                target=Label.from_list([[self._sep_token]])))  # [M=1, T=1]

            beamscores = torch.zeros((len(flattened_items), 1))
            return flattened_items, beamscores

        def compute_next_score_of_beam(seq_len: int, beams: dict, k: int):
            # Shape [M, T]
            _, kv_cache, expl_pred = self._explanation_for_train(**move_to(beams, self.device))
            # Shape [M]
            last_pred: Prediction = expl_pred[:, -1].to('cpu')
            # Shape [M, T]
            target: Label = beams['target']
            # Assign cache
            beams['cached'] = move_to(kv_cache, 'cpu')

            scores = []
            for m_prev in range(target.shape[0]):
                if seq_len > 1 and target.indices[m_prev, -1].item() in {self._sep_token, PAD_ID}:
                    scores += [(0, m_prev, dict(target=[PAD_ID]))]
                    continue

                score_m, token_m = last_pred.log_prob[m_prev].topk(k=k + 1, dim=-1)
                for score, tok in zip(score_m.tolist(), token_m.tolist()):
                    if tok == self._sep_token and seq_len == 1:
                        continue

                    scores.append((score, m_prev, dict(target=[tok])))

            return scores

        def concat_next_fn(prev_beams: dict, beam_selected: List[int], list_of_next: dict):
            if prev_beams['expl_label'].shape[0] == 1:
                # Before expanding beams.
                beamsz = len(beam_selected)
                for key in prev_beams:
                    if key in {'cached', 'target'} or prev_beams[key] is None:
                        continue
                    prev_beams[key] = prev_beams[key].repeat(beamsz)

            prev_beams['target'] = prev_beams['target'][beam_selected].extends_to(list_of_next['target'])

            # Select cache of selected beams. All have shape [M, N, ?, H], so we will shuffle only the first dim.
            prev_beams['cached'] = tuple(tuple(tensor[beam_selected] for tensor in pair)
                                         for pair in prev_beams['cached'])

            return prev_beams

        def is_all_finished(beams: dict):
            return all(f in {self._sep_token, PAD_ID}
                       for f in beams['target'].indices[:, -1].tolist())

        with torch.no_grad():
            # Execute beam search. List[Dict[str, ?]]
            batched_beams = beam_search(initialize_fn, compute_next_score_of_beam,
                                        concat_next_fn, is_all_finished, max_len, beam_size)

            # Select top-scored beam
            explanations = []
            for b, len_b in enumerate(lengths):
                if len_b > 0:
                    explanations.append(Label.build_batch(*[item['target'][0]
                                                            for item in batched_beams[:len_b]]))
                    batched_beams = batched_beams[len_b:]
                else:
                    # Add empty explanation, [0, 0]
                    explanations.append(Label(torch.full((0, 0), fill_value=PAD_ID, dtype=torch.long)))

            return explanations

    def _explanation_batched_for_train(self, text: Optional[Encoded], text_label: Label,
                                       expl_label: List[Label], target: List[Label],
                                       no_pred: bool = False) -> Tuple[List[Encoded], List[Prediction]]:
        encoded = []
        predictions = []

        for b, expl_b in enumerate(target):
            # expl_b: [N, D]
            num_snippet_b = expl_label[b]  # [N, T]
            num_sz = num_snippet_b.shape[0]

            text_b = text[b:b + 1].repeat(num_sz) if text is not None else None  # [1, S]
            text_label_b = text_label[b:b + 1].repeat(num_sz)  # [1, S]

            expl_enc, _, expl_pred = \
                self._explanation_for_train(text=text_b, text_label=text_label_b, expl_label=num_snippet_b,
                                            target=expl_b, no_pred=no_pred)

            encoded.append(expl_enc)
            predictions.append(expl_pred)

        # Return encoded B-List of [N, D] and prediction B-List of [N, D]
        return encoded, predictions

    def set_recombine_policy(self, use_text_prob: float = None):
        if use_text_prob is None:
            self._recombine_policy = self._recombine_policy_default
        else:
            assert use_text_prob == MDL_X_R_BOTH or 0 <= use_text_prob <= 1
            self._recombine_policy = use_text_prob

    def encode_text_step101(self, text: Text) -> dict:
        text_vec, num_enc = self._encode(text.to(self.device))
        return dict(_text=text_vec, _number=num_enc)

    def predict_varcount_step102(self, _text: Encoded = None, _var_lengths: List[int] = None,
                                 dont_generate_expl: bool = False, **kwargs) -> dict:
        return_value = {}
        var_len_tensor = self._predict_var_count(_text, **kwargs)
        if self.training:
            return_value['var_len'] = Prediction(logsoftmax(var_len_tensor))  # [B, |V|]
            # Index should begin from 0
            return_value['var_len_tgt'] = Label(torch.tensor(_var_lengths, device=self.device, dtype=torch.long) - 1)
        elif not dont_generate_expl:
            # Override var_len_list variable with predicted result
            return_value['_var_lengths'] = [int(max_id) + 1  # Var count should begin with 1
                                            for max_id in var_len_tensor.argmax(dim=-1).tolist()]
        return return_value

    def generate_expl_step103(self, text: Text, beam: int = 3,
                              _num_expl: List[Label] = None, _var_expl: List[Label] = None, _text: Encoded = None,
                              _var_lengths: List[int] = None, enforce_training: bool = False, **kwargs) -> dict:
        return_value = {}

        if self.training or enforce_training:
            # Case: Training
            # 1-3-1. Prepare argument for generating explanation
            generate_kwargs = {
                'var': {
                    'expl_label': [self.explanation.var_labels[:v].to(self.device)
                                   for v in _var_lengths],
                    'target': move_to(_var_expl, self.device)
                },
                'num': {
                    'expl_label': move_to(text.snippets, self.device),
                    'target': move_to(_num_expl, self.device)
                }
            }

            # 1-3-2. Run prediction for each target
            for key, kwg in generate_kwargs.items():
                enc, pred = self._explanation_batched_for_train(text=_text, text_label=text.tokens, **kwg)
                return_value.update({
                    '%s_expl' % key: pred,
                    '_%s_expl_enc' % key: enc
                })
        else:
            # Case: Evaluation & generation required (by default)
            # 1-3-1. Prepare argument for generating explanation
            generate_kwargs = {
                'var': {
                    'expl_label': [self.explanation.var_labels[:v].to(self.device)
                                   for v in _var_lengths],
                    'beam_size': beam
                },
                'num': {
                    'expl_label': text.snippets,
                    'beam_size': beam
                }
            }

            # 1-3-2. Run prediction for each target
            for key, kwg in generate_kwargs.items():
                expl = self._explanation_for_eval(text=_text, text_label=text.tokens, **kwg)
                return_value.update({
                    '%s_expl' % key: expl,
                    '_%s_expl' % key: expl  #: Copy for internal use
                })

            return_value['explanation'] = [Explanation([n], [v], 0)
                                           for n, v in zip(return_value['num_expl'],
                                                           return_value['var_expl'])]

        return return_value

    def reconstruct_problem_step201(self, text: Text, num_expl: List[Label], var_expl: List[Label], **kwargs) -> Text:
        with torch.no_grad():
            batch_sz = len(num_expl)
            concat_labels = []
            concat_numbers = []

            for b, (use_text, use_recon) in enumerate(self._get_recombine_policy(batch_sz)):
                assert use_text or use_recon
                text_b = text[b].tokens.indices
                textnum_b = text[b].numbers.indices
                num_max = num_expl[b].shape[0]

                if use_text:
                    concat_b = [tok for tok in text_b.tolist() if tok != PAD_ID]
                    if use_recon:
                        numpos_b = [PAD_ID] * len(concat_b)
                    else:
                        numpos_b = [tok for tok in textnum_b.tolist()][:len(concat_b)]
                else:
                    concat_b = [self._cls_token]
                    numpos_b = [PAD_ID]

                if use_recon:
                    expl_b = num_expl[b].indices.tolist() + var_expl[b].indices.tolist()
                    concat_set_b = []
                    numpos_set_b = []
                    for nid, expl_bn in enumerate(expl_b):
                        expl_bn = [tok for tok in expl_bn if tok not in self.explanation.tokens_ignored]
                        if expl_bn == self.explanation.empty_sequence:
                            continue

                        num_bn = [PAD_ID] * len(expl_bn)

                        if nid < num_max:
                            expl_bn += self._is_a_number
                            num_bn += [PAD_ID] * len(self._is_a_number)
                            num_bn[-2] = nid

                            num_tokens = text_b.masked_select(textnum_b.eq(nid)).tolist()
                            expl_bn = expl_bn[:-1] + num_tokens + expl_bn[-1:]
                            num_bn = num_bn[:-1] + ([nid] * len(num_tokens)) + num_bn[-1:]
                        else:
                            expl_bn = self._what_is[:-1] + expl_bn + self._what_is[-1:]
                            num_bn = [PAD_ID] * len(expl_bn)
                            num_bn[0] = nid

                        concat_set_b.append(expl_bn)
                        numpos_set_b.append(num_bn)

                        assert len(concat_set_b) == len(numpos_set_b) > 0
                        assert nid in num_bn

                    # Concatenate all explanations
                    concat_b += sum(concat_set_b, [])
                    numpos_b += sum(numpos_set_b, [])
                else:
                    # We should add variables as EPTX treats variables as written numbers.
                    for vid in range(var_expl[b].shape[0]):
                        var_text = self.explanation.var_labels[vid].indices.tolist()
                        concat_b += var_text
                        numpos_b += [vid + num_max] * len(var_text)

                concat_b.append(self._sep_token)
                numpos_b.append(PAD_ID)

                if len(concat_b) > 500:
                    concat_b = concat_b[-500:]
                    numpos_b = numpos_b[-500:]

                concat_labels.append(Label.from_list(concat_b))
                concat_numbers.append(Label.from_list(numpos_b))

            return Text(raw=None, tokens=Label.build_batch(*concat_labels), numbers=Label.build_batch(*concat_numbers),
                        snippets=None)

    def generate_eqn_step203(self, equation: Equation, _text: Encoded = None, _number: Encoded = None,
                             beam: int = 3, **kwargs) -> dict:
        return_value = {}
        eqn_kwargs = {} if _number is not None else {'text_label': kwargs['_label'],
                                                     'num_label': kwargs['_num_label']}

        if self.training:
            number_len: List[int] = kwargs['number_len']
            eqn_tgt: Equation = equation.treat_variables_as_defined(number_len)
            if _number is None:
                _num_label: Label = kwargs['_num_label']
                eqn_tgt: Equation = eqn_tgt.treat_text_as_prev_result(_num_label)

            equation = self._equation_for_train(target=eqn_tgt, text=_text, number=_number, **eqn_kwargs)[-1]
            return_value['equation'] = equation
            return_value['equation_tgt'] = eqn_tgt
        else:
            equation = self._equation_for_eval(text=_text, number=_number, beam_size=beam, **eqn_kwargs)
            if _number is None:
                _num_label: Label = kwargs['_num_label']
                return_value['equation'] = equation.restore_numbers(_num_label)

            number_len: List[int] = kwargs['number_len']
            variable_len: List[int] = kwargs['variable_len']
            return_value['equation'] = equation.restore_variables(number_len, variable_len)

        return return_value

    def forward_explanation(self, text: Text, explanation: List[Explanation] = None,
                            dont_generate_expl: bool = False, beam_expl: int = 3, **kwargs) -> Tuple[dict, dict]:
        # (1-0) Prepare kwargs
        if self.training:
            num_expl = [d.number_for_train for d in explanation]
            var_expl = [d.variable_for_train for d in explanation]
            var_lengths = [d.shape[0] for d in var_expl]
        elif dont_generate_expl:
            num_expl = [d.numbers[0] for d in explanation]
            var_expl = [d.variables[0] for d in explanation]
            var_lengths = [d.shape[0] for d in var_expl]
        else:
            num_expl = []
            var_expl = []
            var_lengths = []
        return_value = {'_num_expl': num_expl, '_var_expl': var_expl, '_var_lengths': var_lengths}

        # (1-1) Read text
        return_value.update(self.encode_text_step101(text))

        # (1-2) Predict var count
        return_value.update(self.predict_varcount_step102(**return_value, dont_generate_expl=dont_generate_expl))

        # (1-3) Generate explanation
        if self.training or not dont_generate_expl:
            return_value.update(self.generate_expl_step103(text, beam=beam_expl, **return_value))

        # Separate internal outputs
        external = {}
        for key in return_value:
            if not key.startswith('_'):
                external[key] = return_value[key]

        return external, return_value

    def forward_equation(self, text: Text, equation: Equation = None, beam: int = 3,
                         _num_expl: List[Label] = None, _var_expl: List[Label] = None,
                         **kwargs) -> Tuple[dict, dict]:

        # (2-1) Concat text and explanation
        new_text = self.reconstruct_problem_step201(text, _num_expl, _var_expl)

        # (2-2) Compute explanation vector (Re-use step 1-1)
        encode_result = self.encode_text_step101(new_text.to(self.device))

        # (2-3) Read/Generate Equation
        number_len = [d.shape[0] for d in _num_expl]
        variable_len = [d.shape[0] for d in _var_expl]
        return_value = self.generate_eqn_step203(equation=equation, number_len=number_len,
                                                 variable_len=variable_len, beam=beam,
                                                 **encode_result)

        # Separate internal outputs
        external = {}
        for key in return_value:
            if not key.startswith('_'):
                external[key] = return_value[key]

        return external, return_value

    def forward(self, text: Text, **kwargs):
        # Ignore OPR_NEW_VAR_ID on training
        return_value = {'eqn_ignore': {OPR_NEW_VAR_ID}}

        """ (Phase 1) Explaining numbers/variables """
        p1_external, p1_internal = self.forward_explanation(text, **kwargs)
        return_value.update(p1_external)

        """ (Phase 2) Building solution equations """
        if self.training and self._shuffle_on_training:
            explanation = kwargs.get('explanation', None)
            random_keys = [(d, self._rng.integers(len(d.numbers)))
                           for d in explanation]
            p1_internal.update({
                '_num_expl': [d.numbers[r] for d, r in random_keys],
                '_var_expl': [d.variables[r] for d, r in random_keys]
            })

        p2_external, p2_internal = self.forward_equation(text, **p1_internal, **kwargs)
        return_value.update(p2_external)

        return return_value


class EPTX(EPTXBase):
    def __init__(self, **config):
        super().__init__(**config)
        self._recombine_policy = self._recombine_policy_default = MDL_X_R_BOTH


class EPTXOriginalOnly(EPTXBase):
    def __init__(self, **config):
        super().__init__(**config)
        self._recombine_policy = self._recombine_policy_default = MDL_X_R_COMP


class EPTXRecombinedOnly(EPTXBase):
    def __init__(self, **config):
        super().__init__(**config)
        self._recombine_policy = self._recombine_policy_default = MDL_X_R_SUFF


class EPTXPhase1Only(EPTX):
    def forward_equation(self, text: Text, equation: Equation = None, beam: int = 3,
                         _num_expl: List[Label] = None, _var_expl: List[Label] = None,
                         **kwargs) -> Tuple[dict, dict]:
        if self.training:
            # Do nothing for the equations
            return {}, {}
        else:
            # Return empty equation
            batchsz = text.shape[0]
            external = dict(equation=Equation.get_generation_base(batchsz))
            return external, {}


__all__ = ['EPTX', 'EPTXRecombinedOnly', 'EPTXOriginalOnly', 'EPTXPhase1Only']
