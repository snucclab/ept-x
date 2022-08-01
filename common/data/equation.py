import logging
from typing import List, Dict, Optional, Tuple, Callable, Any

import torch

from common.const.operand import *
from common.const.operator import *
from common.const.pad import PAD_ID
from common.pen.parse import parse_infix, RELATION_CLASSES
from .base import TypeTensorBatchable, TypeSelectable
from .label import Label
from .prediction import Prediction


def _operator_reader(token: int) -> str:
    return '' if token == PAD_ID else OPR_TOKENS[token]


def _operand_reader(token: int) -> str:
    if token == PAD_ID:
        return ''
    elif token < CON_END:
        return CON_TOKENS[token]
    elif token < NUM_END:
        return NUM_FORMAT % (token - NUM_BEGIN)
    else:
        return RES_FORMAT % (token - RES_BEGIN)


def _read_equation(operator: list, operands: list, shape: tuple) -> dict:
    result = []

    if len(shape) == 1:
        operator = [operator]
        operands = [[operand_j] for operand_j in operands]

    # Equation: [B, T] and A * [B, T]
    for b in range(len(operator)):
        func = operator[b]
        args = list(zip(*[operand_j[b] for operand_j in operands]))
        expr_b = []

        counter = 0
        for t, f_t in enumerate(func):
            if f_t == 0:
                counter = 0
            elif f_t == -1:
                break
            else:
                expr_b.append('%s: %s(%s)' % (RES_FORMAT % counter, _operator_reader(f_t),
                                              ', '.join([_operand_reader(a) for a in args[t]])))
                counter += 1

        result.append(' '.join(expr_b))

    if len(result) == 1:
        return dict(shape=list(shape), tokens=result[0])
    else:
        return dict(shape=list(shape), tokens=result)


def _number_label_as_map(label: Label, index: int, reverse: bool = False) -> Dict[int, int]:
    numbers = {}
    for t, nid in enumerate(label.indices[index].tolist()):
        if nid != PAD_ID:
            if not reverse and nid not in numbers:
                numbers[nid] = t - 1  # Token right before the number (where the number is generated)
            elif reverse:
                numbers[t - 1] = nid

    return numbers


class EquationPrediction(TypeSelectable):
    #: Operator prediction
    operator: Prediction
    #: List of operand predictions, [operand0, operand1, ...]. Each item is either batched or not.
    operands: List[Prediction]

    def __init__(self, operator: Prediction, operands: List[Prediction]):
        super().__init__()
        self.operator = operator
        self.operands = operands

    @classmethod
    def from_tensors(cls, operator: torch.Tensor, operands: List[torch.Tensor]) -> 'EquationPrediction':
        return EquationPrediction(operator=Prediction(operator),
                                  operands=[Prediction(operand_j) for operand_j in operands])

    def as_dict(self) -> dict:
        return dict(operator=self.operator, operands=self.operands)

    def to_human_readable(self) -> dict:
        operator = self.operator.topmost_index.tolist()
        operands = [operand_j.topmost_index.tolist()
                    for operand_j in self.operands]

        return _read_equation(operator, operands, self.operator.shape)


class Equation(TypeTensorBatchable, TypeSelectable):
    #: Operator label
    operator: Label
    #: List of operand labels, [operand0, operand1, ...]. Each item is either batched or not.
    operands: List[Label]

    def __init__(self, operator: Label, operands: List[Label]):
        super().__init__()
        assert all(operator.shape == operand_j.shape for operand_j in operands)
        assert len(operands) == OPR_MAX_ARITY
        self.operator = operator
        self.operands = operands

    def __repr__(self) -> str:
        return f'Equation(operator={self.operator}, operands={self.operands})'

    @property
    def shape(self) -> torch.Size:
        return self.operator.shape

    @property
    def pad(self) -> torch.BoolTensor:
        return self.operator.pad

    @property
    def device(self) -> torch.device:
        return self.operator.device

    @property
    def sequence_lengths(self) -> torch.LongTensor:
        return self.operator.sequence_lengths

    @classmethod
    def from_dict(cls, raw: dict, var_list_out: List[str]) -> 'Equation':
        # Sort equation by number occurrences
        equations = []

        # Collect all variables in the order of occurrence
        for eqn in raw['equations']:
            for tok in eqn.split():
                if tok.startswith(NUM_PREFIX) or not tok[0].isalpha():
                    continue
                if tok not in var_list_out:
                    var_list_out.append(tok)
                    equations.append((OPR_NEW_VAR_ID,) + (PAD_ID,) * OPR_MAX_ARITY)

        # Parse equation
        for eqn in raw['equations']:
            equations += parse_infix(eqn, var_list_out, offset=len(equations))

        # Prepend NEW_EQN() and Append DONE()
        # Note: NEW_EQN() is inserted here because it should be ignored when numbering results.
        equations.insert(0, (OPR_NEW_EQN_ID,) + (PAD_ID,) * OPR_MAX_ARITY)
        equations.append((OPR_DONE_ID,) + (PAD_ID,) * OPR_MAX_ARITY)

        # Separate operator and operands
        operator, *operands = zip(*equations)

        return Equation(operator=Label.from_list(operator),
                        operands=[Label.from_list(operand_j) for operand_j in operands])

    @classmethod
    def build_batch(cls, *items: 'Equation') -> 'Equation':
        operands = zip(*[item.operands for item in items])
        return Equation(operator=Label.build_batch(*[item.operator for item in items]),
                        operands=[Label.build_batch(*operand_j) for operand_j in operands])

    @classmethod
    def concat(cls, *items: 'Equation', dim: int = 0) -> 'Equation':
        operands = zip(*[item.operands for item in items])
        return Equation(operator=Label.concat(*[item.operator for item in items], dim=dim),
                        operands=[Label.concat(*operand_j, dim=dim) for operand_j in operands])

    @classmethod
    def from_tensors(cls, operator: torch.LongTensor, operands: List[torch.LongTensor]) -> 'Equation':
        return Equation(operator=Label(operator),
                        operands=[Label(operand_j) for operand_j in operands])

    @classmethod
    def get_generation_base(cls, batchsz: int = 1) -> 'Equation':
        operator = Label(torch.LongTensor([[OPR_NEW_EQN_ID] for _ in range(batchsz)]))  # [M=1, T=1]
        operands = [Label(torch.LongTensor([[PAD_ID] for _ in range(batchsz)]))  # [M=1, T=1]
                    for _ in range(OPR_MAX_ARITY)]
        return Equation(operator, operands)

    def as_dict(self) -> dict:
        return dict(operator=self.operator, operands=self.operands)

    def smoothed_cross_entropy(self, pred: EquationPrediction, smoothing: float = 0.1,
                               shift_target: bool = True) -> Dict[str, torch.Tensor]:
        assert len(pred.operands) == len(self.operands)
        return dict(
            operator=self.operator.smoothed_cross_entropy(pred.operator, smoothing, shift_target),
            **{'operand%s' % j: gold_j.smoothed_cross_entropy(pred_j, smoothing, shift_target)
               for j, (gold_j, pred_j) in enumerate(zip(self.operands, pred.operands))}
        )

    def accuracy_of(self, pred: EquationPrediction) -> Dict[str, float]:
        assert len(pred.operands) == len(self.operands)

        operator_acc = self.operator.accuracy_of(pred.operator)
        operands_acc = [gold_j.accuracy_of(pred_j)
                        for gold_j, pred_j in zip(self.operands, pred.operands)]

        all_seq = torch.stack([operator_acc.pop('seq')] + [acc_j.pop('seq') for acc_j in operands_acc]).prod(dim=0)
        all_seq = float(all_seq.prod(dim=1).float().mean())

        return dict(
            seq_acc_all=all_seq,
            **{key + '_operator': value
               for key, value in operator_acc.items()},
            **{key + '_operand%s' % j: value
               for j, acc_j in enumerate(operands_acc)
               for key, value in acc_j.items()}
        )

    def extends_to(self, next_operator: torch.LongTensor, next_operands: List[torch.LongTensor]) -> 'Equation':
        return Equation(operator=self.operator.extends_to(next_operator),
                        operands=[prev_j.extends_to(next_j)
                                  for prev_j, next_j in zip(self.operands, next_operands)])

    def unsqueeze(self, dim: int) -> 'Equation':
        return Equation(operator=self.operator.unsqueeze(dim),
                        operands=[operand_j.unsqueeze(dim)
                                  for operand_j in self.operands])

    def to_human_readable(self, **kwargs) -> dict:
        operator = self.operator.indices.tolist()
        operands = [operand_j.indices.tolist()
                    for operand_j in self.operands]

        return _read_equation(operator, operands, self.operator.shape)

    def _build_forest(self, var_list: List[str], variable_fn, operator_fn) -> list:
        assert not self.is_batched

        func: List[int] = self.operator.indices.tolist()
        args: List[Tuple[int, ...]] = list(zip(*[operand_j.indices.tolist() for operand_j in self.operands]))

        workspace = []
        var_index = 0

        for f, a in zip(func, args):
            if f == OPR_NEW_EQN_ID:
                workspace.clear()
                continue
            if f == OPR_DONE_ID:
                break

            if f == OPR_NEW_VAR_ID:
                if len(var_list) > var_index:
                    var_name = var_list[var_index]
                else:
                    var_name = VAR_FORMAT % var_index

                workspace.append(variable_fn(var_name))
                var_index += 1
            else:
                arity = OPR_VALUES[f][KEY_ARITY]
                operands = a[:arity]
                if len(operands) < arity:
                    missed = arity - len(operands)
                    logging.warning('Formula has %s missing argument(s): %s%s' % (missed, f, repr(tuple(a))))
                    # Append 'PAD' for empty spaces
                    operands += [PAD_ID] * missed

                workspace.append(operator_fn(f, operands, workspace))

        return workspace

    def to_sympy(self, var_list: List[str]) -> List[sympy.Expr]:
        def _variable_fn(var_name):
            return sympy.Symbol(var_name, real=True)

        def _operand_to_sympy(token: int, workspace: list) -> Optional[sympy.Expr]:
            if token == PAD_ID:
                return None
            elif token < CON_END:
                return sympy.Number(CON_TOKENS[token])
            elif token < NUM_END:
                return sympy.Symbol(NUM_FORMAT % (token - NUM_BEGIN), real=True)
            else:
                rid = token - RES_BEGIN
                return workspace[rid] if rid < len(workspace) else None

        def _operator_fn(opr_id, operands, workspace):
            info = OPR_VALUES[opr_id]
            operator = info[KEY_CONVERT]
            operands = [_operand_to_sympy(a_j, workspace)
                        for a_j in operands]
            return operator(*operands)

        try:
            return [expr
                    for expr in self._build_forest(var_list, variable_fn=_variable_fn, operator_fn=_operator_fn)
                    if isinstance(expr, RELATION_CLASSES)]
        except Exception as e:
            logging.warning('We ignored the following issue on converting equations, and returned [].', exc_info=e)
            return []

    def to_tree_dict(self, var_list: List[str]) -> dict:
        def _tree_dict(typename: str, name: Any = None, children: List = None, commutative: bool = False):
            if commutative and children is not None:
                children = sorted(children, key=lambda t: t['repr'])

            if children is None:
                representation = str(name)
            else:
                representation = name + '(' + ','.join([t['repr'] for t in children]) + ')'

            return {
                'type': typename,
                'name': name,
                'children': children if children is not None else [],
                'repr': representation
            }

        def _variable_fn(var_name):
            return _tree_dict(typename='var', name=var_name)

        def _operand_to_dict(token: int, workspace: list) -> dict:
            if token == PAD_ID:
                return _tree_dict(typename='none')
            elif token < CON_END:
                return _tree_dict(typename='const', name=CON_TOKENS[token])
            elif token < NUM_END:
                return _tree_dict(typename='num', name=NUM_FORMAT % (token - NUM_BEGIN))
            else:
                rid = token - RES_BEGIN
                return workspace[rid] if rid < len(workspace) else _tree_dict(typename='none')

        def _operator_fn(opr_id, operands, workspace):
            name = OPR_TOKENS[opr_id]
            commutative = OPR_VALUES[opr_id][KEY_COMMUTATIVE]
            operands = [_operand_to_dict(a_j, workspace)
                        for a_j in operands]
            return _tree_dict(typename='expression', name=name, children=operands, commutative=commutative)

        try:
            children = [expr
                        for expr in self._build_forest(var_list, variable_fn=_variable_fn, operator_fn=_operator_fn)
                        if expr['type'] == 'expression' and OPR_TOKENS.index(expr['name']) in OPR_TOP_LV]
            return _tree_dict(typename='system', name='system', children=children, commutative=True)
        except Exception as e:
            logging.warning('We ignored the following issue on converting equations, and returned [].', exc_info=e)
            return {}

    def ignore_labels(self, eqn_excluded: set) -> 'Equation':
        return Equation(operator=self.operator.ignore_labels(eqn_excluded),
                        operands=self.operands)

    def transform(self, fn: Callable[[int, List[int]], List[Tuple[int, List[int]]]]) -> 'Equation':
        # We expect calling this method for each item
        assert not self.is_batched

        operator = self.operator.indices.tolist()  # [T]
        operands = list(zip(*[operand.indices.tolist() for operand in self.operands]))  # [T, A]

        new_operator = []
        new_operands = [[] for _ in range(len(self.operands))]
        for t in range(len(operator)):
            for f_t, a_t in fn(operator[t], list(operands[t])):
                new_operator.append(f_t)
                for j in range(OPR_MAX_ARITY):
                    new_operands[j].append(a_t[j])

        return Equation(Label.from_list(new_operator),
                        [Label.from_list(operand_j) for operand_j in new_operands])

    def treat_variables_as_defined(self, number_max: list) -> 'Equation':
        # We expect calling this method after construct a batch.
        assert self.is_batched

        batch_sz = self.shape[0]
        new_batch = []
        for b in range(batch_sz):
            var_map = {}

            def function(operator: int, operands: List[int]) -> List[Tuple[int, List[int]]]:
                if operator == OPR_NEW_VAR_ID:
                    var_map[RES_BEGIN + len(var_map)] = NUM_BEGIN + number_max[b] + len(var_map)
                    return []
                if operator == PAD_ID:
                    return []

                return [(operator, [var_map.get(op_j, op_j - len(var_map) if op_j >= RES_BEGIN else op_j)
                                    for op_j in operands])]

            new_batch.append(self[b].transform(function))

        return Equation.build_batch(*new_batch).to(self.device)

    def treat_text_as_prev_result(self, number_label: Label):
        # We expect calling this method after construct a batch.
        assert self.is_batched

        batch_sz = self.shape[0]
        tid_max = number_label.shape[1]
        res_offset = RES_BEGIN - CON_END

        new_batch = []
        for b in range(batch_sz):
            nid_to_tid = _number_label_as_map(number_label, b)

            def function(operator: int, operands: List[int]) -> List[Tuple[int, List[int]]]:
                if operator == PAD_ID:
                    return []

                return [(operator, [nid_to_tid[op_j - NUM_BEGIN] + CON_END if NUM_BEGIN <= op_j < NUM_END
                                    else (op_j - res_offset + tid_max if op_j >= RES_BEGIN else op_j)
                                    for op_j in operands])]

            new_batch.append(self[b].transform(function))

        return Equation.build_batch(*new_batch).to(self.device)

    def restore_variables(self, number_max: list, variable_max: list) -> 'Equation':
        # We expect calling this method after construct a batch.
        assert self.is_batched

        batch_sz = self.shape[0]
        new_batch = []
        for b in range(batch_sz):
            num_end = NUM_BEGIN + number_max[b]
            var_len = variable_max[b]

            def function(operator: int, operands: List[int]) -> List[Tuple[int, List[int]]]:
                if operator == PAD_ID:
                    return []

                new_operands = [op_j - num_end + RES_BEGIN if num_end <= op_j < RES_BEGIN
                                else (op_j + var_len if op_j >= RES_BEGIN else op_j)
                                for op_j in operands]

                suffix = []
                if operator == OPR_NEW_EQN_ID:
                    # Append variables
                    suffix = [(OPR_NEW_VAR_ID, [PAD_ID] * OPR_MAX_ARITY)] * var_len

                return [(operator, new_operands)] + suffix

            new_batch.append(self[b].transform(function))

        return Equation.build_batch(*new_batch).to(self.device)

    def restore_numbers(self, number_label: Label):
        # We expect calling this method after construct a batch.
        assert self.is_batched

        batch_sz = self.shape[0]
        tid_max = number_label.shape[1]
        res_offset = RES_BEGIN - CON_END

        new_batch = []
        for b in range(batch_sz):
            tid_to_nid = _number_label_as_map(number_label, b, reverse=True)

            def function(operator: int, operands: List[int]) -> List[Tuple[int, List[int]]]:
                if operator == PAD_ID:
                    return []

                return [(operator, [tid_to_nid[op_j - CON_END] + NUM_BEGIN if CON_END <= op_j < CON_END + tid_max
                                    else (op_j - tid_max + res_offset if op_j >= CON_END + tid_max else op_j)
                                    for op_j in operands])]

            new_batch.append(self[b].transform(function))

        return Equation.build_batch(*new_batch).to(self.device)
