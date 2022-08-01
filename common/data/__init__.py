from .base import *
from .explanation import *
from .encoded import *
from .equation import *
from .label import *
from .prediction import *
from .text import *


def _compute_accuracy_from_list(items: list, key: str = '') -> dict:
    values = {}
    for tgt in {'token', 'seq'}:
        for field in {'corrects', 'total'}:
            values[tgt + field] = sum([item[tgt][field] for item in items])

    return {
        f'token_acc_{key}': values['tokencorrects'] / values['tokentotal'] if values['tokentotal'] else float('NaN'),
        f'seq_acc_{key}': values['seqcorrects'] / values['seqtotal'] if values['seqtotal'] else float('NaN')
    }


class ExtraInfo(TypeBase):
    item_id: str
    answers: List[Dict[str, sympy.Number]]
    numbers: Dict[str, sympy.Number]
    variables: List[str]
    split: Optional[str]
    raw: Optional[dict]

    def __init__(self, item_id: str, answers: List[Dict[str, sympy.Number]], numbers: Dict[str, sympy.Number],
                 variables: List[str], split: str = None, raw: dict = None):
        super().__init__()
        self.item_id = item_id
        self.answers = answers
        self.numbers = numbers
        self.variables = variables
        self.split = split
        self.raw = raw

    @classmethod
    def from_dict(cls, raw: dict) -> 'ExtraInfo':
        answers = [{key: sympy.Number(var)
                    for key, var in item.items() if not key.startswith('_')}
                   for item in raw['answers'] if item['_selected']]
        numbers = {num['key']: sympy.Number(num['value'])
                   for num in raw['numbers']}
        return ExtraInfo(item_id=raw['_id'], split=None, answers=answers, numbers=numbers, variables=[], raw=raw)

    def filter_answers(self) -> 'ExtraInfo':
        kwargs = self.as_dict()
        kwargs['answers'] = [{key: value
                              for key, value in answer_tuple.items()
                              if key in kwargs['variables']}
                             for answer_tuple in kwargs['answers']]
        return ExtraInfo(**kwargs)

    def as_dict(self) -> dict:
        return dict(item_id=self.item_id, answers=self.answers, numbers=self.numbers, variables=self.variables,
                    split=self.split, raw=self.raw)

    def to_human_readable(self) -> dict:
        result = self.as_dict()
        result.pop('raw')
        return human_readable_form(result)


class Example(TypeBatchable):
    text: Text
    equation: Equation
    explanation: Union[Explanation, List[Explanation]]
    info: Union[ExtraInfo, List[ExtraInfo]]

    def __init__(self, text: Text, equation: Equation, explanation: Union[Explanation, List[Explanation]],
                 info: Union[ExtraInfo, List[ExtraInfo]]):
        super().__init__()
        self.text = text
        self.equation = equation
        self.explanation = explanation
        self.info = info

    @property
    def device(self) -> torch.device:
        return self.text.device

    @property
    def is_batched(self) -> bool:
        return self.text.is_batched

    @property
    def batch_size(self):
        return self.text.shape[0] if self.is_batched else 1

    @classmethod
    def build_batch(cls, *items: 'Example') -> 'Example':
        return Example(text=Text.build_batch(*[item.text for item in items]),
                       equation=Equation.build_batch(*[item.equation for item in items]),
                       explanation=[item.explanation for item in items],
                       info=[item.info for item in items])

    @classmethod
    def concat(cls, *items: 'Example') -> 'Example':
        raise NotImplementedError('This operation is not supported')

    @classmethod
    def from_dict(cls, raw: dict, tokenizer, number_window: int = 5) -> 'Example':
        _info = ExtraInfo.from_dict(raw)
        _text = Text.from_dict(raw, tokenizer=tokenizer, number_window=number_window)
        _equation = Equation.from_dict(raw, var_list_out=_info.variables)
        _explanation = Explanation.from_dict(raw, n_numbers=len(_info.numbers),
                                             var_list=_info.variables, tokenizer=tokenizer)

        # Filter out not-used variables from the answers
        _info = _info.filter_answers()

        return Example(text=_text, equation=_equation, explanation=_explanation, info=_info)

    def as_dict(self) -> dict:
        return dict(text=self.text, equation=self.equation, explanation=self.explanation, info=self.info)

    def get_item_size(self) -> int:
        return max(self.text.shape[-1], self.equation.shape[-1], self.explanation.number_for_train.shape[-1],
                   self.explanation.variable_for_train.shape[-1])

    def item_of_batch(self, index: int) -> 'Example':
        assert self.is_batched
        return Example(text=self.text[index],
                       equation=self.equation[index],
                       explanation=self.explanation[index],
                       info=self.info[index])

    def to_human_readable(self, tokenizer=None) -> dict:
        if self.is_batched:
            return dict(
                info=[i.to_human_readable() for i in self.info],
                text=self.text.to_human_readable(tokenizer),
                equation=self.equation.to_human_readable(),
                explanation=[d.to_human_readable(tokenizer) for d in self.explanation]
            )
        else:
            return dict(
                info=self.info.to_human_readable(),
                text=self.text.to_human_readable(tokenizer),
                equation=self.equation.to_human_readable(),
                explanation=self.explanation.to_human_readable(tokenizer)
            )

    def accuracy_of(self, **kwargs) -> dict:
        # equation: EquationPrediction [B, T]
        # num_expl?: B-List of Prediction [N, D]
        # var_expl?: B-List of Prediction [V, D] or Prediction [B, VD]
        # var_target?: Label [B, VD]
        result = {}
        if 'equation' in kwargs:
            if 'equation_tgt' in kwargs:
                eqn_tgt = kwargs.pop('equation_tgt')
            elif 'eqn_ignore' in kwargs:
                eqn_tgt = self.equation.ignore_labels(kwargs.pop('eqn_ignore'))
            else:
                eqn_tgt = self.equation
            result.update(eqn_tgt.accuracy_of(kwargs.pop('equation')))

        if 'num_expl' in kwargs:
            num_expl_cnt = [gold.number_for_train.num_corrects(pred)
                            for gold, pred in zip(self.explanation, kwargs.pop('num_expl'))]
            result.update(_compute_accuracy_from_list(num_expl_cnt, key='num'))

        if 'var_expl' in kwargs:
            var_expl_cnt = [gold.variable_for_train.num_corrects(pred)
                            for gold, pred in zip(self.explanation, kwargs.pop('var_expl'))]
            result.update(_compute_accuracy_from_list(var_expl_cnt, key='var'))

        if 'var_len' in kwargs:
            var_len: torch.Tensor = kwargs.pop('var_len_tgt').indices.float()
            var_len_pred: torch.Tensor = kwargs.pop('var_len').log_prob.argmax(dim=-1).float()
            result['diff_var_len'] = (var_len_pred - var_len).mean().item()

        return result

    def smoothed_cross_entropy(self, **kwargs) -> Dict[str, torch.Tensor]:
        # equation: EquationPrediction [B, T]
        # num_expl?: B-List of Prediction [N, D]
        # var_expl?: B-List of Prediction [V, D] or Prediction [B, VD]
        # var_target?: Label [B, VD]
        result = {}
        if 'equation' in kwargs:
            if 'equation_tgt' in kwargs:
                eqn_tgt = kwargs.pop('equation_tgt')
            elif 'eqn_ignore' in kwargs:
                eqn_tgt = self.equation.ignore_labels(kwargs.pop('eqn_ignore'))
            else:
                eqn_tgt = self.equation
            result.update(eqn_tgt.smoothed_cross_entropy(kwargs.pop('equation'), smoothing=0.01))

        if 'num_expl' in kwargs and 'var_expl' in kwargs:
            num_loss = [gold.number_for_train.smoothed_cross_entropy(pred, smoothing=0.01)
                        for gold, pred in zip(self.explanation, kwargs.pop('num_expl'))]
            var_loss = [gold.variable_for_train.smoothed_cross_entropy(pred, smoothing=0.01)
                        for gold, pred in zip(self.explanation, kwargs.pop('var_expl'))]

            batch_sz = len(var_loss)
            losses = torch.stack(num_loss + var_loss)
            result['expl'] = sum(losses) / batch_sz

        if 'var_len' in kwargs:
            var_len: Label = kwargs.pop('var_len_tgt')
            var_len_pred: Prediction = kwargs.pop('var_len')
            result['var_len'] = var_len.smoothed_cross_entropy(var_len_pred, smoothing=0.01, shift_target=False)

        return result


__all__ = ['Example', 'Text', 'Equation', 'EquationPrediction', 'Explanation',
           'ExtraInfo', 'Encoded', 'Label', 'Prediction']
