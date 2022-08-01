from collections import defaultdict
from typing import List, Dict

from common.const.operand import NUM_BEGIN, RES_BEGIN, NUM_FORMAT, VAR_FORMAT
from common.const.pad import PAD_ID, UNEXPLAINED_NUMBER
from .base import *
from .label import Label
from .prediction import Prediction


class Explanation(TypeBatchable):
    #: Tokenized explanations for numbers. Each item corresponds to a worker. Shape: (List of) [N, D].
    numbers: List[Label]
    #: Tokenized explanations for variables. Each item corresponds to a worker. Shape: (List of) [V, D].
    variables: List[Label]
    #: Worker for training experiments
    worker: int = 0
    workers: List[str] = []

    def __init__(self, numbers: List[Label], variables: List[Label], worker: int, workers: list):
        super().__init__()
        assert len(numbers) == len(variables)
        self.numbers = numbers
        self.variables = variables
        self.worker = worker
        self.workers = workers

    def __repr__(self) -> str:
        return f'Explanation(numbers=${self.numbers}, variables=${self.variables}, worker=${self.worker})'

    @property
    def is_batched(self) -> bool:
        return False

    @property
    def number_for_train(self) -> Label:
        return self.numbers[self.worker]

    @property
    def variable_for_train(self) -> Label:
        return self.variables[self.worker]

    @property
    def device(self) -> torch.device:
        return self.number_for_train.device

    @classmethod
    def build_batch(cls, *items: 'Explanation') -> 'Explanation':
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, raw: dict, n_numbers: int, var_list: List[str], tokenizer, field: str = 'explanations') -> 'Explanation':
        explanations = defaultdict(list)
        workers = []
        numbers = []
        variables = []
        SEP = [tokenizer.sep_token_id]

        for worker, work in raw[field].items():
            num_label = []
            var_label = []

            for nid in range(n_numbers):
                key = NUM_FORMAT % nid
                expl = work.get(key, '')
                if expl == '':
                    expl = UNEXPLAINED_NUMBER
                else:
                    explanations[nid + NUM_BEGIN].append(expl)

                # Form: "[SEP] explanation.... [SEP]" (No [CLS] at the beginning)
                num_label.append(SEP + tokenizer.encode(expl, add_special_tokens=False) + SEP)

            for vid, key in enumerate(var_list):
                expl = work.get(key, '')
                assert expl != ''

                # Form: "[SEP] explanation.... [SEP]"
                var_label.append(SEP + tokenizer.encode(expl, add_special_tokens=False) + SEP)
                explanations[vid + RES_BEGIN].append(expl)

            numbers.append(Label.from_list(num_label))
            variables.append(Label.from_list(var_label))
            workers.append(worker)

        assert all(len(workers) == len(num_worked) for num_worked in explanations.values()), raw['_id']
        selected_worker = raw.get('worker_for_train', None)
        worker_index = workers.index(selected_worker) if selected_worker in workers else -1
        return Explanation(numbers=numbers, variables=variables, worker=worker_index, workers=workers)

    @classmethod
    def from_tensors(cls, numbers: torch.LongTensor, variables: torch.Tensor, n_numbers: int) -> 'Explanation':
        # Truncate padded positions
        numbers = Label(numbers[:n_numbers])
        variables = Label(variables)

        return Explanation([numbers], [variables], worker=0, workers=[''])

    def as_dict(self) -> dict:
        return dict(numbers=self.numbers, variables=self.variables, worker=self.worker)

    def to(self, *args, **kwargs) -> 'Explanation':
        return move_to(self, *args, **kwargs)

    def smoothed_cross_entropy(self, numbers: Prediction, variables: Prediction,
                               smoothing: float = 0.1) -> torch.Tensor:
        losses = []
        pred_concat = Prediction.concat(numbers, variables)
        for worker in range(len(self.numbers)):
            label_concat = Label.concat(self.numbers[worker], self.variables[worker])

            trim_shape = min(label_concat.shape[1], pred_concat.shape[1])
            label_concat = label_concat[:, :trim_shape]
            pred_concat_w = pred_concat[:, :trim_shape]

            losses.append(label_concat.smoothed_cross_entropy(pred_concat_w, smoothing))

        return torch.min(torch.stack(losses, dim=0))

    def accuracy_of(self, numbers: Prediction, variables: Prediction) -> Dict[str, float]:
        accuracies = []
        for worker in range(len(self.numbers)):
            labels = Label.concat(self.numbers[worker], self.variables[worker])
            items = Prediction.concat(numbers, variables)
            accuracies.append(labels.accuracy_of(items))

        return {key + '_explanation': max([acc[key] for acc in accuracies])
                for key in accuracies[0] if key.endswith('_acc')}

    def to_human_readable(self, tokenizer=None) -> Dict[str, List[str]]:
        return self.to_id_explanation_dict(tokenizer)

    def to_id_explanation_dict(self, tokenizer=None) -> Dict[str, List[str]]:
        explanations = defaultdict(list)

        for w in range(len(self.numbers)):
            for fmt, expl in [(NUM_FORMAT, self.numbers[w]),
                              (VAR_FORMAT, self.variables[w])]:
                for nid, row in enumerate(expl.pad_fill(tokenizer.pad_token_id).tolist()):
                    expl_n = tokenizer.decode(row, skip_special_tokens=True).strip()
                    if expl_n != UNEXPLAINED_NUMBER:
                        explanations[fmt % nid].append(expl_n)

        return explanations

    def extends_to(self, next_number: torch.LongTensor, next_variable: torch.LongTensor) -> 'Explanation':
        return Explanation(numbers=[num.extends_to(next_number) for num in self.numbers],
                           variables=[var.extends_to(next_variable) for var in self.variables],
                           worker=self.worker, workers=self.workers)
