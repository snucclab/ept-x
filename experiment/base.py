import abc
from typing import Dict, Any, Tuple, List, Optional

from zss import simple_distance

from common.const.operand import VAR_FORMAT, VAR_MAX
from common.data import Equation, Example
from common.dataset import Dataset
from common.pen.solve import Solver
from model import EPTX


_VAR_NAMES = [VAR_FORMAT % i for i in range(VAR_MAX)]


def _revert_transform_in_equation(eqn: Equation, changes: Optional[dict]) -> Equation:
    if changes is None or not changes:
        return eqn
    else:
        def function(operator: int, operands: List[int]) -> List[Tuple[int, List[int]]]:
            new_operands = [changes.get(operand, operand) for operand in operands]
            return [(operator, new_operands)]

        return eqn.transform(function).to(eqn.device)


def _write_pair(text, control, experiment, score, tokenizer) -> dict:
    return {
        'text': text.to_human_readable(tokenizer)['tokens'],
        'metric': float(score),
        'explanation': {
            'before': dict(control['explanation']),  # Remove defaultdict
            'after': dict(experiment[0].to_human_readable(tokenizer))  # Remove defaultdict
        },
        'equation': {
            'before': [repr(eq) for eq in control['equation'].to_sympy(_VAR_NAMES)],
            'after': [repr(eq) for eq in experiment[1].to_sympy(_VAR_NAMES)]
        }
    }


def _get_children_of_tree_dict(t: dict):
    return t['children']


def _get_label_of_tree_dict(t: dict):
    return t['name']


def _label_distance_of_identity(a, b):
    return int(a != b)


def _compute_tree_edit_distance(a, b):
    return float(simple_distance(a, b,
                                 get_children=_get_children_of_tree_dict,
                                 get_label=_get_label_of_tree_dict,
                                 label_dist=_label_distance_of_identity))


class ExperimentBase(abc.ABC):
    def __init__(self, module: EPTX, dataset: Dataset, batch_size: int = 4, **solver_args):
        self._module = module
        self._dataset = dataset
        self._batch_size = batch_size
        self._control_group_result = {}
        self._group_difference = {}
        self._sample_pairs = []
        self._checker = Solver(**solver_args)
        self._is_answer_required = False

    def __repr__(self) -> str:
        return f'{self._module.__class__.__name__}/{self.__class__.__name__}'

    def close(self):
        self._checker.close()

    def set_test_split(self, split_file: str):
        self._dataset.select_items_with_file(split_file)
        self._control_group_result.clear()
        self._group_difference.clear()
        self._sample_pairs.clear()

    def register_control_result(self, result: Dict[str, Tuple[Example, dict]]):
        assert set(result.keys()) == self._dataset.keys()
        result_transformed = {key: {field: value[1] if type(value) is tuple else value
                                    for field, value in res.items()}
                              for key, (_, res) in result.items()}

        self._control_group_result.update(result_transformed)

    def register_experiment_result(self, samples: int = 1, **kwargs):
        assert samples > 0

        self._prepare_experiment(**kwargs)
        batches = self._dataset.get_minibatches(self._batch_size, for_testing=True)
        tokenizer = self._dataset.tokenizer
        for g, batch in enumerate(batches):
            generated_samples = []
            last_sample = None
            for i in range(samples):
                # Retrieve new experimental group
                batch_dict = self._transform_to_experiment_group(batch)
                output = self._module.forward(**batch_dict)
                generated: Equation = output['equation'].to('cpu')
                generated_samples.append(generated)

                # Store last sample for error analysis
                last_sample = (batch_dict['explanation'], generated)

                if i % 10 == 9:
                    # Print progress
                    print(f'[{repr(self)}] Batch {g:3d}/{len(batches):3d} Sample {i:3d}/{samples:3d}')

            for b, item_b in enumerate(batch.info):
                item_id = item_b.item_id
                control_result = self._control_group_result[item_id]

                differences = []
                for gen in generated_samples:
                    eqn_b = gen[b]
                    exp_out = dict(equation=eqn_b)
                    if self._is_answer_required:
                        result, exception = self._checker.solve(eqn_b.to_sympy(item_b.variables), item_b.numbers)
                        correct = not exception and self._checker.check_answer(item_b.answers, result)
                        exp_out.update(correctness=correct, answer=result)

                    differences.append(self._measure_difference(control_result, exp_out))

                # Store a representative sample
                last_difference = differences[-1]
                if last_difference > 0:
                    sample_b = tuple([x[b] for x in last_sample])
                    self._sample_pairs.append(_write_pair(batch.text[b], control_result, sample_b, differences[-1],
                                                          tokenizer))

                # Add mean
                self._group_difference[item_id] = sum(differences) / len(differences)

    def get_measurements(self) -> Dict[str, float]:
        return self._group_difference

    def get_paired_samples(self) -> List[dict]:
        return self._sample_pairs

    @abc.abstractmethod
    def _prepare_experiment(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _transform_to_experiment_group(self, batch: Example) -> dict:
        raise NotImplementedError()

    @abc.abstractmethod
    def _measure_difference(self, control: Dict[str, Any], experiment: Dict[str, Any]) -> float:
        """
        measure difference between the two final output

        :param control: dictionary of execution result of an example item (in control group, not a batch)
            - 'explanation' contains generated explanations, as a single Explanation instance.
            - 'equation' contains generated equations, as a single Equation instance (not batched).
            - 'answer' contains generated answer by the 'equation's, as a dictionary.
            - 'correctness' indicates whether the 'answer' was correct or not, as a boolean value.

        :param experiment: dictionary of execution result of an example item (in experimental group)
            - 'explanation' contains generated explanations, as a single Explanation instance.
            - 'equation' contains generated equations, as a single Equation instance (not batched).
            - 'answer' contains generated answer by the 'equation's, as a dictionary.
            - 'correctness' indicates whether the 'answer' was correct or not, as a boolean value.
        """
        raise NotImplementedError()


class ExperimentWithTreeChange(ExperimentBase, abc.ABC):
    def _measure_difference(self, control: Dict[str, Any], experiment: Dict[str, Any]) -> float:
        num_vars = control['explanation_generated'].variables[0].shape[0]
        variables = [VAR_FORMAT % i for i in range(num_vars)]

        control_tree = control['equation'].to_tree_dict(variables)
        experiment_tree = experiment['equation'].to_tree_dict(variables)
        all_removed_tree = {
            'children': [],
            'name': '@'  # This name is not permitted in the original tree dict, so it ensures replacement effect
        }

        ctrl_to_expr_dist = _compute_tree_edit_distance(control_tree, experiment_tree)
        #ctrl_remove_all_dist = _compute_tree_edit_distance(control_tree, all_removed_tree)
        #normalized = min(1.0, ctrl_to_expr_dist / ctrl_remove_all_dist)

        #return normalized
        return ctrl_to_expr_dist


class ExperimentWithCorrectnessChange(ExperimentBase, abc.ABC):
    def __init__(self, module: EPTX, dataset: Dataset, batch_size: int = 4, **solver_args):
        super().__init__(module, dataset, batch_size, **solver_args)
        self._is_answer_required = True

    def _measure_difference(self, control: Dict[str, Any], experiment: Dict[str, Any]) -> float:
        control_correct = control['correctness']
        experiment_correct = experiment['correctness']

        if control_correct == experiment_correct:
            # 0 for both correct or incorrect
            return 0.0
        if control_correct:
            # -1 for control correct, experiment incorrect
            return -1.0
        else:
            # +1 for control incorrect, experiment correct
            return +1.0
