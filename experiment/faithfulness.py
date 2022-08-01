import abc
from typing import Tuple, Optional, List

from common.const.model import MDL_X_R_COMP, MDL_X_R_SUFF
from common.data import Example
from experiment.base import ExperimentWithTreeChange


class ErasureFaithfulnessExperiment(ExperimentWithTreeChange, abc.ABC):
    def _transform_to_experiment_group(self, batch: Example) -> dict:
        # No modification for the batch
        batch = batch.to(self._module.device)
        # Build-up generated explanation before, if exists
        explanation = [self._control_group_result[info.item_id].get('explanation_generated', None)
                       for info in batch.info]

        return dict(text=batch.text, explanation=explanation, dont_generate_expl=True)


class ComprehensivenessErasureExperiment(ErasureFaithfulnessExperiment):
    def _prepare_experiment(self, **kwargs):
        # Comprehensiveness: Use original text 100%
        if hasattr(self._module, 'set_recombine_policy'):
            self._module.set_recombine_policy(use_text_prob=MDL_X_R_COMP)


class SufficiencyErasureExperiment(ErasureFaithfulnessExperiment):
    def _prepare_experiment(self, **kwargs):
        # Sufficiency: Use original text 0%
        if hasattr(self._module, 'set_recombine_policy'):
            self._module.set_recombine_policy(use_text_prob=MDL_X_R_SUFF)
