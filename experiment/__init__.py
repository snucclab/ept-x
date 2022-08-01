from .faithfulness import SufficiencyErasureExperiment, ComprehensivenessErasureExperiment
from .endtask import CorrectnessErrorPropagationExperiment, TreeErrorPropagationExperiment

COMP_ERASURE = 'comprehensiveness'
SUFF_ERASURE = 'sufficiency'
ERROR_CORR = 'error_propagation_corr'
ERROR_TREE = 'error_propagation_tree'

EXPERIMENT_TYPES = {
    SUFF_ERASURE: SufficiencyErasureExperiment,
    COMP_ERASURE: ComprehensivenessErasureExperiment,
    ERROR_CORR: CorrectnessErrorPropagationExperiment,
    ERROR_TREE: TreeErrorPropagationExperiment
}
