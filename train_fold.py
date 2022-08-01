import logging
import pickle
from argparse import ArgumentParser
from os import cpu_count
from sys import argv

from ray import tune, init, shutdown
from ray.tune import Experiment
from ray.tune.trial import Trial
from ray.tune.utils.util import is_nan_or_inf
from torch.cuda import device_count
from shutil import rmtree

from common.const.model import *
from common.trial import trial_dirname_creator_generator
from learner import *
from model import MODELS, MODEL_CLS

CPU_FRACTION = 1.0
GPU_FRACTION = 0.5


def read_arguments():
    parser = ArgumentParser()

    env = parser.add_argument_group('Dataset & Evaluation')
    env.add_argument('--name', '-name', type=str, required=True)
    env.add_argument('--experiment-dir', '-exp', type=str, nargs='+', required=True)
    env.add_argument('--model-config', '-model', type=str, nargs='+', required=True)

    log = parser.add_argument_group('Logger setup')
    log.add_argument('--log-path', '-log', type=str, default='./runs')

    return parser.parse_args()


def build_experiment_config(max_iter, exp_dir):
    exp_path = Path(exp_dir)
    experiments = {}
    for file in exp_path.glob('*'):
        if not file.is_file():
            continue

        experiment_dict = {KEY_SPLIT_FILE: str(file.absolute())}
        if file.name != KEY_TRAIN:
            experiment_dict[KEY_EVAL_PERIOD] = max_iter

        experiments[file.name] = experiment_dict

    return experiments


def build_configuration(args):
    configurations = []
    for model in args.model_config:
        with Path(model).open('rb') as fp:
            base_config = pickle.load(fp)

        max_iter = base_config[KEY_SCHEDULER]['num_total_epochs']
        for exp_dir in args.experiment_dir:
            exp_conf = base_config.copy()
            exp_conf[KEY_EXPERIMENT] = build_experiment_config(max_iter, exp_dir)
            configurations.append(exp_conf)

    return configurations


def get_experiment_name(args, dataset):
    from datetime import datetime
    now = datetime.now().strftime('%m%d%H%M%S')
    return f'{Path(dataset).stem}_{args.name}_{now}'


def summary(reports) -> Dict[str, dict]:
    result = {}
    for key in reports[0]:
        values = [r[key] for r in reports]
        result[key] = {
            'values': values,
            'mean': float(mean(values)),
            'stderr': float(std(values, ddof=1)) / (len(values) ** 0.5)
        }

    return result


if __name__ == '__main__':
    args = read_arguments()
    if not Path(args.log_path).exists():
        Path(args.log_path).mkdir(parents=True)

    # Enable logging system
    file_handler = logging.FileHandler(filename=Path(args.log_path, 'meta.log'), encoding='UTF-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%m/%d %H:%M:%S'))
    file_handler.setLevel(logging.INFO)

    logger = logging.getLogger('Cross Validation')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info('========================= CMD ARGUMENT =============================')
    logger.info(' '.join(argv))
    init(num_cpus=cpu_count(), num_gpus=device_count())

    configurations = build_configuration(args)
    experiment_name = get_experiment_name(args, configurations[0][KEY_DATASET])
    max_iter = configurations[0][KEY_SCHEDULER]['num_total_epochs']
    stop_condition = dict(training_iteration=max_iter)

    experiments = []
    for conf in configurations:
        experiments.append(Experiment(experiment_name, SupervisedTrainer, stop=stop_condition,
                                      config=conf, local_dir=args.log_path, checkpoint_at_end=True, checkpoint_freq=max_iter,
                                      trial_dirname_creator=trial_dirname_creator_generator()))

    trials: List[Trial] = tune.run_experiments(experiments, reuse_actors=True, raise_on_failed_trial=False)

    # Record trial information
    logger.info('========================= RESULTS =============================')
    logger.info('Hyperparameter search is finished!')
    model_scores = defaultdict(list)

    # Read result of the fold
    for trial in trials:
        if trial.status != Trial.TERMINATED:
            logger.info('\tTrial %10s (%-40s): FAILED', trial.trial_id, trial.experiment_tag)
            continue

        model = trial.config[KEY_MODEL][MODEL_CLS]
        fold = Path(trial.config[KEY_EXPERIMENT][KEY_TEST][KEY_SPLIT_FILE]).parent.stem
        scores = trial.last_result[KEY_TEST]
        logger.info('\t%10s %20s: %s', model, fold, scores)
        model_scores[model].append(scores)

    # Record the best configuration
    scores_dumped = {}
    for cls, scores in model_scores.items():
        logger.info('--------------------------------------------------------')
        logger.info('Summary for %s', cls)
        summarized = summary(scores)
        logger.info(yaml_dump(summarized))
        logger.info('--------------------------------------------------------')

        scores_dumped[cls] = summarized

    # Save configuration as pickle and yaml
    outpath = Path(trials[0].logdir).parent
    with (outpath / 'cross-validation-summary.pkl').open('wb') as fp:
        pickle.dump(scores_dumped, fp)
    with (outpath / 'cross-validation-summary.yaml').open('w+t') as fp:
        yaml_dump(scores_dumped, fp, allow_unicode=True, default_style='|')

    shutdown()
