import itertools
import os
from argparse import ArgumentParser
from os import cpu_count
from typing import Tuple

import ray
import tqdm
from torch.cuda import device_count

from common.pen.bootstrap import get_metric_summary, make_resamples, Metric, get_confidence_interval
from experiment import *
from learner import *
from model import model_load


def read_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model', '-model', type=str, required=True, nargs='+')
    parser.add_argument('--dataset', '-data', type=str, required=True)
    parser.add_argument('--experiment-dir', '-exp', type=str, required=True)
    parser.add_argument('--seed', '-seed', type=int, default=1)
    parser.add_argument('--faithfulness', '-faith', type=str, nargs='+', choices=EXPERIMENT_TYPES.keys(),
                        default=list(EXPERIMENT_TYPES.keys()))
    parser.add_argument('--bootstrap-trials', '-ntr', type=int, default=1000)
    parser.add_argument('--sample-size', '-smp', type=int, default=100)
    parser.add_argument('--repeating-counts', '-rep', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--digressing-counts', '-dig', type=int, nargs='+', default=[1])
    parser.add_argument('--perturbation-samples', '-ptr', type=int, default=100)
    parser.add_argument('--num-cpu', '-cpu', type=float, default=1)
    parser.add_argument('--num-gpu', '-gpu', type=float, default=1)

    return parser.parse_args()


def load_config(path):
    with Path(path, 'config.pkl').open('rb') as fp:
        conf = pickle.load(fp)
    
    return conf


def run_model_once(model_path, **exp_dict):
    set_seed(exp_dict[KEY_SEED])
    outdir = Path(model_path)
    pretrained = model_load(model_path)
    if device_count() > 0:
        pretrained.to('cuda')

    config = load_config(model_path)
    dataset = Dataset(exp_dict[KEY_DATASET], langmodel=config[KEY_MODEL][MDL_ENCODER], seed=exp_dict[KEY_SEED],
                      number_window=config[KEY_WINDOW])
    tester = Tester()

    with torch.no_grad():
        for experiment in exp_dict[KEY_EXPERIMENT]:
            pkl_out = (outdir / (experiment.stem + '.p'))
            yaml_out = (outdir / (experiment.stem + '-summary.yaml'))
            finished = (outdir / (experiment.stem + '.f'))

            if finished.exists():
                continue

            dataset.select_items_with_file(str(experiment))
            output_pairs = []
            results = {}
            for batch in dataset.get_minibatches(config[KEY_BATCH_SZ], for_testing=True):
                output = pretrained.forward(text=batch.text.to(pretrained.device),
                                            beam=config[KEY_BEAM], beam_expl=config[KEY_BEAM_DESC])

                # Un-batch output
                equation = output['equation']
                explanation = output.get('explanation', None)
                for b in range(batch.batch_size):
                    item = batch.item_of_batch(b)
                    pairs = dict(equation=(item.equation, equation[b]))

                    if 'explanation' in output:
                        pairs['explanation'] = (item.explanation.to_id_explanation_dict(dataset.tokenizer),
                                                explanation[b].to_id_explanation_dict(dataset.tokenizer))
                        pairs['explanation_generated'] = explanation[b]

                    output_pairs.append((item, pairs))

                    item_id = item.info.item_id
                    results[item_id] = (item, pairs)

            test_result = tester.check(output_pairs, tokenizer=dataset.tokenizer)
            with pkl_out.open('wb') as fp:
                for dump_k in test_result.pop('dump'):
                    key = dump_k['info']['item_id']
                    results[key][1].update({
                        'correctness': dump_k['correct'],
                        'answer': dump_k['answer_generated']
                    })

                pickle.dump(results, fp)

            with yaml_out.open('a+t', encoding='UTF-8') as fp:
                yaml_dump(test_result,
                          fp, allow_unicode=True)
                fp.write('\n')

            finished.touch()

    del pretrained
    del dataset
    tester.close()


def run_faithfulness(model_path, exp_header, faith_args=None, **exp_dict):
    set_seed(exp_dict[KEY_SEED])

    pretrained = model_load(model_path)
    if device_count() > 0:
        pretrained.to('cuda')

    if faith_args is None:
        faith_args = [dict()]

    config = load_config(model_path)
    dataset = Dataset(exp_dict[KEY_DATASET], langmodel=config[KEY_MODEL][MDL_ENCODER], seed=exp_dict[KEY_SEED],
                      number_window=config[KEY_WINDOW])
    exp_cls = EXPERIMENT_TYPES[exp_header](pretrained, dataset, config[KEY_BATCH_SZ])
    outdir = Path(model_path)

    with torch.no_grad():
        for experiment in exp_dict[KEY_EXPERIMENT]:
            exp_cls.set_test_split(str(experiment))
            with (outdir / (experiment.stem + '.p')).open('rb') as fp:
                exp_cls.register_control_result(pickle.load(fp))

            for arg in faith_args:
                arg_string = [exp_header] + ['%s_%s' % (k[0], v) for k, v in arg.items()]
                faith_exp = '-'.join(arg_string)
                pkl_out = (outdir / (experiment.stem + '-%s.p' % faith_exp))
                yaml_out = (outdir / (experiment.stem + '-summary.yaml'))
                sample_out = (outdir / (experiment.stem + '-%s-sample.yaml' % faith_exp))
                finished = (outdir / (experiment.stem + '-%s.f' % faith_exp))

                if not experiment.is_file() or experiment.stem == KEY_TRAIN or finished.exists():
                    continue

                exp_cls.register_experiment_result(**arg)
                measures = exp_cls.get_measurements()

                with pkl_out.open('wb') as fp:
                    pickle.dump(measures, fp)

                with yaml_out.open('a+t', encoding='UTF-8') as fp:
                    yaml_dump({faith_exp: get_metric_summary(list(measures.values()))._asdict()},
                              fp, allow_unicode=True)
                    fp.write('\n')

                with sample_out.open('w+t', encoding='UTF-8') as fp:
                    yaml_dump(exp_cls.get_paired_samples(), fp, allow_unicode=True)
                    fp.write('\n')

                finished.touch()

    exp_cls.close()
    del pretrained


def compute_resample(model_path: str, split: str, keys: List[str],
                     resample_index: int) -> List[Tuple[str, str, int, Metric]]:
    outdir = Path(model_path)
    model_name = outdir.stem
    output = []
    tester = Tester()

    for file in outdir.glob('%s*.p' % split):
        with file.open('rb') as fp:
            result = pickle.load(fp)
            sample = [result[key] for key in keys if key in result]

        if type(sample[0]) is tuple:
            # Example-output pair
            for metric, value in tester.check(sample).items():
                output.append((model_name, file.stem + '-' + metric, resample_index,
                               Metric(value, None, len(sample), None)))
        else:
            # A single value
            sample = [float(val) for val in sample]
            output.append((model_name, file.stem, resample_index, get_metric_summary(sample)))

    tester.close()
    return output


if __name__ == '__main__':
    DEBUG = 'DEBUG' in os.environ
    args = read_arguments()
    set_seed(args.seed)

    models = [str(Path(m).absolute())
              for m in args.model]
    experiments = [f.absolute()
                   for f in Path(args.experiment_dir).glob('*')
                   if f.is_file() and f.stem != KEY_TRAIN]
    exp_base = {
        KEY_DATASET: str(Path(args.dataset).absolute()),
        KEY_EXPERIMENT: experiments,
        KEY_SEED: args.seed
    }

    # Prepare experimental results before bootstrap sampling, since bootstrapping is independent to model's computation.
    if DEBUG:
        for model in tqdm.tqdm(models):
            run_model_once(model, **exp_base)
            for faith in tqdm.tqdm(args.faithfulness):
                if 'EPT' not in model:
                    run_faithfulness(model, faith, **exp_base)
    else:
        num_devices = device_count()
        if args.num_gpu > 0 and num_devices > 0:
            ray.init(num_cpus=min(cpu_count(), int(num_devices / args.num_gpu) + 1), num_gpus=device_count())
        else:
            ray.init(num_cpus=cpu_count())

        run_model_ray = ray.remote(run_model_once).options(num_gpus=args.num_gpu, num_cpus=args.num_cpu)
        run_faith_ray = ray.remote(run_faithfulness).options(num_gpus=args.num_gpu, num_cpus=args.num_cpu)

        ray.get([run_model_ray.remote(model, **exp_base)
                 for model in models])
        ray.get([run_faith_ray.remote(model, faith, **exp_base)
                 for model in models if 'EPT' not in model
                 for faith in args.faithfulness])

    # Run bootstrapping
    for split_file in tqdm.tqdm(experiments):
        keys = [key.strip()
                for key in split_file.read_text('UTF-8').splitlines()]

        if args.bootstrap_trials > 0:
            # If bootstrapping is set, use resampling strategy and get an interval of estimation
            resamples = list(make_resamples(keys, args.sample_size, args.bootstrap_trials))
        else:
            # If bootstrapping is unset, use whole dataset and get a single estimation
            resamples = [keys]

        if DEBUG:
            metrics = [compute_resample(model, split_file.stem, samples, i)
                       for model in models
                       for i, samples in enumerate(resamples)]
        else:
            compute_resample_ray = ray.remote(compute_resample).options(num_gpus=0, num_cpus=args.num_cpu)
            metrics = ray.get([compute_resample_ray.remote(model, split_file.stem, samples, i)
                               for model in models
                               for i, samples in enumerate(resamples)])

        # Group by models and measures
        metric_group = {}
        for resample in metrics:
            for model, metricname, rindex, metricvalue in resample:
                if metricname not in metric_group:
                    metric_group[metricname] = defaultdict(list)
                metric_group[metricname][model].append((rindex, metricvalue))

        bootstrap_summary = {}
        for metricname, items in metric_group.items():
            items = {model: [x for _, x in sorted(values, key=lambda t: t[0])]
                     for model, values in items.items()}
            model_keys = items.keys()

            bootstrap_summary[metricname] = {
                'each': {model: get_confidence_interval(values)._asdict()
                         for model, values in items.items()},
                'paired': {'[%s]-[%s]' % (a, b): get_confidence_interval(items[a], items[b])._asdict()
                           for [a, b] in itertools.combinations(model_keys, 2)} if len(model_keys) > 1 else None
            }

        outdir = Path(models[0]).parent
        with (outdir / (split_file.stem + '-summary.yaml')).open('a+t', encoding='UTF-8') as fp:
            yaml_dump(bootstrap_summary,
                      fp, allow_unicode=True)
            fp.write('\n')

    if not DEBUG:
        ray.shutdown()
