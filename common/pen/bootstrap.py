from collections import namedtuple
from typing import List

from numpy import mean, std, quantile
from numpy.random import Generator, PCG64

Metric = namedtuple('Metric', ('mean', 'stdev', 'sample_sz', 'quantiles'))


def make_resamples(keys: list, resample_sz: int = 20, num_resamples: int = 1000):
    generator = Generator(PCG64(1))

    for _ in range(num_resamples):
        samples = generator.choice(len(keys), size=resample_sz)
        yield [keys[i] for i in samples]


def get_metric_summary(metrics: List[float], sz: int = 0) -> Metric:
    mu = float(mean(metrics))
    sigma = float(std(metrics, ddof=1))
    if sz == 0:
        sz = len(metrics)

    names = []
    quantiles = []

    for conf_level in [90, 95, 99]:
        p_level = 1 - conf_level / 100
        left_p_both = p_level / 2
        right_p_both = 1 - left_p_both
        right_p_onetail = 1 - p_level
        positions = [p_level, right_p_onetail, left_p_both, right_p_both]
        names += ['right_onetail_%s' % conf_level, 'left_onetail_%s' % conf_level, 'confidence_low_%s' % conf_level, 'confidence_high_%s' % conf_level]
        quantiles += [float(q) for q in quantile(metrics, positions).tolist()]
    quantiles = {n: q for n, q in zip(names, quantiles)}

    return Metric(mean=mu, stdev=sigma, sample_sz=sz, quantiles=quantiles)


def get_confidence_interval(resamples: List[Metric], paired: List[Metric] = None) -> Metric:
    sz = resamples[0].sample_sz
    assert all(m.sample_sz == sz for m in resamples)

    if paired is not None:
        assert len(resamples) == len(paired)
        assert all(m.sample_sz == sz for m in paired)

        resamples = [Metric(r.mean - p.mean, None, sz, None)
                     for r, p in zip(resamples, paired)]

    return get_metric_summary([m.mean for m in resamples], sz)

