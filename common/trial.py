from ray.tune.trial import Trial

from learner import SupervisedTrainer


def trial_dirname_creator_generator(suffix: str = ''):
    def trial_dirname_creator(trial: Trial) -> str:
        trial_id = trial.trial_id
        config = trial.config

        name = SupervisedTrainer.get_trial_name(config, trial_id)

        if suffix:
            name += '_' + suffix

        return name

    return trial_dirname_creator


__all__ = ['trial_dirname_creator_generator']
