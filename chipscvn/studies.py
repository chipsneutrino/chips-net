"""Study classes for carrying out SHERPA studies

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module contains all the study classes used to conduct SHERPA studies
on the different models. All are derived from the BaseStudy class and
allow for the customisation of the study depending on the model.
"""

import sherpa
import chipscvn.data
import chipscvn.models
import chipscvn.utils


class BaseStudy(object):
    """Base study class which all implementations derive from."""
    def __init__(self, config):
        self.config = config
        self.init_study()

    def init_study(self):
        """Initialise the SHERPA study, overide in derived model class."""
        raise NotImplementedError

    def run(self):
        """Run the SHERPA study."""
        for trial in self.study:
            # We need to take the trial parameters and adjust the configuration to match them
            for key in trial.parameters.keys():
                if 'data.' in key:
                    self.config[key.replace('data.', '')] = trial.parameters[key]
                elif 'model.' in key:
                    self.config[key.replace('model.', '')] = trial.parameters[key]
                elif 'trainer.' in key:
                    self.config[key.replace('trainer.', '')] = trial.parameters[key]
                else:
                    print('Error: invalid study parameter key!')
                    raise SystemExit

            data = chipscvn.data.DataLoader(self.config)
            model = chipscvn.utils.get_model(self.config)
            model.summarise()
            trainer = chipscvn.utils.get_trainer(self.config, model, data)
            cb = [self.study.keras_callback(trial,
                                            objective_name=self.objective,
                                            context_names=self.context)]
            trainer.train(cb)
            self.study.finalize(trial)

        self.study.save()


class SingleParStudy(BaseStudy):
    """Single parameter model study class."""
    def __init__(self, config):
        super().__init__(config)

    def init_study(self):
        pars = [
            sherpa.Ordinal(name='data.batch_size', range=self.config.study.data.batch_size),
            sherpa.Continuous(name='model.lr', range=self.config.study.model.lr, scale='log'),
            sherpa.Ordinal(name='model.dense_units', range=self.config.study.model.dense_units),
            sherpa.Continuous(name='model.dropout', range=self.config.study.model.dropout),
            sherpa.Ordinal(name='model.kernel_size', range=self.config.study.model.kernel_size),
            sherpa.Ordinal(name='model.filters', range=self.config.study.model.filters),
        ]
        algorithm = sherpa.algorithms.RandomSearch(max_num_trials=self.config.study.trials)
        self.study = sherpa.Study(parameters=pars, algorithm=algorithm, lower_is_better=True,
                                  output_dir=self.config.exp.exp_dir)
        self.objective = 'val_loss'
        self.context = ['val_mae', 'val_mse']


class ClassificationStudy(BaseStudy):
    """Classification model study class."""
    def __init__(self, config):
        super().__init__(config)

    def init_study(self):
        pars = [
            sherpa.Ordinal(name='data.batch_size', range=self.config.study.data.batch_size),
            sherpa.Continuous(name='model.lr', range=self.config.study.model.lr, scale='log'),
            sherpa.Ordinal(name='model.dense_units', range=self.config.study.model.dense_units),
            sherpa.Continuous(name='model.dropout', range=self.config.study.model.dropout),
            sherpa.Ordinal(name='model.kernel_size', range=self.config.study.model.kernel_size),
            sherpa.Ordinal(name='model.filters', range=self.config.study.model.filters),
        ]
        algorithm = sherpa.algorithms.RandomSearch(max_num_trials=self.config.study.trials)
        self.study = sherpa.Study(parameters=pars, algorithm=algorithm, lower_is_better=True,
                                  output_dir=self.config.exp.exp_dir)
        self.objective = 'val_loss'
        self.context = ['val_pdg_accuracy', 'val_type_accuracy']


class MultiTaskStudy(BaseStudy):
    """Multi-task model study class."""
    def __init__(self, config):
        super().__init__(config)

    def init_study(self):
        pars = [
            sherpa.Ordinal(name='data.batch_size', range=self.config.study.data.batch_size),
            sherpa.Continuous(name='model.lr', range=self.config.study.model.lr, scale='log'),
            sherpa.Ordinal(name='model.dense_units', range=self.config.study.model.dense_units),
            sherpa.Continuous(name='model.dropout', range=self.config.study.model.dropout),
            sherpa.Ordinal(name='model.kernel_size', range=self.config.study.model.kernel_size),
            sherpa.Ordinal(name='model.filters', range=self.config.study.model.filters),
        ]
        algorithm = sherpa.algorithms.RandomSearch(max_num_trials=self.config.study.trials)
        self.study = sherpa.Study(parameters=pars, algorithm=algorithm, lower_is_better=True,
                                  output_dir=self.config.exp.exp_dir)
        self.objective = 'val_loss'
        self.context = ['val_pdg_accuracy', 'val_type_accuracy', 'val_nuEnergy_mae']
