# -*- coding: utf-8 -*-

"""Module containing SHERPA study class
"""

from sherpa import Choice, Continuous, Ordinal, algorithms, Study
import chipsnet.data
import chipsnet.models
import chipsnet.trainer


class SherpaStudy(object):
    """Sherpa study class
    """

    def __init__(self, config):
        """Initialise the SherpaStudy.
        Args:
            config (str): Dotmap configuration namespace
        """
        self.config = config
        self.init_study()

    def init_study(self):
        """Initialise the SHERPA study, overide in derived model class.
        """
        pars = [
            Choice(name='data.unstack', range=self.config.study.data.unstack),
            Choice(name='data.augment', range=self.config.study.data.augment),
            Continuous(name='model.lr', range=self.config.study.model.lr, scale='log'),
            Continuous(name='model.lr_decay', range=self.config.study.model.lr_decay),
            Ordinal(name='model.dense_units', range=self.config.study.model.dense_units),
            Continuous(name='model.dropout', range=self.config.study.model.dropout),
            Ordinal(name='model.filters', range=self.config.study.model.filters),
            Choice(name='model.reco_pars', range=self.config.study.model.reco_pars),
            Ordinal(name='model.se_ratio', range=self.config.study.model.se_ratio),
            Choice(name='model.learn_weights', range=self.config.study.model.learn_weights),
            Choice(name='model.precision_policy', range=self.config.study.model.precision_policy),
            Ordinal(name='trainer.batch_size', range=self.config.study.trainer.batch_size),
        ]

        algorithm = algorithms.RandomSearch(max_num_trials=self.config.study.trials)
        self.study = Study(parameters=pars, algorithm=algorithm, lower_is_better=True,
                           output_dir=self.config.exp.exp_dir)

        self.objective = 'val_loss'
        self.context = ['val_mae', 'val_mse']

        self.objective = 'val_loss'
        self.context = ['val_accuracy']

        self.objective = 'val_loss'
        self.context = ['val_t_cat_accuracy', 'val_t_nuEnergy_mae']

    def run(self):
        """Run the SHERPA study.
        """
        for trial in self.study:
            print(" *** Trial: {} *** ".format(trial.id))
            print("*******************")
            print(trial.parameters)
            print("*******************")
            # We need to take the trial parameters and adjust the configuration to match them
            for key in trial.parameters.keys():
                if 'data.' in key:
                    self.config.data[key.replace('data.', '')] = trial.parameters[key]
                elif 'model.' in key:
                    self.config.model[key.replace('model.', '')] = trial.parameters[key]
                elif 'trainer.' in key:
                    self.config.trainer[key.replace('trainer.', '')] = trial.parameters[key]
                else:
                    print('Error: invalid study parameter key!')
                    raise SystemExit

            data = chipsnet.data.Reader(self.config)
            model = chipsnet.models.Model(self.config)
            trainer = chipsnet.trainer.Trainer(self.config, model, data)

            print(self.config.trainer.epochs)
            for epoch in range(self.config.trainer.epochs):
                history = trainer.train(epochs=1)
                self.study.add_observation(trial=trial, iteration=epoch,
                                           objective=history.history['val_accuracy'])
            self.study.finalize(trial)
            self.study.save()
