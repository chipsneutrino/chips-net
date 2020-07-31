# -*- coding: utf-8 -*-

"""Module containing the SherpaStudy used for hyperparamter optimisation using SHERPA."""

from sherpa import Choice, Continuous, Ordinal, algorithms, Study
import chipsnet.data
import chipsnet.models
import chipsnet.trainer


class SherpaStudy(object):
    """Sherpa study class."""

    def __init__(self, config):
        """Initialise the SherpaStudy.

        Args:
            config (dotmap.DotMap): configuration namespace
        """
        self.config = config
        self.init_study()

    def init_study(self):
        """Initialise the SHERPA study, overide in derived model class."""
        pars = [
            Choice(
                name="data.seperate_channels",
                range=self.config.study.data.seperate_channels,
            ),
            Choice(name="data.augment", range=self.config.study.data.augment),
            Continuous(name="model.lr", range=self.config.study.model.lr, scale="log"),
            Continuous(name="model.lr_decay", range=self.config.study.model.lr_decay),
            Ordinal(
                name="model.dense_units", range=self.config.study.model.dense_units
            ),
            Continuous(name="model.dropout", range=self.config.study.model.dropout),
            Ordinal(name="model.filters", range=self.config.study.model.filters),
            Choice(name="model.reco_pars", range=self.config.study.model.reco_pars),
            Ordinal(name="model.se_ratio", range=self.config.study.model.se_ratio),
            Choice(
                name="model.learn_weights", range=self.config.study.model.learn_weights
            ),
            Ordinal(
                name="trainer.batch_size", range=self.config.study.trainer.batch_size
            ),
        ]

        algorithm = algorithms.RandomSearch(max_num_trials=self.config.study.trials)
        self.study = Study(
            parameters=pars,
            algorithm=algorithm,
            lower_is_better=False,
            output_dir=self.config.exp.exp_dir,
            dashboard_port=8880,
        )

    def run(self):
        """Run the SHERPA study."""
        for trial in self.study:
            print(" *** Trial: {} *** ".format(trial.id))
            print("*******************")
            print(trial.parameters)
            print("*******************")
            # We need to take the trial parameters and adjust the configuration to match them
            for key in trial.parameters.keys():
                if "data." in key:
                    self.config.data[key.replace("data.", "")] = trial.parameters[key]
                elif "model." in key:
                    self.config.model[key.replace("model.", "")] = trial.parameters[key]
                elif "trainer." in key:
                    self.config.trainer[key.replace("trainer.", "")] = trial.parameters[
                        key
                    ]
                else:
                    print("Error: invalid study parameter key!")
                    raise SystemExit

            data = chipsnet.data.Reader(self.config)
            model = chipsnet.models.Model(self.config)
            trainer = chipsnet.trainer.Trainer(self.config, model, data)

            history = trainer.train()
            print(history.history[self.config.study.objective])
            for epoch in range(self.config.trainer.epochs):
                context_dict = {
                    context: history.history[context][epoch]
                    for context in self.config.study.context
                }
                self.study.add_observation(
                    trial=trial,
                    iteration=epoch,
                    objective=history.history[self.config.study.objective][epoch],
                    context=context_dict,
                )
            self.study.finalize(trial)
            self.study.save()
