SHELL := /bin/bash

default: train_beam
.PHONY: create train_cosmic train_beam train_energy study test

create:
	python chipsnet/run.py chipsnet/config/create.yaml

train_cosmic:
	python chipsnet/run.py chipsnet/config/train_cosmic.yaml

train_beam:
	python chipsnet/run.py chipsnet/config/train_beam.yaml

train_energy:
	python chipsnet/run.py chipsnet/config/train_energy.yaml

study:
	python chipsnet/run.py chipsnet/config/study.yaml

test:
	pytest --pydocstyle --flake8 --black -v -W ignore::pytest.PytestDeprecationWarning .