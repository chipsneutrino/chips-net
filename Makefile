SHELL := /bin/bash
ANALYSIS_DIR = ./config/analysis

default: train_beam

create:
	python chipsnet/run.py ./config/create.yaml

train_cosmic:
	python chipsnet/run.py ./config/train_cosmic.yaml

train_beam:
	python chipsnet/run.py ./config/train_beam.yaml

train_energy:
	python chipsnet/run.py ./config/train_energy.yaml

study:
	python chipsnet/run.py ./config/study.yaml

analysis: 
	@for f in $(shell ls ${ANALYSIS_DIR}); do python chipsnet/run.py $${f}; done

update:
	conda env update --file environment.yaml

clean:
	rm -rf ./data/models/*
	rm -rf ./data/output/*

black:
	black chipsnet/
	black tests/

test:
	pytest --pydocstyle --flake8 --black -v -W ignore::pytest.PytestDeprecationWarning .

.PHONY: analysis clean test dependencies
