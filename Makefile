SHELL := /bin/bash
ANALYSIS_DIR = ./config/analysis

default: train

create:
	python chipsnet/run.py ./config/create.yaml

train:
	python chipsnet/run.py ./config/train.yaml

study:
	python chipsnet/run.py ./config/study.yaml

analysis: 
	@for f in $(shell ls ${ANALYSIS_DIR}); do python chipsnet/run.py $${f}; done

clean:
	rm -rf ./data/models/*
	rm -rf ./data/output/*

test:
	pytest --pydocstyle --flake8 --black -v .

.PHONY: analysis clean test dependencies
