from os import path
import pytest  # noqa: F401
import chipscvn.config as config


def test_get():
    config_path = './data/config/train.yaml'
    conf = config.get(config_path)
    assert conf.task == 'train'


def test_setup_dirs():
    config_path = './data/config/train.yaml'
    conf = config.get(config_path)
    config.setup_dirs(conf, False)
    assert path.exists(conf.exp.exp_dir)
