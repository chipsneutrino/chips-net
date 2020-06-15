from os import path
import chipsnet.config as config


def test_get():
    config_path = "./config/train.yaml"
    conf = config.get(config_path)
    assert conf.task == "train"


def test_setup_dirs():
    config_path = "./config/train.yaml"
    conf = config.get(config_path)
    config.setup_dirs(conf, False)
    assert path.exists(conf.exp.exp_dir)
