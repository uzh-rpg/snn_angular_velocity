from .logging import LogTestConfig
from .config import TestConfig


def getTestConfigs(log_dir: str, config_path: str):
    configs = dict()
    configs['log'] = LogTestConfig(log_dir, config_path)
    configs['general'] = TestConfig(config_path)
    return configs
