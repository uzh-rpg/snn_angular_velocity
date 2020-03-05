import inspect
import os
from pathlib import Path
import shutil
import time


class LogTestConfig:
    def __init__(self, log_dir: str, config_path: str):
        assert os.path.exists(config_path)

        log_dir = os.path.expanduser(log_dir)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(log_dir, timestr)

        os.makedirs(log_dir)

        config_dir = os.path.join(log_dir, 'config')
        os.mkdir(config_dir)
        # Copy the test config file for reproducability.
        shutil.copyfile(config_path, os.path.join(config_dir, 'test_config.yaml'))

        mdl_dir = os.path.join(config_dir, 'model')
        os.mkdir(mdl_dir)

        out_dir = os.path.join(log_dir, 'out')
        os.mkdir(out_dir)

        self.config_dir = config_dir
        self.mdl_dir = mdl_dir
        self.out_dir = out_dir

    def copyModelFile(self, network_object):
        # Copy the model file for reproducability.
        net_filename = inspect.getsourcefile(network_object.__class__)
        dest_filename = os.path.join(self.mdl_dir, Path(net_filename).name)
        assert not os.path.exists((dest_filename))
        shutil.copyfile(net_filename, dest_filename)
        with open(os.path.join(self.mdl_dir, 'mdl_pwd.txt'), 'w') as f:
            f.write(net_filename)

    def getOutDir(self):
        return self.out_dir
