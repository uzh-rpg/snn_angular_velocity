import os
import torch

from strictyaml import load, Map, Str, Int, EmptyNone


class TestConfig:
    _str_to_dtype = {
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'uint8': torch.uint8,
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
    }
    _schema = Map({
        'simulation': Map({
            'Ts': Int(),                                # Time-discretization in milliseconds
            'tSample': Int(),                           # Number of simulation steps
            'tStartLoss': Int(),                        # Start computing loss at this time-step
        }),
        'model': Map({
            'type': Str(),                              # {cnn5-avgp-fc1}
            'CkptFile': Str(),                          # Path to checkpoint
            'dtype': EmptyNone() | Str(),               # {float16, float32, float64, uint8, int8, int16, int32, int64}
        }),
        'batchsize': Int(),
        'hardware': Map({
            'readerThreads': EmptyNone() | Int(),       # {empty: cpu_count, 0: main thread, >0: num threads used}
            'gpuDevice': Int(),                         # GPU to be used by device number
        }),
    })

    def __init__(self, config_filepath):
        with open(config_filepath, 'r') as stream:
            self.dictionary = load(stream.read(), self._schema).data


        # Some sanity checks.
        assert self.dictionary['simulation']['Ts'] == 1, "Only 1 ms time-step is tested"
        assert self.dictionary['simulation']['tSample'] == 100, "Only 100 ms available"
        assert self.dictionary['simulation']['tSample'] > self.dictionary['simulation']['tStartLoss']
        assert os.path.exists(self.dictionary['model']['CkptFile'])
        assert self.dictionary['batchsize'] >= 1

        model_dtype_str = self.dictionary['model']['dtype']
        if model_dtype_str is None:
            self.dictionary['model']['dtype'] = torch.float32
        else:
            self.dictionary['model']['dtype'] = self._str_to_dtype[model_dtype_str]

        self.dictionary['hardware']['gpuDevice'] = torch.device('cuda:{}'.format(
            self.dictionary['hardware']['gpuDevice']))

        if self.dictionary['hardware']['readerThreads'] is None:
            self.dictionary['hardware']['readerThreads'] = os.cpu_count()

    def __getitem__(self, key):
        return self.dictionary[key]

    def __setitem__(self, key, value):
        self.dictionary[key] = value
