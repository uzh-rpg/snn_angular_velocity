import os

from .base import DatasetBase


class TestDatabase(DatasetBase):
    def __init__(self, data_dir: str):
        super().__init__(mode='test')
        assert os.path.isdir(data_dir)
        self.test_dir = os.path.join(data_dir, 'test')
        assert os.path.isdir(self.test_dir)

        self.subsequences = list()
        for dirpath, dirnames, filenames in os.walk(self.test_dir):
            assert not dirnames, 'This fails if the directory contains sub-directories. This should not happen'
            for filename in filenames:
                assert self.isDataFile(filename)
                self.subsequences.append(os.path.join(dirpath, filename))
