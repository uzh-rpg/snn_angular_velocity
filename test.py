import argparse
import os
# Must be set before importing torch.
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from config.utils import getTestConfigs
#from utils import Tester
from utils.testing import Tester


def main():
    parser = argparse.ArgumentParser(description='Test the SNN model')
    parser.add_argument('--datadir',
                        default=os.path.join(os.getcwd(), 'data'),
                        help='Data directory')
    parser.add_argument('--logdir',
                        default=os.path.join(os.getcwd(), 'logs/test'),
                        help='Test logging directory')
    parser.add_argument('--config',
                        default=os.path.join(os.getcwd(), 'test_config.yaml'),
                        help='Path to test config file')
    parser.add_argument('--write',
                        action='store_true',
                        help='Write network predictions to logging sub-directory')

    args = parser.parse_args()
    configs = getTestConfigs(args.logdir, args.config)

    tester = Tester(args.datadir, args.write, configs['log'], configs['general'])
    tester.test()


if __name__ == '__main__':
    main()
