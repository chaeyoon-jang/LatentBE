import argparse
import torch

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n-epochs', '-e', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--mode', type=str , default='teacher',
                        help="Training mode [KD/latentBE/latentBE_div/teacher]")
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Number of examples per one mini-batch')
    parser.add_argument('--ensemble_size', default=4, type=int,
                        help='Ensemble size')
        
    return parser

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()