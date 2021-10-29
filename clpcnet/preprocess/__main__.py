import argparse
from pathlib import Path

import clpcnet


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='vctk',
        help='The dataset to preprocess')
    parser.add_argument(
        '--directory',
        type=Path,
        default=clpcnet.DATA_DIR,
        help='The data directory')
    parser.add_argument(
        '--cache',
        type=Path,
        default=clpcnet.CACHE_DIR,
        help='The cache directory')

    # Extend directories with dataset name
    args = parser.parse_args()
    args.directory = args.directory / args.dataset
    args.cache = args.cache / args.dataset

    return args


if __name__ == '__main__':
    clpcnet.preprocess.from_dataset_to_files(**vars(parse_args()))
