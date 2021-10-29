import argparse
import functools
import itertools
import json
import random
from pathlib import Path

import tqdm

import clpcnet


###############################################################################
# Partition
###############################################################################


def daps_segmented(directory):
    """Partition daps-segmented dataset"""
    files = list(directory.rglob('*.wav'))

    # Get files corresponding to each selected speaker
    speaker_files = {
        s: [f for f in files if f.stem.split('_')[0] == s]
        for s in ['f1', 'f3', 'f4', 'f5', 'f6', 'm1', 'm3', 'm4', 'm5', 'm6']}

    # Deterministic but random selection
    random.seed(0)
    test_files = itertools.chain(
        *[random.sample(f, 10) for f in speaker_files.values()])

    return {'test': [f.stem for f in test_files]}


def ravdess_hifi(directory):
    """Partition ravdess dataset"""
    partition_file = clpcnet.ASSETS_DIR / 'partition' / 'ravdess-variable.json'
    with open(partition_file) as file:
        pairs = json.load(file)['test']
    stems = set(list(itertools.chain(*pairs)))
    return {'test': list(stems)}


def ravdess_variable(directory, gpu=None):
    """Partition ravdess dataset into prosody transfer pairs"""
    pairs = []
    generator = clpcnet.evaluate.prosody.ravdess_generator(directory, gpu)
    for transfer in tqdm.tqdm(generator):
        pairs.append(transfer.name.split('_'))
    return {'test': pairs}


def vctk(directory, rejects=['p341_101']):
    """Partition vctk dataset"""
    # Load speaker info
    with open(directory / 'speaker-info.txt') as file:
        lines = file.readlines()
        speakers = [VCTKSpeaker(line) for line in lines[1:]]

        # Filter out speakers where mic 2 is not available
        speakers = [s for s in speakers if s.id not in ['p280', 'p315']]

    # Shuffle speakers
    random.seed(0)
    random.shuffle(speakers)

    # Partition speakers
    male = [s.id for s in speakers if s.gender == 'M']
    female = [s.id for s in speakers if s.gender == 'F']
    train_speaker = male[:-4] + female[:-4]
    test_speaker = male[-4:] + female[-4:]

    # Get file lists relative to root directory
    text_directory = directory / 'txt'
    train_files = chain_list_files(text_directory, train_speaker)
    test_files = chain_list_files(text_directory, test_speaker)

    # Require mic 2 be available
    train_files = vctk_mic_check(train_files)
    test_files = vctk_mic_check(test_files)

    # Move some train files to a separate test partition of seen speakers
    test_seen_speaker = male[:10] + female[:10]
    test_seen_files = [
        random.sample([f for f in train_files
                       if s in f.stem and f.stem not in rejects], 5)
        for s in test_seen_speaker]
    test_seen_files = list(itertools.chain(*test_seen_files))
    train_files = [f for f in train_files if f not in test_seen_files]

    # Pack partition dictionary
    return {
        'train': sorted([f.stem for f in train_files
                         if f.stem not in rejects]),
        'test': sorted([f.stem for f in test_files
                        if f.stem not in rejects]),
        'test-seen': sorted([f.stem for f in test_seen_files])}


###############################################################################
# Utilities
###############################################################################


class VCTKSpeaker:

    def __init__(self, line):
        line = self.strip_comment(line)
        self.id, _, self.gender = line.split()[:3]

    @staticmethod
    def strip_comment(line):
        comment_index = line.find('(')
        return line[:comment_index] if comment_index != -1 else line


def chain_list_files(directory, subdirectories):
    """List files in all subdirectories"""
    return list(itertools.chain(
        *[(directory / sd).glob('**/*') for sd in subdirectories]))


def vctk_mic_check(files):
    """Filter files by whether mic 2 is available"""
    directory = files[0].parent.parent.parent / 'wav48_silence_trimmed'
    result = []
    for file in files:
        speaker = file.parent.name
        if (directory / speaker / f'{file.stem}_mic2.flac').exists():
            result.append(file)
    return result


###############################################################################
# Entry point
###############################################################################


def main():
    """Partition dataset"""
    # Parse command-line arguments
    args = parse_args()

    # Get partitioning function
    if args.dataset == 'daps-segmented':
        partition_fn = daps_segmented
    elif args.dataset == 'ravdess-hifi':
        partition_fn = ravdess_hifi
    elif args.dataset == 'ravdess-variable':
        partition_fn = functools.partial(ravdess_variable, gpu=args.gpu)
    elif args.dataset == 'vctk':
        partition_fn = vctk
    else:
        raise ValueError(f'No dataset {args.dataset}')

    # Partition
    partition = partition_fn(args.directory)

    # Save to disk
    with open(clpcnet.data.partition_file(args.dataset), 'w') as file:
        json.dump(partition, file, indent=4)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='vctk',
        help='The name of the dataset')
    parser.add_argument(
        '--directory',
        type=Path,
        default=clpcnet.DATA_DIR,
        help='The data directory')
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='The gpu to use')

    # Extend directory with dataset name
    args = parser.parse_args()
    dataset = \
        'ravdess-hifi' if args.dataset == 'ravdess-variable' else args.dataset
    args.directory = args.directory / dataset

    return args


if __name__ == '__main__':
    main()
