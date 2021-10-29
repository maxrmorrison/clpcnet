import argparse
import copy
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pyfoal
import soundfile
import tqdm

import clpcnet


###############################################################################
# Gather
###############################################################################


def from_files_to_files(examples, output_directory, gpu=None):
    """Gather files for evaluation"""
    output_directory.mkdir(exist_ok=True, parents=True)
    iterator = tqdm.tqdm(examples,
                         desc='Setting up evaluation directory',
                         dynamic_ncols=True)
    for example in iterator:
        stem = f'{example.stem}-{"" if example.seen else "un"}seen'
        prefix = output_directory / stem

        # Copy audio
        dst_audio_file = f'{prefix}.wav'
        soundfile.write(dst_audio_file,
                        clpcnet.load.audio(example.audio_file),
                        clpcnet.SAMPLE_RATE)

        # Estimate pitch
        pitch, periodicity = clpcnet.pitch.from_file(example.audio_file, gpu)

        # Scale pitch
        low = np.clip(.71 * pitch, clpcnet.FMIN, clpcnet.FMAX)
        high = np.clip(1.41 * pitch, clpcnet.FMIN, clpcnet.FMAX)

        # Save original pitch and periodicity and scaled pitch
        np.save(f'{prefix}-pitch.npy', pitch)
        np.save(f'{prefix}-ps-71-pitch.npy', low)
        np.save(f'{prefix}-ps-141-pitch.npy', high)
        np.save(f'{prefix}-periodicity.npy', periodicity)

        if example.text_file is not None:
            # Copy text
            dst_text_file = f'{prefix}.txt'
            shutil.copy2(example.text_file, dst_text_file)

            # Force align
            alignment = pyfoal.from_file(str(dst_text_file), str(dst_audio_file))

            # Time-stretch alignment by constant factor
            slow = stretch_alignment(alignment, .5)
            fast = stretch_alignment(alignment, 2.)

            # Save alignments
            alignment.save(f'{prefix}.json')
            slow.save(f'{prefix}-ts-50.json')
            fast.save(f'{prefix}-ts-200.json')


###############################################################################
# Example selection
###############################################################################


def daps(directory, stems):
    """Select evaluation examples from daps"""
    selected = [stem for stem in stems if '_script5_' in str(stem)
                if int(stem.split('_')[0][1:]) < 5]
    return [Example(stem, directory / 'clean' / f'{stem}.wav', None, False)
            for stem in selected]


def daps_segmented(directory, stems):
    """Select evaluation examples from daps"""
    examples = []
    for stem in stems:
        # Get files
        audio_file = clpcnet.data.stem_to_file('daps-segmented',
                                                directory,
                                                stem)
        text_file = audio_file.with_suffix('.txt')

        # Create example
        examples.append(Example(stem, audio_file, text_file, False))

    return examples


def ravdess_hifi(directory, stems):
    """Select evaluation samples from ravdess"""
    # Get deterministic but random set of stems
    random.seed(0)
    stems = random.sample(stems, 100)

    # Create examples
    text_directory = clpcnet.ASSETS_DIR / 'text' / 'ravdess'
    return [Example(stem,
                    clpcnet.data.stem_to_file('ravdess-hifi', directory, stem),
                    text_directory / f'{stem.split("-")[4]}.txt',
                    False)
            for stem in stems]


def vctk(directory, unseen, seen):
    """Select evaluation examples from vctk"""
    # Load speaker info
    with open(directory / 'speaker-info.txt') as file:
        lines = file.readlines()
        speakers = [clpcnet.partition.VCTKSpeaker(line) for line in lines[1:]]
    speakers = [s for s in speakers if s.id != 'p362']

    # Pick a few speakers
    random.seed(0)
    unseen_speakers = sample_speakers(speakers, unseen)
    seen_speakers = sample_speakers(speakers, seen)

    # For each speaker, pick a few files
    selected = []
    stems = unseen + seen
    for speaker in unseen_speakers + seen_speakers:
        speaker_stems = [s for s in stems if speaker in s]
        selected.extend(random.sample(speaker_stems, 4))

    # Get absolute paths to audio and text files
    audio_directory = directory / 'wav48_silence_trimmed'
    audio_files = [audio_directory / s.split('_')[0] / f'{s}_mic2.flac'
                   for s in selected]
    text_directory = directory / 'txt'
    text_files = [text_directory / s.split('_')[0] / f'{s}.txt'
                  for s in selected]

    iterator = enumerate(zip(selected, audio_files, text_files))
    return [Example(s, audio, text, i >= len(selected) // 2)
            for i, (s, audio, text) in iterator]


###############################################################################
# Utilities
###############################################################################


class Example:

    def __init__(self, stem, audio_file, text_file, seen):
        self.stem = stem
        self.audio_file = audio_file
        self.text_file = text_file
        self.seen = seen


def sample_speakers(speakers, stems, n=8):
    """Sample from a list of speakers"""
    # Get relevant speakers
    relevant = set(s.split('_')[0] for s in stems)
    stem_speakers = [s for s in speakers if s.id in relevant]

    # Shuffle
    random.shuffle(stem_speakers)

    # Pick first n // 2 of each gender
    male = [s.id for s in stem_speakers if s.gender == 'M']
    female = [s.id for s in stem_speakers if s.gender == 'F']
    return male[:n // 2] + female [:n // 2]


def stretch_alignment(alignment, rate):
    """Stretch the alignment by the given rate"""
    alignment = copy.deepcopy(alignment)
    durations = [(1. / rate) * p.duration() for p in alignment.phonemes()]
    alignment.update(durations=durations)
    return alignment


###############################################################################
# Entry point
###############################################################################


def main():
    """Create a directory of files for evaluation"""
    # Parse command-line arguments
    args = parse_args()

    # Get test partition stems
    partition_file = clpcnet.ASSETS_DIR / 'partition' / f'{args.dataset}.json'
    with open(partition_file) as file:
        partition = json.load(file)

    # Get paths to selected examples
    if args.dataset == 'daps':
        test_unseen = partition['test']
        examples = daps(args.directory, test_unseen)
    elif args.dataset == 'daps-segmented':
        test_unseen = partition['test']
        examples = daps_segmented(args.directory, test_unseen)
    elif args.dataset == 'ravdess-hifi':
        test_unseen = partition['test']
        examples = ravdess_hifi(args.directory, test_unseen)
    elif args.dataset == 'vctk':
        test_unseen, test_seen = partition['test'], partition['test-seen']
        examples = vctk(args.directory, test_unseen, test_seen)
    else:
        raise ValueError(f'No dataset {args.dataset}')

    # Gather files for evaluation
    from_files_to_files(examples, args.output_directory, args.gpu)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='vctk',
        help='The dataset to gather evaluation files from')
    parser.add_argument(
        '--directory',
        type=Path,
        default=clpcnet.DATA_DIR,
        help='The root directory of the dataset')
    parser.add_argument(
        '--output_directory',
        type=Path,
        default=clpcnet.EVAL_DIR / 'objective' / 'constant',
        help='The output evaluation directory')
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='The gpu to use for pitch estimation')

    # Extend directories with dataset name
    args = parser.parse_args()
    args.directory = args.directory / args.dataset
    args.output_directory = args.output_directory / args.dataset / 'data'

    return args


if __name__ == '__main__':
    main()
