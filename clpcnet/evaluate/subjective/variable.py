import argparse
import json
import multiprocessing as mp
from pathlib import Path

import pyfoal
import soundfile
import tqdm

import clpcnet


###############################################################################
# Constants
###############################################################################


DEFAULT_DIRECTORY = clpcnet.DATA_DIR / 'ravdess-hifi'
DEFAULT_OUTPUT_DIRECTORY = clpcnet.EVAL_DIR / \
                           'subjective' / \
                           'variable' / \
                           'ravdess-hifi'


###############################################################################
# Variable-rate pitch shifting
###############################################################################


def evaluate(directory=DEFAULT_DIRECTORY,
             output_directory=DEFAULT_OUTPUT_DIRECTORY,
             run='clpcnet',
             checkpoint=clpcnet.DEFAULT_CHECKPOINT,
             gpu=None):
    """Evaluate variable-rate pitch shifting on ravdess"""
    # Get list of examples to generate
    with open(clpcnet.data.partition_file('ravdess-variable')) as file:
        pairs = json.load(file)['test']

    # Setup output directory
    original_directory = output_directory / 'original'
    run_directory = output_directory / run
    original_directory.mkdir(exist_ok=True, parents=True)
    run_directory.mkdir(exist_ok=True, parents=True)

    # Setup multiprocessing
    pool = mp.get_context('spawn').Pool()

    # Iterate over pairs
    for pair in tqdm.tqdm(pairs):

        # Load text
        statement = pair[0].split('-')[4]
        text_file = clpcnet.ASSETS_DIR / 'text' / 'ravdess' / f'{statement}.txt'
        with open(text_file) as file:
            text = file.read()

        # Load audio
        source_file = clpcnet.data.stem_to_file('ravdess-variable',
                                                directory,
                                                pair[0])
        target_file = clpcnet.data.stem_to_file('ravdess-variable',
                                                directory,
                                                pair[1])
        source = clpcnet.load.audio(source_file)
        target = clpcnet.load.audio(target_file)

        # Compute pitch
        target_pitch, _ = clpcnet.pitch.from_audio(target, gpu)

        # Compute alignment
        source_alignment = pyfoal.align(text, source, clpcnet.SAMPLE_RATE)
        target_alignment = pyfoal.align(text, target, clpcnet.SAMPLE_RATE)

        # Output file template
        template = 'variable_{}_' + f'{pair[0]}-{pair[1]}.wav'

        # Generate with clpcnet
        clpcnet_file = run_directory / template.format(run)
        args = (clpcnet_file, source)
        kwargs = {'source_alignment': source_alignment,
                  'target_alignment': target_alignment,
                  'target_pitch': target_pitch,
                  'checkpoint_file': checkpoint,
                  'verbose': False}
        pool.apply_async(clpcnet.to_file, args, kwargs)
        # clpcnet.to_file(*args, **kwargs)

        # Write original file
        original_file = original_directory / template.format('original')
        soundfile.write(original_file, target, clpcnet.SAMPLE_RATE)

    # Close multiprocessing pool and wait for processes to finish
    pool.close()
    pool.join()

    # Convert to mp3
    wavfiles = list(output_directory.rglob('*.wav'))
    clpcnet.mp3.convert_files(wavfiles)

    # Remove wav files
    for file in wavfiles:
        file.unlink()


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=Path,
        default=DEFAULT_DIRECTORY,
        help='Root directory of the ravdess dataset')
    parser.add_argument(
        '--output_directory',
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help='The location to store files for subjective evaluation')
    parser.add_argument(
        '--run',
        default='clpcnet',
        help='The evaluation run')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=clpcnet.DEFAULT_CHECKPOINT,
        help='The checkpoint to use')
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='The index of the gpu to use')
    return parser.parse_args()


if __name__ == '__main__':
    evaluate(**vars(parse_args()))
