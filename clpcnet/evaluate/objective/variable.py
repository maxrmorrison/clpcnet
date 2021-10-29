import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pyfoal
import soundfile
import torch
import tqdm

import clpcnet


###############################################################################
# Constants
###############################################################################


DEFAULT_DIRECTORY = clpcnet.DATA_DIR / 'ravdess-hifi'


###############################################################################
# Variable-rate pitch shifting
###############################################################################


def evaluate(directory=DEFAULT_DIRECTORY,
             run='clpcnet',
             checkpoint=clpcnet.DEFAULT_CHECKPOINT,
             gpu=None):
    """Evaluate variable-rate pitch shifting on ravdess"""
    # Get list of examples to generate
    with open(clpcnet.data.partition_file('ravdess-variable')) as file:
        pairs = json.load(file)['test']

    # Setup output directory
    output_directory = clpcnet.EVAL_DIR / \
                       'objective' / \
                       'variable' / \
                       'ravdess-hifi' / \
                       run
    output_directory.mkdir(exist_ok=True, parents=True)

    # Setup multiprocessing
    pool = mp.get_context('spawn').Pool()

    # Iterate over pairs
    iterator = tqdm.tqdm(
        pairs,
        total=len(pairs),
        dynamic_ncols=True,
        desc='Generating variable-ratio examples')
    for pair in iterator:

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
        source_pitch, source_periodicity = clpcnet.pitch.from_audio(source, gpu)
        target_pitch, target_periodicity = clpcnet.pitch.from_audio(target, gpu)

        # Compute alignment
        source_alignment = pyfoal.align(text, source, clpcnet.SAMPLE_RATE)
        target_alignment = pyfoal.align(text, target, clpcnet.SAMPLE_RATE)

        # Align periodicity for evaluation
        aligned_periodicity = clpcnet.pitch.align(target_periodicity,
                                                  source_periodicity,
                                                  target_alignment,
                                                  source_alignment)

        # Output file prefix
        prefix = output_directory / f'{pair[0]}_{pair[1]}'
        output_file = prefix.parent / (prefix.stem + '-transfer.wav')

        # Perform pitch shifting
        args = (output_file, source)
        kwargs = {'source_alignment': source_alignment,
                  'target_alignment': target_alignment,
                  'target_pitch': target_pitch,
                  'checkpoint_file': checkpoint,
                  'verbose': False}
        pool.apply_async(clpcnet.to_file, args, kwargs)
        # clpcnet.to_file(*args, **kwargs)

        # Save stuff
        np.save(prefix.parent / (prefix.stem + '-source.npy'), source_pitch)
        np.save(prefix.parent / (prefix.stem + '-target.npy'), target_pitch)
        np.save(prefix.parent / (prefix.stem + '-aligned.npy'), aligned_periodicity)
        np.save(prefix.parent / (prefix.stem + '-sourceharm.npy'),
                source_periodicity)
        np.save(prefix.parent / (prefix.stem + '-targetharm.npy'),
                target_periodicity)
        source_alignment.save(prefix.parent / (prefix.stem + '-source.json'))
        target_alignment.save(prefix.parent / (prefix.stem + '-target.json'))
        with open(prefix.with_suffix('.txt'), 'w') as file:
            file.write(text)
        soundfile.write(f'{prefix}-source.wav', source, clpcnet.SAMPLE_RATE)
        soundfile.write(f'{prefix}-target.wav', target, clpcnet.SAMPLE_RATE)

    # Close multiprocessing pool and wait for processes to finish
    pool.close()
    pool.join()

    # Pitch estimation
    files = list(output_directory.glob('*-transfer.wav'))
    prefixes = [f.parent / f.stem for f in files]
    clpcnet.pitch.from_files_to_files(files, prefixes, gpu)

    # Forced alignment
    pyfoal.from_files_to_files(
        [f.parent / (f.stem[:-9] + '.txt') for f in files],
        files,
        [f.with_suffix('.json') for f in files])

    # Get pitch files to evaluate
    source_pitch_files = sorted(output_directory.glob('*-pitch.npy'))
    target_pitch_files = sorted(output_directory.glob('*-target.npy'))
    source_periodicity_files = sorted(output_directory.glob('*-periodicity.npy'))
    target_periodicity_files = sorted(output_directory.glob('*-aligned.npy'))

    # Evaluate pitch
    rmse, precision, recall, f1, gpe_20, gpe_50, hist = \
        clpcnet.evaluate.pitch.from_files(source_pitch_files,
                                          target_pitch_files,
                                          source_periodicity_files,
                                          target_periodicity_files)
    run_results = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'rmse_cents': rmse,
            'gpe_20': gpe_20,
            'gpe_50': gpe_50}
    hist_file = output_directory / f'{run}.png'
    clpcnet.evaluate.plot.write_histogram(hist_file, hist)

    # Evaluate time-stretch
    duration_dict = {}

    # Get duration files to evaluate
    source_duration_files = sorted(output_directory.glob('*-transfer.json'))
    target_duration_files = sorted(output_directory.glob('*-target.json'))

    # Forced alignment rmse metric
    duration_rmse = clpcnet.evaluate.duration.from_files(
        source_duration_files, target_duration_files)
    duration_dict['force-align'] = {'rmse_seconds': duration_rmse}

    # DTW metrics
    source_audio_files = [f.with_suffix('.wav') for f in source_duration_files]
    target_audio_files = [f.with_suffix('.wav') for f in target_duration_files]
    distance = clpcnet.evaluate.dtw.from_files(source_audio_files,
                                               target_audio_files)
    duration_dict['dtw'] = {'distance': distance}
    run_results.update(duration_dict)

    # Load results file
    try:
        with open(output_directory / 'results.json') as file:
            results = json.load(file)
    except FileNotFoundError:
        results = {}

    # Update results
    results[run] = run_results

    # Write results
    with open(output_directory / 'results.json', 'w') as file:
        json.dump(results, file, indent=4)


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
        '--run',
        default='clpcnet',
        help='The evaluation run')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=clpcnet.DEFAULT_CHECKPOINT,
        help='The model checkpoint')
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='The index of the gpu to use')
    return parser.parse_args()


if __name__ == '__main__':
    evaluate(**vars(parse_args()))
