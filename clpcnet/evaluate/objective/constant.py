import argparse
import contextlib
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import pyfoal

import clpcnet


###############################################################################
# Constants
###############################################################################


DURATION_RATIOS = [50, 200]
PITCH_RATIOS = [71, 141]


###############################################################################
# Prosody evaluation
###############################################################################


def compute_and_update_condition(run,
                                 dataset,
                                 seen,
                                 condition,
                                 directory,
                                 source_pitch_files=None,
                                 target_pitch_files=None,
                                 source_duration_files=None,
                                 target_duration_files=None):
    """Compute and update metrics for a single condition"""
    results_file = directory / 'results.json'
    results_dict = {}

    # Evaluate pitch shifting
    if source_pitch_files is not None and target_pitch_files is not None:
        source_periodicity_files = [
            corresponding_file(str(f).replace('-pitch.npy', '.wav'),
                               '-periodicity.npy') for f in source_pitch_files]
        target_periodicity_files = [
            corresponding_file(str(f).replace('-pitch.npy', '.wav'),
                               '-periodicity.npy') for f in target_pitch_files]
        rmse, precision, recall, f1, gpe_20, gpe_50, hist = \
            clpcnet.evaluate.pitch.from_files(source_pitch_files,
                                              target_pitch_files,
                                              source_periodicity_files,
                                              target_periodicity_files)
        results_dict.update({
            'pitch': {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'rmse_cents': rmse,
                'gpe_20': gpe_20,
                'gpe_50': gpe_50}})
        hist_file = directory / 'conditions' / run / f'{condition}.png'
        clpcnet.evaluate.plot.write_histogram(hist_file, hist)

    # Evaluate time stretching
    if source_duration_files is not None and target_duration_files is not None:
        duration_dict = {}

        # Forced alignment rmse metric
        if dataset != 'daps':
            duration_rmse = clpcnet.evaluate.duration.from_files(
                source_duration_files, target_duration_files)
            duration_dict['force-align'] = {'rmse_seconds': duration_rmse}

        # DTW metrics
        source_audio_files = [f.with_suffix('.wav')
                              for f in source_duration_files]
        target_audio_files = []
        for file in target_duration_files:
            index = file.stem.find('-ts-')
            if index == -1:
                target_audio_file = file.with_suffix('.wav')
            else:
                target_audio_file = file.parent / f'{file.stem[:index]}.wav'
            target_audio_files.append(target_audio_file)
        distance = clpcnet.evaluate.dtw.from_files(source_audio_files,
                                                   target_audio_files)
        duration_dict['dtw'] = {'distance': distance}
        results_dict.update({'duration': duration_dict})

    # Update results
    update_results(results_file, run, seen, condition, results_dict)


def compute_metrics_and_update_results(run,
                                       directory,
                                       dataset,
                                       source_pitch_files,
                                       target_pitch_files,
                                       source_duration_files,
                                       target_duration_files):
    """Computes pitch and duration metrics and updates result json"""
    target_files = target_pitch_files + target_duration_files

    # Seen speakers
    if any(['-seen' in str(file.stem) for file in target_files]):
        compute_seen_or_unseen(run,
                               directory,
                               dataset,
                               True,
                               filter_files(source_pitch_files, '-seen'),
                               filter_files(target_pitch_files, '-seen'),
                               filter_files(source_duration_files, '-seen'),
                               filter_files(target_duration_files, '-seen'))

    # Unseen speakers
    if any(['-unseen' in str(file.stem) for file in target_files]):
        compute_seen_or_unseen(run,
                               directory,
                               dataset,
                               False,
                               filter_files(source_pitch_files, '-unseen'),
                               filter_files(target_pitch_files, '-unseen'),
                               filter_files(source_duration_files, '-unseen'),
                               filter_files(target_duration_files, '-unseen'))


def compute_seen_or_unseen(run,
                           directory,
                           dataset,
                           seen,
                           source_pitch_files,
                           target_pitch_files,
                           source_duration_files,
                           target_duration_files):
    """Compute metrics and update results for either seen or unseen examples"""
    # Unmodified results
    source_pitch = [f for f in source_pitch_files if '-ps-' not in f.stem]
    target_pitch = [f for f in target_pitch_files if '-ps-' not in f.stem]
    source_duration = [
        f for f in source_duration_files if '-ts-' not in f.stem]
    target_duration = [
        f for f in target_duration_files if '-ts-' not in f.stem]
    compute_and_update_condition(run,
                                 dataset,
                                 seen,
                                 'unmodified',
                                 directory,
                                 source_pitch,
                                 target_pitch,
                                 source_duration,
                                 target_duration)

    # Pitch shifting upwards
    source = [f for f in source_pitch_files if '-ps-141' in f.stem]
    target = [f for f in target_pitch_files if '-ps-141' in f.stem]
    compute_and_update_condition(run,
                                 dataset,
                                 seen,
                                 'ps-141',
                                 directory,
                                 source,
                                 target)

    # Pitch shifting downwards
    source = [f for f in source_pitch_files if '-ps-71' in f.stem]
    target = [f for f in target_pitch_files if '-ps-71' in f.stem]
    compute_and_update_condition(run,
                                 dataset,
                                 seen,
                                 'ps-71',
                                 directory,
                                 source,
                                 target)

    # Time-stretching faster
    source = [f for f in source_duration_files if '-ts-200' in f.stem]
    target = [f for f in target_duration_files if '-ts-200' in f.stem]
    compute_and_update_condition(run,
                                 dataset,
                                 seen,
                                 'ts-200',
                                 directory,
                                 source_duration_files=source,
                                 target_duration_files=target)

    # Time-stretching slower
    source = [f for f in source_duration_files if '-ts-50' in f.stem]
    target = [f for f in target_duration_files if '-ts-50' in f.stem]
    compute_and_update_condition(run,
                                 dataset,
                                 seen,
                                 'ts-50',
                                 directory,
                                 source_duration_files=source,
                                 target_duration_files=target)


def from_checkpoint(run,
                    directory,
                    checkpoint,
                    dataset,
                    skip_generation=False,
                    gpu=None):
    """Evaluate and save results to disk"""
    # Directory for output
    output_directory = directory / 'conditions' / run
    output_directory.mkdir(parents=True, exist_ok=True)

    if not skip_generation:

        # Input files
        input_directory = directory / 'data'
        stems = [file[:-4] for file in os.listdir(input_directory)
                 if file.endswith('wav')]

        # Setup multiprocessing
        pool = mp.Pool()

        # Generate all conditions
        for stem in stems:
            # Stem-specific I/O paths
            input_path = input_directory / stem
            output_path = output_directory / stem

            # Copy text file and periodicity
            if dataset != 'daps':
                shutil.copy2(f'{input_path}.txt', f'{output_path}.txt')

            # Launch generation without pitch-shift or time-stretch
            input_file = Path(f'{input_path}.wav')
            default_kwargs = {
                'source_pitch_file': Path(f'{input_path}-pitch.npy'),
                'source_periodicity_file': Path(f'{input_path}-periodicity.npy'),
                'checkpoint_file': checkpoint,
                'verbose': False}
            args = (input_file, Path(f'{output_path}.wav'))
            pool.apply_async(clpcnet.from_file_to_file, args, default_kwargs)

            # Launch time-stretching generation
            for ratio in DURATION_RATIOS:
                args = (input_file, Path(f'{output_path}-ts-{ratio}.wav'),)
                kwargs = {**default_kwargs,
                          **{'constant_stretch': ratio / 100.}}
                pool.apply_async(clpcnet.from_file_to_file, args, kwargs)
                # clpcnet.from_file_to_file(*args, **kwargs)

            # Launch pitch-shifting generation
            for ratio in PITCH_RATIOS:
                target_pitch = Path(f'{input_path}-ps-{ratio}-pitch.npy')
                args = (input_file, Path(f'{output_path}-ps-{ratio}.wav'))
                kwargs = {**default_kwargs,
                          **{'target_pitch_file': target_pitch}}
                pool.apply_async(clpcnet.from_file_to_file, args, kwargs)
                # clpcnet.from_file_to_file(*args, **kwargs)

        # Close the pool and wait until processes finish
        pool.close()
        pool.join()

    # Evaluate generated audio
    from_generated(run, directory, output_directory, dataset, gpu)


def from_generated(run,
                   evaluation_directory,
                   source_directory,
                   dataset,
                   gpu=None):
    """Compute metrics from previously generated audio"""
    # Get files to evaluate
    source_files = os.listdir(source_directory)
    pitch_files = [file for file in source_files
                   if not '-ts-' in file and file.endswith('wav')]
    duration_files = [file for file in source_files
                      if not '-ps-' in file and file.endswith('wav')]

    # Change directory to target directory
    with directory_context(source_directory):

        # Pitch-track original and pitch-shifted audio
        clpcnet.pitch.from_files_to_files(pitch_files,
                                          [file[:-4] for file in pitch_files],
                                          gpu)

        # Force align original and time-stretched audio
        if dataset != 'daps':
            pyfoal.from_files_to_files(
                [corresponding_file(file, '.txt') for file in duration_files],
                duration_files,
                [file.replace('wav', 'json') for file in duration_files])

    # Get pairs of files to compare for metrics
    target_directory = evaluation_directory / 'data'
    target_pitch_files = [
        target_directory / file.replace('.wav', '-pitch.npy')
        for file in pitch_files]
    target_duration_files = [
        target_directory / file.replace('.wav', '.json')
        for file in duration_files]
    source_pitch_files = [
        source_directory / file.replace('.wav', '-pitch.npy')
        for file in pitch_files]
    source_duration_files = [
        source_directory / file.replace('.wav', '.json')
        for file in duration_files]

    # Compute metrics on recovered pitch and duration
    compute_metrics_and_update_results(run,
                                       evaluation_directory,
                                       dataset,
                                       source_pitch_files,
                                       target_pitch_files,
                                       source_duration_files,
                                       target_duration_files)


###############################################################################
# Utilities
###############################################################################


def corresponding_file(file, extension):
    """Retrieve the corresponding file without condition info"""
    index = max(file.find('-ps-'), file.find('-ts-'))
    corresponding = file[:index] if index != -1 else file[:file.find('.wav')]
    return corresponding + extension


@contextlib.contextmanager
def directory_context(directory):
    """Current directory context management"""
    current = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(current)


def filter_files(files, keyword):
    """Return only files containing keyword"""
    return [file for file in files if keyword in str(file.stem)]


def update_results(file,
                   run,
                   seen,
                   condition,
                   results_dict):
    """Update results json"""
    # Read results
    try:
        with open(file) as results_file:
            results = json.load(results_file)
    except FileNotFoundError:
        results = {}

    # Update results
    if run not in results:
        results[run] = {}
    seen_str = f'{"" if seen else "un"}seen'
    if seen_str not in results[run]:
        results[run][seen_str] = {}
    results[run][seen_str][condition] = results_dict

    # Write results
    with open(file, 'w') as results_file:
        json.dump(results, results_file, indent=4)


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=clpcnet.DEFAULT_CHECKPOINT,
        help='The lpcnet checkpoint file')
    parser.add_argument(
        '--run',
        default='clpcnet',
        help='The name of the experiment')
    parser.add_argument(
        '--directory',
        type=Path,
        default=clpcnet.EVAL_DIR,
        help='The evaluation directory')
    parser.add_argument(
        '--dataset',
        default='vctk',
        help='The dataset to evaluate')
    parser.add_argument(
        '--skip_generation',
        action='store_true',
        help='Whether to skip generation and eval a previously generated run')
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='The gpu to run pitch tracking on')

    # Extend directories with dataset name
    args = parser.parse_args()
    args.directory = args.directory / 'objective' / 'constant' / args.dataset

    return args


if __name__ == '__main__':
    from_checkpoint(**vars(parse_args()))

