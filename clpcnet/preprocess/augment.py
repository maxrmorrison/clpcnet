import argparse
import multiprocessing as mp
import os
import random
from pathlib import Path

import numpy as np
import soundfile
import tqdm

import clpcnet


###############################################################################
# Constants
###############################################################################


ALLOWED_SCALES = [50, 67, 75, 80, 125, 133, 150, 200]
DATASET = 'vctk'
PASSES = 8


###############################################################################
# Data augmentation
###############################################################################


def dataset(dataset=DATASET,
            directory=clpcnet.CACHE_DIR / DATASET,
            cache=clpcnet.DATA_DIR / DATASET,
            allowed_scales=ALLOWED_SCALES,
            passes=PASSES,
            gpu=None):
    """Perform data augmentation for a given dataset"""
    # Compute the current histogram from pitch files in cache and determine
    # for each example which scales have been used
    counts, scales = count_cache(cache)

    import pdb; pdb.set_trace()

    # Get list of audio files
    files = clpcnet.data.files(dataset, directory, 'train')
    random.seed(0)
    random.shuffle(files)

    # Preprocessing workers
    feature_pool = mp.Pool(min(os.cpu_count() - 1, 2))
    pitch_pool = mp.Pool(1)

    # Iterate over dataset
    for i in range(passes):
        iterator = tqdm.tqdm(files,
                             dynamic_ncols=True,
                             desc=f'augmentation pass {i}')
        for file in iterator:

            # Load pitch
            stem = clpcnet.data.file_to_stem(dataset, file)
            pitch = np.load(cache / f'{stem}-r100-pitch.npy')
            periodicity = np.load(cache / f'{stem}-r100-periodicity.npy')

            # Threshold pitch
            pitch = clpcnet.pitch.threshold(pitch, periodicity)

            # Select scale to use that maximizes entropy
            scale, counts = select_scale(pitch[~np.isnan(pitch)],
                                        counts,
                                        allowed_scales,
                                        scales[stem])

            # No unused scale for this file
            if scale is None:
                continue

            # Load audio
            audio, sample_rate = soundfile.read(file)

            # Scale audio
            scaled = clpcnet.preprocess.resample(audio,
                                                 (scale / 100.) * sample_rate,
                                                 sample_rate)

            # Resample to lpcnet sample rate
            scaled = clpcnet.preprocess.resample(scaled, sample_rate)

            # Preprocess
            prefix = f'{cache / stem}-r{scale:03}'
            feature_pool.apply_async(clpcnet.preprocess.from_audio_to_file,
                                     (scaled, prefix))
            pitch_pool.apply_async(clpcnet.pitch.from_audio_to_file,
                                   (scaled, prefix, gpu))
            # clpcnet.pitch.from_audio_to_file(scaled, prefix, gpu)

            # Mark scale as used
            scales[stem].append(scale)

    # Close worker pools
    feature_pool.close()
    pitch_pool.close()

    # Wait for preprocessing to finish
    feature_pool.join()
    pitch_pool.join()


###############################################################################
# Utilities
###############################################################################


def count_cache(cache):
    """Compute pitch histogram and used scales of examples in cache"""
    counts = np.zeros(clpcnet.PITCH_BINS, dtype=int)
    scales = {}

    # Loop over pitch files
    for file in cache.rglob('*-pitch.npy'):

        # Load pitch
        pitch = np.load(file)
        periodicity = np.load(str(file).replace('-pitch.npy',
                                                '-periodicity.npy'))

        # Add pitch to histogram
        counts += count_pitch(clpcnet.pitch.threshold(pitch, periodicity))

        # Add scale to used set
        stem = file.stem[:-11]
        if stem not in scales:
            scales[stem] = []
        scales[stem].append(int(file.stem[-9:-6]))

    return counts, scales


def count_pitch(pitch):
    """Compute pitch histogram on pitch in Hz"""
    bins = clpcnet.convert.hz_to_bins(pitch[~np.isnan(pitch)])
    return np.bincount(bins, minlength=clpcnet.PITCH_BINS)


def entropy(counts):
    """Compute the entropy of the categorical distribution defined by counts"""
    # Compute categorical distribution parameters
    distribution = counts / counts.sum(keepdims=True)

    # Compute entropy contribution of each category
    contribution = distribution * np.log2(distribution)
    contribution[np.isnan(contribution)] = 0.

    return - (1. / np.log2(len(distribution))) * contribution.sum()


def scale_pitch(pitch, scale):
    """Scale pitch by scale factor"""
    # Scale
    scale_min = clpcnet.FMIN / pitch.min()
    scale_max = clpcnet.FMAX / pitch.max()
    scale = scale_min if scale < scale_min else scale
    scale = scale_max if scale > scale_max else scale
    pitch = scale * pitch.copy()

    # Interpolate
    scaled = np.interp(np.arange(0, len(pitch), scale),
                       np.arange(len(pitch)),
                       pitch)

    return scaled, int(100 * scale)


def select_scale(pitch, counts, allowed_scales, used_scales):
    """
    Shift the pitch by all allowed scales. If scale causes pitch to be
    outside (50, 550), use the closest scale that keeps pitch in this range.
    Do not use scale values that have already been used for this file.
    """
    best_entropy, best_scale = None, None
    for scale in set(allowed_scales) - set(used_scales):

        # Scale pitch
        scaled, scale = scale_pitch(pitch, scale / 100.)

        # If scale was clipped, make sure we can still use it
        if scale in used_scales:
            continue

        # Get pitch histogram
        scale_counts = counts + count_pitch(scaled)

        # Measure entropy for this scale
        scale_entropy = entropy(scale_counts)

        # Select scale if it maximizes entropy
        if best_entropy is None or \
           (best_entropy is not None and scale_entropy > best_entropy):
            best_entropy, best_scale = scale_entropy, scale
            counts = scale_counts

    return best_scale, counts


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        default=DATASET,
        help='The name of the dataset')
    parser.add_argument(
        '--directory',
        default=clpcnet.DATA_DIR,
        type=Path,
        help='The data directory')
    parser.add_argument(
        '--cache',
        default=clpcnet.CACHE_DIR,
        type=Path,
        help='The cache directory')
    parser.add_argument(
        '--allowed_scales',
        nargs='+',
        type=float,
        default=ALLOWED_SCALES,
        help='The allowable scale values for resampling')
    parser.add_argument(
        '--passes',
        type=int,
        default=PASSES,
        help='The number of augmentation passes to make over the dataset')
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='The index of the gpu to use')

    # Extend directories with dataset name
    args = parser.parse_args()
    args.directory = args.directory / args.dataset
    args.cache = args.cache / args.dataset

    return args


if __name__ == '__main__':
    dataset(**vars(parse_args()))
