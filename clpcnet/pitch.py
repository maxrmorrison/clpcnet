import argparse
import tempfile
from pathlib import Path

import numpy as np
import pypar
import torch
import torchcrepe
import tqdm

import clpcnet


###############################################################################
# Pitch methods
###############################################################################


def crepe(audio, gpu=None):
    """Preprocess crepe pitch from audio"""
    # Highpass
    audio = clpcnet.preprocess.highpass(audio)

    # Convert to torch
    audio = torch.tensor(audio.copy(), dtype=torch.float)[None]

    # Estimate pitch
    pitch, periodicity = torchcrepe.predict(
        audio,
        sample_rate=clpcnet.SAMPLE_RATE,
        hop_length=clpcnet.HOPSIZE,
        fmin=clpcnet.FMIN,
        fmax=clpcnet.FMAX,
        model='full',
        return_periodicity=True,
        batch_size=1024,
        device='cpu' if gpu is None else f'cuda:{gpu}')

    # Set low energy frames to unvoiced
    loudness = torchcrepe.loudness.a_weighted(
        audio,
        clpcnet.SAMPLE_RATE,
        clpcnet.HOPSIZE).to(pitch.device)
    periodicity[loudness < -60.] = 0.

    # Detach from graph
    pitch = pitch.cpu().squeeze().numpy()
    periodicity = periodicity.cpu().squeeze().numpy()

    return pitch, periodicity


def yin(audio):
    """Preprocess yin pitch from audio"""
    with tempfile.TemporaryDirectory() as directory:
        prefix = Path(directory) / 'tmp'

        # Preprocess and save to disk
        clpcnet.preprocess.from_audio_to_file(audio, prefix)

        # Load features
        features = clpcnet.load.features(f'{prefix}-frames.f32')

        # Extrect pitch and periodicity
        pitch = features[0, :, clpcnet.PITCH_IDX]
        periodicity = features[0, :, clpcnet.CORRELATION_IDX]

        # Convert to hz
        pitch = clpcnet.convert.epochs_to_hz(pitch)

        # Bound
        pitch[pitch > clpcnet.FMAX] = clpcnet.FMAX
        pitch[pitch < clpcnet.FMIN] = clpcnet.FMIN

        # Scale periodicity to [0, 1]
        return pitch, (periodicity + .4) / .8


###############################################################################
# Interface
###############################################################################


def from_audio(audio, gpu=None):
    """Preprocess pitch from audio"""
    if clpcnet.ABLATE_CREPE:
        return yin(audio)
    return crepe(audio, gpu)


def from_audio_to_file(audio, prefix, gpu=None):
    """Perform pitch estimation on audio and save to disk"""
    # Perform pitch estimation
    pitch, periodicity = from_audio(audio, gpu)

    # Save to disk
    np.save(f'{prefix}-pitch.npy', pitch)
    np.save(f'{prefix}-periodicity.npy', periodicity)


def from_dataset_to_files(dataset,
                          directory,
                          cache,
                          gpu=None):
    """Perform pitch estimation on dataset and save to disk"""
    # Get filenames
    files = clpcnet.data.files(dataset, directory, 'train')

    # Get prefixes
    prefixes = [
        cache / f'{clpcnet.data.file_to_stem(dataset, file)}-r100'
        for file in files]

    # Perform pitch estimation
    from_files_to_files(files, prefixes, gpu)


def from_file(file, gpu=None):
    """Preprocess crepe pitch from file"""
    # Load and estimate pitch
    return from_audio(clpcnet.load.audio(file), gpu)


def from_file_to_file(file, prefix, gpu=None):
    """Preprocess crepe pitch from file and save to disk"""
    pitch, periodicity = from_file(file, gpu)
    prefix.parent.mkdir(exist_ok=True, parents=True)
    np.save(f'{prefix}-pitch.npy', pitch)
    np.save(f'{prefix}-periodicity.npy', periodicity)


def from_files_to_files(files, prefixes, gpu=None):
    """Preprocess pitch from files and save to disk"""
    iterator = zip(files, prefixes)
    iterator = tqdm.tqdm(iterator, desc='pitch estimation', dynamic_ncols=True)
    for file, prefix in iterator:
        from_file_to_file(file, prefix, gpu)


###############################################################################
# Utilities
###############################################################################


def align(source, target, source_alignment, target_alignment):
    """Align target pitch with source by inverting the alignment"""
    # Get relative rates for each frame
    rates = pypar.compare.per_frame_rate(source_alignment,
                                         target_alignment,
                                         clpcnet.SAMPLE_RATE,
                                         clpcnet.HOPSIZE)

    # Get interpolation indices
    indices = np.cumsum(np.array(rates))

    # Interpolate
    return np.interp(indices, np.arange(len(target)), target)


def threshold(pitch, periodicity):
    """Threshold pitch via periodicity contour"""
    return torchcrepe.threshold.Hysteresis()(
        torch.tensor(pitch)[None],
        torch.tensor(periodicity)[None]).squeeze().numpy()


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='vctk',
        help='The dataset to perform pitch tracking on')
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
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='The gpu to use for pitch tracking')

    # Extend directories with dataset name
    args = parser.parse_args()
    args.directory = args.directory / args.dataset
    args.cache = args.cache / args.dataset

    return args


if __name__ == '__main__':
    from_dataset_to_files(**vars(parse_args()))
