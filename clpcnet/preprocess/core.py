import multiprocessing as mp
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import resampy
import scipy

import clpcnet


__all__ = ['from_audio',
           'from_audio_to_file',
           'from_dataset_to_files',
           'from_file_to_file',
           'from_files_to_files',
           'clip',
           'highpass',
           'pad',
           'preemphasis',
           'resample']


###############################################################################
# Preprocessing transforms
###############################################################################


def clip(audio, threshold=.99):
    """Normalize audio"""
    maximum = np.abs(audio).max()
    return audio * threshold / maximum if maximum > threshold else audio


def highpass(audio, sample_rate=clpcnet.SAMPLE_RATE, cutoff=65., order=5):
    """Highpass audio"""
    # Get filter coefficients
    b, a = scipy.signal.butter(
        order, cutoff / (sample_rate / 2), btype='high')

    # Filter
    return scipy.signal.filtfilt(b, a, audio)


def pad(audio):
    """Pad the audio to be a multiple of the block size"""
    padding = 2 * clpcnet.BLOCK_SIZE - (audio.size % clpcnet.BLOCK_SIZE)
    return np.pad(audio, (clpcnet.HOPSIZE // 2, padding))


def preemphasis(audio, coefficient=clpcnet.PREEMPHASIS_COEF):
    """Apply preemphasis filter"""
    result = np.zeros_like(audio)
    memory = 0.
    for i in range(len(audio)):
        result[i] = audio[i] + memory
        memory = -coefficient * audio[i]
    return result


def resample(audio, sample_rate, target_rate=clpcnet.SAMPLE_RATE):
    """Resample audio"""
    if sample_rate != target_rate:
        return resampy.resample(audio, sample_rate, target_rate)
    return audio


###############################################################################
# Preprocess data
###############################################################################


def from_dataset_to_files(dataset, directory, cache):
    """Preprocess dataset"""
    # Get filenames
    files = clpcnet.data.files(dataset, directory, 'train')

    # Get prefixes
    prefixes = [
        cache / f'{clpcnet.data.file_to_stem(dataset, file)}-r100'
        for file in files]

    # Create cache
    cache.mkdir(exist_ok=True, parents=True)

    # Preprocess from joined audio
    clpcnet.preprocess.from_files_to_files(files, prefixes)


def from_audio(audio):
    """Preprocess audio"""
    # Preprocess to a file in a temporary directory
    with tempfile.TemporaryDirectory() as directory:
        prefix = Path(directory) / 'features'

        # Preprocess
        from_audio_to_file(audio, prefix)

        # Load features
        return clpcnet.load.features(f'{prefix}-frames.f32')


def from_audio_to_file(audio, prefix):
    """Preprocess audio and save to disk"""
    # Get number of frames before padding
    frames = 1 + int(len(audio) // clpcnet.HOPSIZE)

    # Transform
    audio = clpcnet.loudness.limit(preemphasis(highpass(pad(audio))))

    # Convert to 16-bit int
    audio = (audio * clpcnet.MAX_SAMPLE_VALUE).astype(np.int16)

    # Write audio to temporary storage and preprocess
    with tempfile.TemporaryDirectory() as directory:
        file = Path(directory) / 'audio.s16'

        # Save to disk
        audio.tofile(file)

        # Preprocess from file
        from_binary_file_to_file(file, prefix, frames)


def from_file_to_file(file, prefix):
    """Load, preprocess, and save to disk"""
    from_audio_to_file(clpcnet.load.audio(file), prefix)


def from_files_to_files(files, prefixes):
    """Load, preprocess, and save many files"""
    with mp.Pool() as pool:
        pool.starmap(from_file_to_file, zip(files, prefixes))


###############################################################################
# Utilities
###############################################################################


def from_binary_file_to_file(file, prefix, frames):
    """Preprocess from binary s16 file"""
    # Write intermediate output to temporary file
    with tempfile.TemporaryDirectory() as directory:
        frame_file = f'{directory}-frames.f32'
        sample_file = f'{directory}-samples.u8'

        # Preprocess in C
        args = [str(Path(__file__).parent.parent.parent / 'bin' / 'preprocess'),
                str(file),
                frame_file,
                sample_file]
        subprocess.Popen(args).wait()

        # Truncate to original number of frames
        features = np.fromfile(frame_file, dtype=np.float32)
        features = features[:frames * clpcnet.TOTAL_FEATURE_SIZE]
        features.tofile(f'{prefix}-frames.f32')
        samples = np.fromfile(sample_file, dtype=np.uint8)
        samples = samples[:4 * frames * clpcnet.HOPSIZE]
        samples.tofile(f'{prefix}-samples.u8')
