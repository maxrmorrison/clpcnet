import math

import numpy as np

import clpcnet


###############################################################################
# Constants
###############################################################################


# Define these explicitly, as they are used millions of times
INV_LOG_256 = 1. / math.log(256)
LOG_256_OVER_128 = math.log(256) / 128.
LI_TO_MU_SCALE = (clpcnet.PCM_LEVELS - 1.) / clpcnet.MAX_SAMPLE_VALUE
MU_TO_LI_SCALE = clpcnet.MAX_SAMPLE_VALUE / (clpcnet.PCM_LEVELS - 1.)


###############################################################################
# Mulaw encoding and decoding
###############################################################################


def linear_to_mulaw(linear):
    """Mu-law encode signal"""
    # Convert from [-MAX_SAMPLE_VALUE + 1, MAX_SAMPLE_VALUE] to [-126.5, 126.5]
    linear *= LI_TO_MU_SCALE

    # Mu-law encode
    mulaw = np.sign(linear) * (128 * np.log(1 + np.abs(linear)) * INV_LOG_256)

    # Shift to [0, 255]
    mulaw = np.round(mulaw) + 128

    # Clip
    return np.clip(mulaw, 0, 255).astype(np.int16)


def mulaw_to_linear(mulaw):
    """Decode mu-law signal"""
    # Zero-center
    mulaw = mulaw.astype(np.int16) - 128

    # Convert to linear
    linear = np.sign(mulaw) * (np.exp(np.abs(mulaw) * LOG_256_OVER_128) - 1)

    # Scale to [-MAX_SAMPLE_VALUE + 1, MAX_SAMPLE_VALUE]
    return linear * MU_TO_LI_SCALE


###############################################################################
# Pitch representations
###############################################################################


def bins_to_hz(bins,
               fmin=clpcnet.FMIN,
               fmax=clpcnet.FMAX,
               pitch_bins=clpcnet.PITCH_BINS):
    logmin, logmax = np.log2(fmin), np.log2(fmax)

    # Scale to base-2 log-space
    loghz = bins.astype(float) / (pitch_bins - 1) * (logmax - logmin) + logmin

    # Convert to hz
    return 2 ** loghz


def epochs_to_bins(epochs, sample_rate=clpcnet.SAMPLE_RATE):
    """Convert pitch in normalized pitch epochs to quantized bins"""
    return bins_to_hz(epochs_to_hz(epochs, sample_rate))


def epochs_to_hz(epochs, sample_rate=clpcnet.SAMPLE_RATE):
    """Convert pitch in normalized pitch epochs to Hz"""
    return sample_rate / (50 * epochs + 100.1)


def epochs_to_length(epochs):
    """Convert normalized pitch epochs to samples per period"""
    return (50 * epochs + 100.1).astype('int16')


def hz_to_bins(hz,
               fmin=clpcnet.FMIN,
               fmax=clpcnet.FMAX,
               pitch_bins=clpcnet.PITCH_BINS):
    logmin, logmax = np.log2(fmin), np.log2(fmax)

    # Clip pitch in base-2 log-space
    loghz = np.clip(np.log2(hz), logmin, logmax)

    # Scale to 0, 255
    return \
        ((loghz - logmin) / (logmax - logmin) * (pitch_bins - 1)).astype(int)


def hz_to_epochs(hz, sample_rate=clpcnet.SAMPLE_RATE):
    """Convert pitch in Hz to normalized epochs"""
    return (sample_rate / hz - 100.1) / 50.


def hz_to_length(hz, sample_rate=clpcnet.SAMPLE_RATE):
    """Convert pitch in Hz to number of samples per period"""
    return (sample_rate / hz).astype('int16')


def length_to_epochs(length):
    """Convert pitch in number of samples per period to normalized epochs"""
    return (length - 100.1) / 50.


def length_to_hz(length, sample_rate=clpcnet.SAMPLE_RATE):
    """Convert pitch in number of samples per period to Hz"""
    return sample_rate / length


def seconds_to_frames(seconds):
    """Convert time in seconds to number of frames"""
    return 1 + int(seconds * clpcnet.SAMPLE_RATE / clpcnet.HOPSIZE)


def seconds_to_samples(seconds):
    """Convert time in seconds to number of samples"""
    return clpcnet.HOPSIZE * seconds_to_frames(seconds)
