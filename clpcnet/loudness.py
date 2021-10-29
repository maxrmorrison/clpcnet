import warnings

import librosa
import numpy as np

import clpcnet


###############################################################################
# A-weighted loudness
###############################################################################


def a_weighted(audio, n_fft=1024, min_db=-100.):
    """Retrieve the per-frame loudness"""
    # Cache weights so long as n_fft doesn't change
    if not hasattr(a_weighted, 'weights') or \
            (hasattr(a_weighted, 'n_fft') and a_weighted.n_fft != n_fft):
        a_weighted.weights = perceptual_weights(n_fft)
        a_weighted.n_fft = n_fft

    # Take stft
    stft = librosa.stft(audio,
                        n_fft,
                        hop_length=clpcnet.HOPSIZE,
                        win_length=n_fft,
                        pad_mode='constant')

    # Compute magnitude on db scale
    db = librosa.amplitude_to_db(np.abs(stft))

    # Apply A-weighting
    weighted = db + a_weighted.weights

    # Threshold
    weighted[weighted < min_db] = min_db

    # Average over weighted frequencies
    return weighted.mean(axis=0)


def perceptual_weights(n_fft=1024, ref_db=20.):
    """A-weighted frequency-dependent perceptual loudness weights"""
    frequencies = librosa.fft_frequencies(sr=clpcnet.SAMPLE_RATE, n_fft=n_fft)

    # A warning is raised for nearly inaudible frequencies, but it ends up
    # defaulting to -100 db. That default is fine for our purposes.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return librosa.A_weighting(frequencies)[:, None] - ref_db


###############################################################################
# Utilities
###############################################################################


def limit(audio, delay=40, attack_coef=.9, release_coef=.9995, threshold=.99):
    """Apply a limiter to prevent clipping"""
    # Delay compensation
    audio = np.pad(audio, (0, delay - 1))

    current_gain = 1.
    delay_index = 0
    delay_line = np.zeros(delay)
    envelope = 0

    for idx, sample in enumerate(audio):
        # Update signal history
        delay_line[delay_index] = sample
        delay_index = (delay_index + 1) % delay

        # Calculate envelope
        envelope = max(abs(sample), envelope * release_coef)

        # Calcuate gain
        target_gain = threshold / envelope if envelope > threshold else 1.
        current_gain = \
            current_gain * attack_coef + target_gain * (1 - attack_coef)

        # Apply gain
        audio[idx] = delay_line[delay_index] * current_gain

    return audio[delay - 1:]
