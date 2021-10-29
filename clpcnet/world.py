import numpy as np
import pypar
import pyworld
import scipy
import soundfile
import torch

import clpcnet


###############################################################################
# WORLD constants
###############################################################################


ALLOWED_RANGE = .8


###############################################################################
# Pitch-shifting and time-stretching with WORLD
###############################################################################


def from_audio(audio,
               source_alignment=None,
               target_alignment=None,
               target_pitch=None,
               constant_stretch=None,
               constant_shift=None):
    """Pitch-shifting and time-stretching with WORLD"""
    # World parameterization
    audio = audio.squeeze().numpy()
    pitch, spectrogram, aperiodicity = analyze(audio)

    # Variable-ratio pitch-shifting
    if target_pitch is not None:
        target_pitch = target_pitch.squeeze().numpy()

        if (len(target_pitch) != len(pitch) and
            source_alignment is None and
            target_alignment is None):
            raise ValueError(
                f'Source pitch of length {len(pitch)} incompatible ' +
                f'with target pitch of length {len(target_pitch)}.')
        pitch = target_pitch.astype(np.float64)

    # Constant-ratio pitch-shifting
    if constant_shift is not None:
        pitch *= constant_shift

    # Variable-ratio time-stretching
    if source_alignment is not None and target_alignment is not None:

        # Align spectrogram and aperiodicity
        spectrogram = clpcnet.pitch.align(None, spectrogram, target_alignment, source_alignment)
        aperiodicity = clpcnet.pitch.align(None, aperiodicity, target_alignment, source_alignment)

    # Constant-ratio time-stretching
    if constant_stretch is not None:

        # Get new duration
        duration = len(audio) / clpcnet.SAMPLE_RATE / constant_stretch

        # Stretch features
        pitch, spectrogram, aperiodicity = linear_time_stretch(
            pitch, spectrogram, aperiodicity, duration)

    # Synthesize using modified parameters
    vocoded = pyworld.synthesize(pitch,
                                 spectrogram,
                                 aperiodicity,
                                 clpcnet.SAMPLE_RATE,
                                 clpcnet.HOPSIZE / clpcnet.SAMPLE_RATE * 1000.)

    # Trim zero padding
    return vocoded


def from_file_to_file(input_file,
                      output_file,
                      source_alignment_file=None,
                      target_alignment_file=None,
                      target_pitch_file=None,
                      constant_stretch=None,
                      constant_shift=None):
    """Perform pitch-shifting and time-stretching with WORLD on files"""
    source = torch.tensor(clpcnet.load.audio(input_file))[None]

    # Load source alignment
    if source_alignment_file is not None:
        source_alignment = pypar.Alignment(source_alignment_file)
    else:
        source_alignment = None

    # Load target alignment
    if target_alignment_file is not None:
        target_alignment = pypar.Alignment(target_alignment_file)
    else:
        target_alignment = None

    # Load target pitch
    if target_pitch_file is not None:
        target_pitch = torch.tensor(np.load(target_pitch_file))[None]
    else:
        target_pitch = None

    to_file(source,
            output_file,
            source_alignment,
            target_alignment,
            target_pitch,
            constant_stretch,
            constant_shift)


def to_file(source,
            output_file,
            source_alignment=None,
            target_alignment=None,
            target_pitch=None,
            constant_stretch=None,
            constant_shift=None):
    """Perform pitch-shifting and time-stretching with WORLD and save"""
    vocoded = from_audio(source,
                         source_alignment,
                         target_alignment,
                         target_pitch,
                         constant_stretch,
                         constant_shift)
    soundfile.write(output_file, vocoded, clpcnet.SAMPLE_RATE)


###############################################################################
# Vocoding utilities
###############################################################################


def analyze(audio):
    """Convert an audio signal to WORLD parameter representation
    Arguments
        audio : np.array(shape=(samples,))
            The audio being analyzed
    Returns
        pitch : np.array(shape=(frames,))
            The pitch contour
        spectrogram : np.array(shape=(frames, channels))
            The audio spectrogram
        aperiodicity : np.array(shape=(frames,))
            The voiced/unvoiced confidence
    """
    # Cast to double
    audio = audio.astype(np.float64)

    # Hopsize in milliseconds
    frame_period = clpcnet.HOPSIZE / clpcnet.SAMPLE_RATE * 1000.

    # Pitch
    pitch, time = pyworld.dio(audio,
                              clpcnet.SAMPLE_RATE,
                              frame_period=frame_period,
                              f0_floor=clpcnet.FMIN,
                              f0_ceil=clpcnet.FMAX,
                              allowed_range=ALLOWED_RANGE)
    pitch = pyworld.stonemask(audio, pitch, time, clpcnet.SAMPLE_RATE)

    # Spectrogram
    spectrogram = pyworld.cheaptrick(audio, pitch, time, clpcnet.SAMPLE_RATE)

    # Aperiodicity
    aperiodicity = pyworld.d4c(audio, pitch, time, clpcnet.SAMPLE_RATE)

    return pitch, spectrogram, aperiodicity


def linear_time_stretch(prev_pitch,
                        prev_spectrogram,
                        prev_aperiodicity,
                        duration):
    """Apply time stretch in WORLD parameter space
    Arguments
        prev_pitch : np.array(shape=(frames,))
            The pitch to be stretched
        prev_spectrogram : np.array(shape=(frames, frequencies))
            The spectrogram to be stretched
        prev_aperiodicity : np.array(shape=(frames, frequencies))
            The aperiodicity to be stretched
        duration : float
            The new duration in seconds
    """
    # Number of frames before and after
    prev_frames = len(prev_pitch)
    next_frames = clpcnet.convert.seconds_to_frames(duration)

    # Time-aligned grid before and after
    prev_grid = np.linspace(0, prev_frames - 1, prev_frames)
    next_grid = np.linspace(0, prev_frames - 1, next_frames)

    # Apply time stretch to pitch
    pitch = linear_time_stretch_pitch(
        prev_pitch, prev_grid, next_grid, next_frames)

    # Allocate spectrogram and aperiodicity buffers
    frequencies = prev_spectrogram.shape[1]
    spectrogram = np.zeros((next_frames, frequencies))
    aperiodicity = np.zeros((next_frames, frequencies))

    # Apply time stretch to all channels of spectrogram and aperiodicity
    for i in range(frequencies):
        spectrogram[:, i] = scipy.interp(
            next_grid, prev_grid, prev_spectrogram[:, i])
        aperiodicity[:, i] = scipy.interp(
            next_grid, prev_grid, prev_aperiodicity[:, i])

    return pitch, spectrogram, aperiodicity


def linear_time_stretch_pitch(pitch, prev_grid, next_grid, next_frames):
    """Perform time-stretching on pitch features"""
    if (pitch == 0.).all():
        return np.zeros(next_frames)

    # Get unvoiced tokens
    unvoiced = pitch == 0.

    # Linearly interpolate unvoiced regions
    pitch[unvoiced] = np.interp(
        np.where(unvoiced)[0], np.where(~unvoiced)[0], pitch[~unvoiced])

    # Apply time stretch to pitch
    pitch = scipy.interp(next_grid, prev_grid, pitch)

    # Apply time stretch to unvoiced sequence
    unvoiced = scipy.interp(next_grid, prev_grid, unvoiced)

    # Reapply unvoiced tokens
    pitch[unvoiced > .5] = 0.

    return pitch
