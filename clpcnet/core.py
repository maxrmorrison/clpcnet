import multiprocessing as mp

import numpy as np
import pypar
import soundfile
import tqdm

import clpcnet


__all__ = ['from_audio',
           'from_features',
           'from_file',
           'from_file_to_file',
           'from_files_to_files',
           'to_file']


###############################################################################
# Vocode
###############################################################################


def from_audio(audio,
               sample_rate=clpcnet.SAMPLE_RATE,
               source_alignment=None,
               target_alignment=None,
               constant_stretch=None,
               source_pitch=None,
               source_periodicity=None,
               target_pitch=None,
               constant_shift=None,
               checkpoint_file=clpcnet.DEFAULT_CHECKPOINT,
               gpu=None,
               verbose=True):
    """Pitch-shift and time-stretch speech audio

    Arguments
        audio : np.array(shape=(samples,))
            The audio to regenerate
        sample_rate : int
            The audio sampling rate
        source_alignment : pypar.Alignment or None
            The original alignment. Used only for time-stretching.
        target_alignment : pypar.Alignment or None
            The target alignment. Used only for time-stretching.
        constant_stretch : float or None
            A constant value for time-stretching
        source_pitch : np.array(shape=(1 + int(samples / hopsize))) or None
            The original pitch contour. Allows us to skip pitch estimation.
        source_periodicity : np.array(shape=(1 + int(samples / hopsize))) or None
            The original periodicity. Allows us to skip pitch estimation.
        target_pitch : np.array(shape=(1 + int(samples / hopsize))) or None
            The desired pitch contour
        constant_shift : float or None
            A constant value for pitch-shifting
        checkpoint_file : Path
            The model weight file
        gpu : int or None
            The gpu to run inference on. Defaults to cpu.
        verbose : bool
            Whether to display a progress bar

    Returns
        vocoded : np.array(shape=(samples * clpcnet.SAMPLE_RATE / sample_rate,))
            The generated audio at 16 kHz
    """
    # Resample audio
    audio = clpcnet.preprocess.resample(audio, sample_rate)

    # Require a minimum peak amplitude
    maximum = np.abs(audio).max()
    if maximum < .2:
        audio = audio / maximum * .4

    # Compute features
    features = clpcnet.preprocess.from_audio(audio)

    # Pitch and pitch ratio for each frame
    features, pitch_bins = per_frame_pitch(audio,
                                           clpcnet.SAMPLE_RATE,
                                           features,
                                           source_alignment,
                                           target_alignment,
                                           source_pitch,
                                           source_periodicity,
                                           target_pitch,
                                           constant_shift,
                                           gpu)

    # Generate audio from features
    generated = from_features(features,
                              pitch_bins,
                              source_alignment,
                              target_alignment,
                              constant_stretch,
                              checkpoint_file,
                              gpu,
                              verbose)

    # Scale to original peak amplitude if necessary
    if maximum < .2:
        generated = generated / .4 * maximum

    return generated


def from_features(features,
                  pitch_bins,
                  source_alignment=None,
                  target_alignment=None,
                  constant_stretch=None,
                  checkpoint_file=clpcnet.DEFAULT_CHECKPOINT,
                  gpu=None,
                  verbose=True):
    """Pitch-shift and time-stretch from features

    Arguments
        features : np.array(shape=(1, frames, clpcnet.SPECTRAL_FEATURE_SIZE))
            The frame-rate features
        pitch_bins : np.array(shape=(1 + int(samples / hopsize)))
            The pitch contour as integer pitch bins
        source_alignment : pypar.Alignment or None
            The original alignment. Used only for time-stretching.
        target_alignment : pypar.Alignment or None
            The target alignment. Used only for time-stretching.
        constant_stretch : float or None
            A constant value for time-stretching
        checkpoint_file : Path
            The model weight file
        gpu : int or None
            The gpu to run inference on. Defaults to cpu.
        verbose : bool
            Whether to display a progress bar

    Returns
        vocoded : np.array(shape=(frames * clpcnet.HOPSIZE,))
            The generated audio at 16 kHz
    """
    # Number of samples to generate for each frame
    hopsizes = per_frame_hopsizes(features,
                                  source_alignment,
                                  target_alignment,
                                  constant_stretch)

    # Create session and pre-load model
    if not hasattr(from_features, 'session') or \
       (from_features.session.gpu != gpu) or \
       (checkpoint_file is not None and
            from_features.session.file != checkpoint_file):
        clpcnet.load.model(checkpoint_file, gpu)

    # Setup tensorflow session
    with from_features.session.context():

        # Run frame-rate network
        frame_rate_feats = from_features.session.encoder.predict(
            [features[:, :, :clpcnet.SPECTRAL_FEATURE_SIZE], pitch_bins])

        # Run sample-rate network
        return decode(features, frame_rate_feats, hopsizes, verbose)


def from_file(audio_file,
              source_alignment_file=None,
              target_alignment_file=None,
              constant_stretch=None,
              source_pitch_file=None,
              source_periodicity_file=None,
              target_pitch_file=None,
              constant_shift=None,
              checkpoint_file=clpcnet.DEFAULT_CHECKPOINT,
              gpu=None,
              verbose=True):
    """Pitch-shift and time-stretch from file on disk

    Arguments
        audio_file : string
            The audio file
        source_alignment_file : Path or None
            The original alignment on disk. Used only for time-stretching.
        target_alignment_file : Path or None
            The target alignment on disk. Used only for time-stretching.
        constant_stretch : float or None
            A constant value for time-stretching
        source_pitch_file : Path or None
            The file containing the original pitch contour
        source_periodicity_file : Path or None
            The file containing the original periodicity
        target_pitch_file : Path or None
            The file containing the desired pitch
        constant_shift : float or None
            A constant value for pitch-shifting
        checkpoint_file : Path
            The model weight file
        gpu : int or None
            The gpu to run inference on. Defaults to cpu.
        verbose : bool
            Whether to display a progress bar

    Returns
        vocoded : np.array(shape=(samples,))
            The generated audio at 16 kHz
    """
    # Load audio
    audio = clpcnet.load.audio(audio_file)

    # Load phoneme alignments
    source_alignment = None if source_alignment_file is None \
                       else pypar.Alignment(source_alignment_file)
    target_alignment = None if target_alignment_file is None \
                       else pypar.Alignment(target_alignment_file)

    # Load pitch and periodicity
    source_pitch = None if source_pitch_file is None \
        else np.load(source_pitch_file)
    source_periodicity = None if source_periodicity_file is None \
        else np.load(source_periodicity_file)
    target_pitch = None if target_pitch_file is None \
        else np.load(target_pitch_file)

    # Generate audio
    return from_audio(audio,
                      clpcnet.SAMPLE_RATE,
                      source_alignment,
                      target_alignment,
                      constant_stretch,
                      source_pitch,
                      source_periodicity,
                      target_pitch,
                      constant_shift,
                      checkpoint_file,
                      gpu,
                      verbose)


def from_file_to_file(audio_file,
                      output_file,
                      source_alignment_file=None,
                      target_alignment_file=None,
                      constant_stretch=None,
                      source_pitch_file=None,
                      source_periodicity_file=None,
                      target_pitch_file=None,
                      constant_shift=None,
                      checkpoint_file=clpcnet.DEFAULT_CHECKPOINT,
                      gpu=None,
                      verbose=True):
    """Pitch-shift and time-stretch from file on disk and save to disk

    Arguments
        audio_file : Path
            The audio file
        output_file : Path
            The file to save the generated audio
        source_alignment_file : Path or None
            The original alignment on disk. Used only for time-stretching.
        target_alignment_file : Path or None
            The target alignment on disk. Used only for time-stretching.
        constant_stretch : float or None
            A constant value for time-stretching
        source_pitch : Path or None
            The file containing the original pitch contour
        source_periodicity_file : Path or None
            The file containing the original periodicity
        target_pitch_file : Path or None
            The file containing the desired pitch
        constant_shift : float or None
            A constant value for pitch-shifting
        checkpoint_file : Path
            The model weight file
        gpu : int or None
            The gpu to run inference on. Defaults to cpu.
        verbose : bool
            Whether to display a progress bar
    """
    # Generate
    audio = from_file(audio_file,
                      source_alignment_file,
                      target_alignment_file,
                      constant_stretch,
                      source_pitch_file,
                      source_periodicity_file,
                      target_pitch_file,
                      constant_shift,
                      checkpoint_file,
                      gpu,
                      verbose)

    # Write to disk
    soundfile.write(output_file, audio, clpcnet.SAMPLE_RATE)


def from_files_to_files(audio_files,
                        output_files,
                        source_alignment_files=None,
                        target_alignment_files=None,
                        constant_stretch=None,
                        source_pitch_files=None,
                        source_periodicity_files=None,
                        target_pitch_files=None,
                        constant_shift=None,
                        checkpoint_file=clpcnet.DEFAULT_CHECKPOINT):
    """Pitch-shift and time-stretch from files on disk and save to disk

    Arguments
        audio_files : list(Path)
            The audio files
        output_files : list(Path)
            The files to save the generated audio
        source_alignment_files : list(Path) or None
            The original alignments on disk. Used only for time-stretching.
        target_alignment_files : list(Path) or None
            The target alignments on disk. Used only for time-stretching.
        constant_stretch : float or None
            A constant value for time-stretching
        source_pitch_files : list(Path) or None
            The files containing the original pitch contours
        source_periodicity_files : list(Path) or None
            The files containing the original periodicities
        target_pitch_files : list(Path) or None
            The files containing the desired pitch contours
        constant_shift : float or None
            A constant value for pitch-shifting
        checkpoint_file : Path
            The model weight file
    """
    # Expand None-valued defaults
    if source_alignment_files is None:
        source_alignment_files = [None] * len(audio_files)
    if target_alignment_files is None:
        target_alignment_files = [None] * len(audio_files)
    if source_periodicity_files is None:
        source_periodicity_files = [None] * len(audio_files)
    if source_pitch_files is None:
        source_pitch_files = [None] * len(audio_files)
    if target_pitch_files is None:
        target_pitch_files = [None] * len(audio_files)

    # Setup multiprocessing
    pool = mp.get_context('spawn').Pool()

    # Setup iterator
    iterator = zip(audio_files,
                    output_files,
                    source_alignment_files,
                    target_alignment_files,
                    source_pitch_files,
                    source_periodicity_files,
                    target_pitch_files)
    for af, of, saf, taf, spif, spef, tpf in iterator:

        # Bundle arguments
        args = (af, of, saf, taf, constant_stretch, spif, spef,
                tpf, constant_shift, checkpoint_file, None, False)

        # Vocode and save to disk
        pool.apply_async(from_file_to_file, args)
        # from_file_to_file(*args)
    pool.close()
    pool.join()


def to_file(output_file,
            audio,
            sample_rate=clpcnet.SAMPLE_RATE,
            source_alignment=None,
            target_alignment=None,
            constant_stretch=None,
            source_pitch=None,
            source_periodicity=None,
            target_pitch=None,
            constant_shift=None,
            checkpoint_file=clpcnet.DEFAULT_CHECKPOINT,
            gpu=None,
            verbose=True):
    """Pitch-shift and time-stretch audio and save to disk

    Arguments
        output_file : Path
            The file to save the generated audio
        audio : np.array(shape=(samples,))
            The audio to regenerate
        sample_rate : int
            The audio sampling rate
        source_alignment : pypar.Alignment or None
            The original alignment. Used only for time-stretching.
        target_alignment : pypar.Alignment or None
            The target alignment. Used only for time-stretching.
        constant_stretch : float or None
            A constant value for time-stretching
        source_pitch : np.array(shape=(1 + int(samples / hopsize))) or None
            The original pitch contour. Allows us to skip pitch estimation.
        source_periodicity : np.array(shape=(1 + int(samples / hopsize))) or None
            The original periodicity. Allows us to skip pitch estimation.
        target_pitch : np.array(shape=(1 + int(samples / hopsize))) or None
            The desired pitch contour
        constant_shift : float or None
            A constant value for pitch-shifting
        checkpoint_file : Path
            The model weight file
        gpu : int or None
            The gpu to run inference on. Defaults to cpu.
        verbose : bool
            Whether to display a progress bar
    """
    # Generate with LPCNet
    generated = from_audio(audio,
                           sample_rate,
                           source_alignment,
                           target_alignment,
                           constant_stretch,
                           source_pitch,
                           source_periodicity,
                           target_pitch,
                           constant_shift,
                           checkpoint_file,
                           gpu,
                           verbose)

    # Write to disk
    soundfile.write(output_file, generated, clpcnet.SAMPLE_RATE)


###############################################################################
# Utilities
###############################################################################


def decode(features,
           frame_rate_feats,
           hopsizes,
           verbose=True):
    """Decode the output of the frame-rate network to samples"""
    # Output buffer
    total_samples = sum(hopsizes)
    audio = np.zeros(total_samples)
    # excitations = np.zeros(total_samples)

    # Intermediate buffers (128 corresponds to bin representing 0)
    samples = np.zeros(total_samples)
    current_sample_rate_feats = np.zeros((1, 1, 3), dtype='int16') + 128

    # Allocate gru hidden states
    gru_a_state = np.zeros((1, clpcnet.GRU_A_SIZE), dtype='float32')
    gru_b_state = np.zeros((1, clpcnet.GRU_B_SIZE), dtype='float32')

    # Assume last sample in preemphasis filter was 0
    preemphasis_buffer = 0

    # Skip the first LPC_ORDER samples
    samples_to_skip = clpcnet.LPC_ORDER + 1

    # Current sample index
    idx = samples_to_skip

    # Setup optional monitoring
    iterator = range(0, features.shape[1])
    if verbose:
        iterator = tqdm.tqdm(iterator,
                             desc='lpcnet - vocoding',
                             dynamic_ncols=True)
    frame = 0
    for _ in iterator:

        # Extract precomputed LPC coefficients
        lpc_coefficients = features[
            0, frame, clpcnet.TOTAL_FEATURE_SIZE - clpcnet.LPC_ORDER:]

        # Potentially skip short first frames
        if samples_to_skip >= hopsizes[frame]:
            samples_to_skip -= hopsizes[frame]
            continue

        for i in range(samples_to_skip, hopsizes[frame]):

            # Calculate LPC prediction
            start, end = idx - 1, idx - clpcnet.LPC_ORDER - 1
            lpc_prediction = -sum(lpc_coefficients * samples[start:end:-1])

            # Convert linear to mu-law
            current_sample_rate_feats[0, 0, 1] = \
                clpcnet.convert.linear_to_mulaw(lpc_prediction)

            # Decode to categorical distribution over sample values
            probabilities, gru_a_state, gru_b_state = \
                from_features.session.decoder.predict([
                    current_sample_rate_feats,
                    frame_rate_feats[:, frame:frame + 1, :],
                    gru_a_state,
                    gru_b_state])

            # Sample the excitation
            current_sample_rate_feats[0, 0, 2] = sample(
                probabilities,
                features[0, frame, clpcnet.CORRELATION_IDX])

            # Convert excitation from mulaw to linear
            excitation = clpcnet.convert.mulaw_to_linear(
                current_sample_rate_feats[0, 0, 2])

            # New sample is the prediction + excitation
            samples[idx] = lpc_prediction + excitation

            # Convert sample from linear to mulaw
            current_sample_rate_feats[0, 0, 0] = \
                clpcnet.convert.linear_to_mulaw(samples[idx])

            # Apply inverse pre-emphasis
            preemphasis_buffer = \
                clpcnet.PREEMPHASIS_COEF * preemphasis_buffer + samples[idx]

            # Write sample to file
            audio[idx] = np.round(preemphasis_buffer)
            idx += 1

        # Don't skip samples after first frame
        samples_to_skip = 0

        frame += 1

    # Convert to [-1., 1.]
    audio = audio / clpcnet.MAX_SAMPLE_VALUE

    # Normalize
    # Note: the epsilon is necessary to prevent clipping
    maximum = np.abs(audio).max()
    audio = audio / (maximum + 1e-4) if maximum >= 1. else audio

    # Remove padding
    return audio[clpcnet.HOPSIZE // 2:-clpcnet.HOPSIZE // 2]


def per_frame_hopsizes(features,
                       source_alignment,
                       target_alignment,
                       constant_stretch):
    """Compute hopsize of each frame based on alignments"""
    # Hopsizes without time-stretching
    if source_alignment is None or target_alignment is None:
        constant_stretch = 1. if constant_stretch is None else constant_stretch
        hopsize = int((1. / constant_stretch) * clpcnet.HOPSIZE)
        return [hopsize] * features.shape[1]

    # Hopsizes with time-stretching
    # Get rate difference between each phoneme
    rates = pypar.compare.per_frame_rate(
        source_alignment,
        target_alignment,
        clpcnet.SAMPLE_RATE,
        clpcnet.HOPSIZE)

    # Convert to value [0, 1] of progress through utterance
    progress = np.cumsum(np.array([0] + rates))
    progress /= progress[-1]

    # Interpolate to find precise indices of frame boundaries
    samples = clpcnet.convert.seconds_to_samples(target_alignment.duration())
    indices = np.interp(
        progress,
        np.linspace(0., 1., features.shape[1]),
        np.linspace(0., samples, features.shape[1])).astype(int)

    # Get hopsize per frame
    hopsizes = indices[1:] - indices[:-1]

    return hopsizes


def per_frame_pitch(audio,
                    sample_rate,
                    features,
                    source_alignment=None,
                    target_alignment=None,
                    source=None,
                    periodicity=None,
                    target=None,
                    constant_shift=None,
                    gpu=None):
    """Compute pitch and pitch shift ratio at each frame"""
    pitch_is_given = source is not None and periodicity is not None

    # Compute pitch if necessary
    if not pitch_is_given:
        source, periodicity = clpcnet.pitch.from_audio(audio, gpu)

    # Scale crepe periodicity to match range of yin
    periodicity = .8 * periodicity - .4

    # Replace periodicity
    features[0, :, clpcnet.CORRELATION_IDX] = periodicity

    # Default to no pitch shifting
    target = source.copy() if target is None else target

    # Handle pitch contours with mismatched length
    if len(target) != len(source):

        # Invert alignment on target pitch
        if source_alignment is not None and \
           target_alignment is not None:
            target = clpcnet.pitch.align(source,
                                         target,
                                         source_alignment,
                                         target_alignment)
        else:
            raise ValueError(
                f'Source pitch of length {len(source)} incompatible ' + \
                f'with target pitch of length {len(target)}.')

    # Apply constant-rate pitch-shifting
    if constant_shift is not None:
        target *= constant_shift

    # Upper bound pitch
    if clpcnet.FMAX is not None:
        source[source > clpcnet.FMAX] = clpcnet.FMAX
        target[target > clpcnet.FMAX] = clpcnet.FMAX

    # Lower bound pitch
    if clpcnet.FMIN is not None:
        source[source < clpcnet.FMIN] = clpcnet.FMIN
        target[target < clpcnet.FMIN] = clpcnet.FMIN

    # Replace pitch
    features[:, :, clpcnet.PITCH_IDX] = clpcnet.convert.hz_to_epochs(
        target)[None]

    # Extract pitch features for embedding
    if clpcnet.ABLATE_PITCH_REPR:
        # Non-uniform pitch resolution
        pitch_bins = clpcnet.convert.hz_to_length(target).reshape(1, -1, 1)

    else:
        # Uniform pitch resolution
        pitch_bins = clpcnet.convert.hz_to_bins(target).reshape(1, -1, 1)

    return features, pitch_bins


def sample(probabilities, periodicity, cutoff=.001):
    """Perform sampling on categorical probabilities made by neural network"""
    if clpcnet.ABLATE_SAMPLING:
        cutoff = .002
        probabilities *= \
            np.power(probabilities, np.maximum(0, 1.5 * periodicity - .5))

    # Normalize
    probabilities = probabilities / (1e-10 + np.sum(probabilities))

    # Cut off the tail of the distribution
    if not clpcnet.ABLATE_SAMPLING_TAIL:
        probabilities = np.maximum(probabilities - cutoff, 0)

    # Note: we must cast to double here as otherwise np.random.multinomial will
    #       complain about probabilities not adding to 1
    probabilities = probabilities.astype(np.float64)

    # Normalize
    probabilities = probabilities / (1e-10 + np.sum(probabilities))

    # Draw a sample (as a histogram)
    histogram = np.random.multinomial(1, probabilities[0, 0, :], 1)

    # Get the sample index
    return np.argmax(histogram)
