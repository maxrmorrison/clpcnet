import numpy as np
import soundfile

import clpcnet


def audio(file):
    """Load audio from disk

    Arguments
        file : string
            The audio file to load

    Returns
        audio : np.array(shape=(samples,))
            The audio
    """
    # Load
    audio, sample_rate = soundfile.read(file)

    # Convert to mono if necessary
    if audio.ndim == 2:
        if audio.shape[1] == 2:
            audio = audio.mean(1)
        else:
            audio = audio.squeeze()

    # Resample
    return clpcnet.preprocess.resample(audio, sample_rate)


def features(file):
    """Load frame-rate features from disk for inference

    Arguments
        file : string
            The feature file

    Returns
        features : np.array(shape=(frames, clpcnet.TOTAL_FEATURE_SIZE))
    """
    # Load test features
    features = np.fromfile(file, dtype=np.float32)

    # shape=(time, channels)
    features = np.reshape(features, (-1, clpcnet.TOTAL_FEATURE_SIZE))

    # Zero-out unused bark-scale coefficients
    features[:, 18:36] = 0
    return features[None]


def model(file=clpcnet.DEFAULT_CHECKPOINT, gpu=None):
    """Setup the LPCNet model for training

    Arguments
        file : string
            The model weight file
        use_gpu : bool
            Whether to use gpu compute
    """
    # Bind to generate function
    clpcnet.from_features.session = clpcnet.Session(file, gpu)


def yin(file):
    """Load yin pitch and periodicity from file"""
    # Load features
    yin_features = features(file)

    # Slice yin pitch and periodicity
    return yin_features[0, :, clpcnet.PITCH_IDX], \
           yin_features[0, :, clpcnet.CORRELATION_IDX]
