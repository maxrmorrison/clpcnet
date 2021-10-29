import librosa
import numpy as np
import scipy

import clpcnet


###############################################################################
# Dtw alignment evaluation
###############################################################################


def from_files(source_files, target_files):
    """Evaluate dtw alignment score"""
    dtw = DtwAlignmentScore()

    # Evaluate each pair of files
    for source_file, target_file in zip(source_files, target_files):
        dtw.update(clpcnet.load.audio(source_file),
                   clpcnet.load.audio(target_file))

    # Compute aggregate dtw alignment score over files
    return dtw()


class DtwAlignmentScore:
    """Batch update dtw alignment score"""

    def __init__(self):
        self.distance_sum = 0.
        self.count = 0

    def __call__(self):
        """Return the rmse with the diagonal and the mean cosine distance"""
        distance = self.distance_sum / self.count
        return distance

    def update(self, source, target):
        """Compute the dtw alignment score"""
        # Compute mel features
        source_feats = features(source)
        target_feats = features(target)

        # Resample target features
        interp_fn = scipy.interpolate.interp1d(
            np.arange(target_feats.shape[1]),
            target_feats)
        target_feats_interp = interp_fn(
            np.linspace(0, target_feats.shape[1] - 1, source_feats.shape[1]))

        # Perform alignment
        D, path = librosa.sequence.dtw(target_feats_interp + 1e-10,
                                       source_feats + 1e-10,
                                       backtrack=True,
                                       metric='cosine')

        # Update metrics
        self.distance_sum += D[path[0, 0], path[0, 1]]
        self.count += len(path)


###############################################################################
# Utilities
###############################################################################


def features(audio):
    """Compute spectral features to use for dtw alignment"""
    # Compute mfcc without energy
    return librosa.feature.mfcc(audio, sr=clpcnet.SAMPLE_RATE)[1:]

