import numpy as np
import pypar


###############################################################################
# Duration evaluation
###############################################################################


def from_files(source_files, target_files):
    """Evaluate phoneme duration rmse"""
    rmse = DurationRmse()

    # Evaluate each pair of files
    for source_file, target_file in zip(source_files, target_files):
        source = pypar.Alignment(source_file)
        target = pypar.Alignment(target_file)
        rmse.update(source, target)

    # Compute aggregate rmse over files
    return rmse()


class DurationRmse:
    """Batch update rmse"""

    def __init__(self):
        self.sum = 0.
        self.count = 0

    def __call__(self):
        """Return the rmse value"""
        return np.sqrt(self.sum / self.count)

    def update(self, source, target):
        """Compute the rmse of the phoneme durations"""
        source_durations = np.array([p.duration() for p in source.phonemes()])
        target_durations = np.array([p.duration() for p in target.phonemes()])
        source_mask, target_mask = self.mask(source.phonemes(),
                                             target.phonemes())

        # First and last are very often long silences with large error
        differences = source_durations[source_mask][1:-1] - \
                      target_durations[target_mask][1:-1]

        self.sum += (differences ** 2).sum()
        self.count += differences.size

    @staticmethod
    def mask(source, target):
        """Retrieve the mask over values to use for evaluation"""
        source_mask = np.full(len(source), True)
        target_mask = np.full(len(target), True)

        source_idx, target_idx = 0, 0
        while source_idx < len(source) or target_idx < len(target):

            # Handle only one alignment ending with silence
            if target_idx >= len(target):
                source_mask[source_idx] = False
                source_idx += 1
                continue
            if source_idx >= len(source):
                target_mask[target_idx] = False
                target_idx += 1
                continue

            # Phonemes match
            if str(source[source_idx]) == str(target[target_idx]):
                source_idx += 1
                target_idx += 1
                continue

            # Phonemes don't match
            if str(source[source_idx]) == pypar.SILENCE:
                source_mask[source_idx] = False
                source_idx += 1
            elif str(target[target_idx]) == pypar.SILENCE:
                target_mask[target_idx] = False
                target_idx += 1
            else:
                raise ValueError('Phonemes don\'t match!')

        return source_mask, target_mask
