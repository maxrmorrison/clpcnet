import math

import numpy as np
import torch
import torchcrepe
import tqdm

import clpcnet


###############################################################################
# Pitch evaluation
###############################################################################


def from_files(source_pitch_files,
               target_pitch_files,
               source_periodicity_files,
               target_periodicity_files):
    """Evaluate pitch rmse in voiced regions and f1 of voiced/unvoiced"""
    metrics = PitchMetrics()

    # Voiced/unvoiced thresholding fn
    threshold = torchcrepe.threshold.Hysteresis()

    # Evaluate each pair of files
    iterator = zip(source_pitch_files,
                   target_pitch_files,
                   source_periodicity_files,
                   target_periodicity_files)
    iterator = tqdm.tqdm(iterator, desc='Evaluating pitch', dynamic_ncols=True)
    for source_pitch_file, \
        target_pitch_file, \
        source_periodicity_file, \
        target_periodicity_file in iterator:

        # Load files
        source_pitch = np.load(source_pitch_file)
        target_pitch = np.load(target_pitch_file)
        source_periodicity = np.load(source_periodicity_file)
        target_periodicity = np.load(target_periodicity_file)

        # Convert to torch
        source_pitch = torch.tensor(source_pitch)[None]
        target_pitch = torch.tensor(target_pitch)[None]
        source_periodicity = torch.tensor(source_periodicity)[None]
        target_periodicity = torch.tensor(target_periodicity)[None]

        # Threshold
        source = threshold(source_pitch, source_periodicity)
        target = threshold(target_pitch, target_periodicity)

        # Bound pitch
        source = torch.clamp(source, clpcnet.FMIN, clpcnet.FMAX)
        target = torch.clamp(target, clpcnet.FMIN, clpcnet.FMAX)

        # Compute metrics
        metrics.update(source, target)

    # Compute aggregate metrics over files
    return metrics()


class PitchMetrics:
    """Batch update pitch metrics"""

    gpe_20_threshold = 20. / 1200.
    gpe_50_threshold = 50. / 1200.

    def __init__(self, gpe_threshold=50):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.sum = 0.
        self.gpe_20_count = 0
        self.gpe_50_count = 0
        self.count = 0
        self.differences = []

    def __call__(self):
        """Compute the aggregate rmse, precision, recall, and f1"""
        precision = \
            self.true_positives / (self.true_positives + self.false_positives)
        recall = \
            self.true_positives / (self.true_positives + self.false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        rmse = 1200 * math.sqrt(self.sum / self.count)
        gpe_20 = self.gpe_20_count / self.count
        gpe_50 = self.gpe_50_count / self.count
        differences = 1200 * torch.cat(self.differences)
        return rmse, precision, recall, f1, gpe_20, gpe_50, differences

    def update(self, source, target):
        """Update the rmse, precision, recall, and f1"""
        source_voiced = ~torch.isnan(source)
        target_voiced = ~torch.isnan(target)
        overlap = source_voiced & target_voiced
        differences = torch.log2(source[overlap]) - torch.log2(target[overlap])
        self.true_positives += overlap.sum().item()
        self.false_positives += (~source_voiced & target_voiced).sum().item()
        self.false_negatives += (source_voiced & ~target_voiced).sum().item()
        self.sum += (differences ** 2).sum().item()
        self.gpe_20_count += (differences.abs() > self.gpe_20_threshold).sum().item()
        self.gpe_50_count += (differences.abs() > self.gpe_50_threshold).sum().item()
        self.count += source.numel()
        self.differences.append(differences)
