import itertools
import random
import warnings

import numpy as np
import pyfoal
import pypar
import soundfile

import clpcnet


###############################################################################
# Prosody transfer representation
###############################################################################


class ProsodyTransfer:
    """Representation for a prosody transfer task"""

    def __init__(self, name, text, source_audio, target_audio, gpu=None):
        self.name = name
        self.text = text
        self.source_audio = source_audio
        self.target_audio = target_audio
        self.gpu = gpu

    def is_valid(self):
        """Check if phoneme alignments match"""
        # Get phoneme alignments
        source = self.source_alignment()
        target = self.target_alignment()

        # Get phonemes
        source_phonemes = source.phonemes()
        target_phonemes = target.phonemes()

        # Length and phoneme checks
        iterator = zip(source_phonemes, target_phonemes)
        if len(source_phonemes) != len(target_phonemes) or \
           any([str(s) != str(t) for s, t in iterator]):
           return False

        # Relative rate check
        rates = np.array(pypar.compare.per_phoneme_rate(source, target))
        if any(rates > 4.) or any(rates < .25):
            return False

        # Get pitch
        source_pitch, source_harm = self.source_pitch(return_periodicity=True)
        target_pitch = self.target_pitch()

        # Invert target
        aligned_pitch = clpcnet.pitch.align(source_pitch,
                                            target_pitch,
                                            source,
                                            target)

        # Threshold
        source_pitch = clpcnet.pitch.threshold(source_pitch, source_harm)

        # Extract voiced
        voiced = ~np.isnan(source_pitch)
        source = source_pitch[voiced]
        target = aligned_pitch[voiced]

        # Pitch range check
        if any(source > 400) or any(source < 65) or \
           any(target > 400) or any(target < 65):
            return False

        # Pitch shift range check
        rate = np.abs(target / source)
        return all(rate <= 2.5) and all(rate >= .4)


    @classmethod
    def from_file(cls, prefix, gpu=None):
        """Load from disk"""
        # Load text
        with open(prefix.with_suffix('.txt')) as file:
            text = file.read()

        # Load audio
        source_audio = clpcnet.load.audio(
            prefix.parent / (prefix.stem + '-source.wav'))
        target_audio = clpcnet.load.audio(
            prefix.parent / (prefix.stem + '-target.wav'))

        # Make transfer
        return cls(prefix.stem, text, source_audio, target_audio, gpu)

    def save(self, directory):
        """Save audio files to directory"""
        prefix = directory / f'{self.name}'

        # Save text
        with open(prefix.parent / (prefix.stem + '.txt'), 'w') as file:
            file.write(self.text)

        # Save audio
        soundfile.write(prefix.parent / (prefix.stem + '-source.wav'),
                        self.source_audio,
                        clpcnet.SAMPLE_RATE)
        soundfile.write(prefix.parent / (prefix.stem + '-target.wav'),
                        self.target_audio,
                        clpcnet.SAMPLE_RATE)

    def source_alignment(self):
        """Retrieve the source alignment"""
        if not hasattr(self, '_source_alignment'):
            self._source_alignment = pyfoal.align(self.text,
                                                  self.source_audio,
                                                  clpcnet.SAMPLE_RATE)
        return self._source_alignment

    def source_pitch(self, return_periodicity=False):
        """Retrieve the source pitch"""
        if not hasattr(self, '_source_pitch'):
            self._source_pitch = clpcnet.pitch.from_audio(self.source_audio,
                                                          self.gpu)
        return \
            self._source_pitch if return_periodicity else self._source_pitch[0]

    def target_alignment(self):
        """Retrieve the target alignment"""
        if not hasattr(self, '_target_alignment'):
            self._target_alignment = pyfoal.align(self.text,
                                                  self.target_audio,
                                                  clpcnet.SAMPLE_RATE)
        return self._target_alignment

    def target_pitch(self, return_periodicity=False):
        """Retrieve the target pitch"""
        if not hasattr(self, '_target_pitch'):
            self._target_pitch = clpcnet.pitch.from_audio(self.target_audio,
                                                          self.gpu)
        return \
            self._target_pitch if return_periodicity else self._target_pitch[0]


###############################################################################
# Dataset generators
###############################################################################


def ravdess_generator(directory, gpu=None):
    """Generator over examples in ravdess dataset"""
    # Get audio files
    files = sorted(directory.glob('Actor_*/*.wav'))

    # Get file metadata
    metadata = [RavdessFileMetadata(f) for f in files]

    # Filter out high intensity
    metadata = [m for m in metadata if m.intensity == 1]

    # Statement text
    text = {1: 'Kids are talking by the door',
            2: 'Dogs are sitting by the door'}

    # We make five matches per statement per speaker. There are 20 speakers
    # that satisfy this given our filtering, for a total of 100 matches.
    for speaker in range(1, 25):

        # Skip speakers that cannot produce 5 matches
        if speaker in [4, 9, 14, 20]:
            continue

        for statement in range(1, 3):

            # Get relevant files
            candidates = [
                m for m in metadata
                if m.actor == speaker and m.statement == statement]

            # Iterate over unique pairs in random order
            matches = 0
            iterator = list(itertools.combinations(candidates, 2))
            random.shuffle(iterator)
            for sample_a, sample_b in iterator:

                # Create match
                transfer = ProsodyTransfer(
                    f'{sample_a.file.stem}_{sample_b.file.stem}',
                    text[statement],
                    clpcnet.load.audio(sample_a.file),
                    clpcnet.load.audio(sample_b.file),
                    gpu=gpu)

                # Check if phoneme alignments match
                if transfer.is_valid():
                    yield transfer

                    # Check if we've made enough matches
                    matches += 1
                    if matches == 5:
                        break

            # Raise if we couldn't find enough matches
            if matches != 5:
                warnings.warn(f'Can only find {matches} of 5 matches')
                # raise ValueError(f'Can only find {matches} of 5 matches')


###############################################################################
# Utilities
###############################################################################


class RavdessFileMetadata:
    """Parses the filename into metadata"""

    def __init__(self, file):
        self.file = file

        entries = file.stem.split('-')
        self.modality = int(entries[0])
        self.channel = int(entries[1])
        self.emotion = int(entries[2])
        self.intensity = int(entries[3])
        self.statement = int(entries[4])
        self.repetition = int(entries[5])
        self.actor = int(entries[6])
