import abc
import itertools
import json
from pathlib import Path

import clpcnet


###############################################################################
# Functional interface - file access
###############################################################################


def files(name, directory, partition=None):
    """Get audio filenames"""
    return resolve(name).files(directory, partition)


def partition_file(name):
    """Get name of partition file"""
    return resolve(name).partition_file()


def partitions(name):
    """Get split of stems into partitions"""
    return resolve(name).partitions()


def stems(name, partition=None):
    """Get stems"""
    return resolve(name).stems(partition)


###############################################################################
# Functional interface - conversions
###############################################################################


def file_to_stem(name, file):
    """Convert file to stem"""
    return resolve(name).file_to_stem(file)


def stem_to_file(name, directory, stem):
    """Convert stem to file"""
    return resolve(name).stem_to_file(directory, stem)


###############################################################################
# Base dataset
###############################################################################


class Dataset(abc.ABC):

    ###########################################################################
    # File access
    ###########################################################################

    @classmethod
    def files(cls, directory, partition=None):
        """Get filenames"""
        # Get stems
        stems = cls.stems(partition)

        # Convert to files
        return [cls.stem_to_file(directory, stem) for stem in stems]

    @classmethod
    def partition_file(cls):
        """Get name of partition file"""
        return clpcnet.ASSETS_DIR / 'partition' / f'{cls.name}.json'

    @classmethod
    def partitions(cls):
        """Get split of stems into partitions"""
        with open(cls.partition_file()) as file:
            return json.load(file)

    @classmethod
    def stems(cls, partition=None):
        """Get stems"""
        # Get partitions
        partitions = cls.partitions()

        # Return all stems
        if partition is None:
            return itertools.chain(*partitions.values())

        # Return stems of a given partition
        return partitions[partition]

    ###########################################################################
    # Conversions
    ###########################################################################

    @staticmethod
    @abc.abstractmethod
    def file_to_stem(file):
        """Convert file to stem"""
        pass

    @staticmethod
    @abc.abstractmethod
    def stem_to_file(directory, stem):
        """Convert stem to file"""
        pass


###############################################################################
# Datasets
###############################################################################


class Daps(Dataset):

    name = 'daps'

    @staticmethod
    def file_to_stem(file):
        """Convert daps filename to stem"""
        return file.stem[:-4]

    @staticmethod
    def stem_to_file(directory, stem):
        """Convert daps stem to file"""
        return Path(directory, 'clean', f'{stem}.wav')


class DapsSegmented(Dataset):

    name = 'daps-segmented'

    @staticmethod
    def file_to_stem(file):
        """Convert daps-segmented filename to stem"""
        return file.stem

    @staticmethod
    def stem_to_file(directory, stem):
        """Convert daps-segmented stem to filen"""
        return Path(directory, f'{stem[:-6]}-sentences', f'{stem}.wav')


class RavdessHifi(Dataset):

    name = 'ravdess-hifi'

    @staticmethod
    def file_to_stem(file):
        """Convert ravdess filename to stem"""
        return file.stem

    @staticmethod
    def stem_to_file(directory, stem):
        """Convert ravdess stem to filename"""
        return directory / f'Actor_{stem[-2:]}' / f'{stem}.wav'


class RavdessVariable(RavdessHifi):

    name = 'ravdess-variable'


class Vctk(Dataset):

    name = 'vctk'

    @staticmethod
    def file_to_stem(file):
        """Convert vctk filename to stem"""
        return file.stem[:-5]

    @staticmethod
    def stem_to_file(directory, stem):
        """Convert vctk stem to file"""
        return Path(directory,
                    'wav48_silence_trimmed',
                    stem.split('_')[0],
                    f'{stem}_mic2.flac')


class VctkProMo(Dataset):

    name = 'vctk-promo'

    @staticmethod
    def file_to_stem(file):
        """Convert vctk file to stem"""
        return f'{file.parent.name}/{file.stem}'

    @staticmethod
    def stem_to_file(directory, stem):
        return directory / f'{stem}.wav'


###############################################################################
# Utilities
###############################################################################


def resolve(name):
    """Resolve name of dataset to static template"""
    if name == 'daps':
        return Daps
    elif name == 'daps-segmented':
        return DapsSegmented
    elif name == 'ravdess-hifi':
        return RavdessHifi
    elif name == 'ravdess-variable':
        return RavdessVariable
    elif name == 'vctk':
        return Vctk
    elif name == 'vctk-promo':
        return VctkProMo
    raise ValueError(f'Dataset {name} is not defined')
