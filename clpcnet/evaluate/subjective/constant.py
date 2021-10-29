import argparse
import multiprocessing as mp
from pathlib import Path

import soundfile

import clpcnet


###############################################################################
# Constants
###############################################################################


DURATION_RATIOS = [50, 71, 100, 141, 200]
PITCH_RATIOS = [67, 80, 100, 125, 150]


###############################################################################
# Subjective evaluation generation
###############################################################################


def generate(directory,
             run='clpcnet',
             checkpoint=clpcnet.DEFAULT_CHECKPOINT,
             gpu=None):
    """Preprare files for subjective evaluation on daps"""
    # Get daps files for evaluation
    files = clpcnet.data.files('daps-segmented', directory, 'test')

    # Setup output directory
    output_directory = clpcnet.EVAL_DIR / \
                       'subjective' / \
                       'constant' / \
                       'daps-segmented'
    output_directory.mkdir(exist_ok=True, parents=True)

    # Setup multiprocessing
    pool = mp.get_context('spawn').Pool()

    # Generate pitch-shifting examples
    generate_pitch(output_directory / 'constant-pitch',
                   files,
                   pool,
                   run,
                   checkpoint,
                   gpu)

    # Generate time-stretching examples
    generate_duration(output_directory / 'constant-duration',
                      files,
                      pool,
                      run,
                      checkpoint)

    # Close the pool and wait until processes finish
    pool.close()
    pool.join()

    # Convert to mp3
    wavfiles = list(output_directory.rglob('*.wav'))
    clpcnet.mp3.convert_files(wavfiles)

    # Remove wav files
    for file in wavfiles:
        file.unlink()


###############################################################################
# Constant-rate duration generation
###############################################################################


def generate_duration(output_directory,
                      files,
                      pool,
                      run='clpcnet',
                      checkpoint=clpcnet.DEFAULT_CHECKPOINT):
    """Prepare constant-rate time-stretching files for subjective evaluation"""
    # Setup output directory
    original_directory = output_directory / 'original'
    original_directory.mkdir(exist_ok=True, parents=True)

    # Iterate over utterances
    for file in files:

        # Write original audio
        original_file = \
            original_directory / \
            f'constant-duration_original_{file.stem.replace("_", "-")}.wav'
        soundfile.write(original_file,
                        clpcnet.load.audio(file),
                        clpcnet.SAMPLE_RATE)

        # Constant shifting with lpcnet
        pool.apply_async(generate_duration_lpcnet,
                         (file, output_directory, run, checkpoint))
        # generate_duration_lpcnet(file, output_directory, run, checkpoint)


def generate_duration_lpcnet(file,
                             output_directory,
                             run='clpcnet',
                             checkpoint=clpcnet.DEFAULT_CHECKPOINT):
    """Generate examples using lpcnet"""
    for ratio in DURATION_RATIOS:

        # Get run name
        name = f'{run}-{ratio:03d}'

        # Make output directory
        directory = output_directory / name
        directory.mkdir(exist_ok=True, parents=True)

        # Generate
        output_file = directory / \
            f'constant-duration_{name}_{file.stem.replace("_", "-")}.wav'
        clpcnet.from_file_to_file(file,
                                  output_file,
                                  constant_stretch=ratio / 100.,
                                  checkpoint_file=checkpoint,
                                  verbose=False)


###############################################################################
# Constant-rate pitch generation
###############################################################################


def generate_pitch(output_directory,
                   files,
                   pool,
                   run='clplcnet',
                   checkpoint=clpcnet.DEFAULT_CHECKPOINT,
                   gpu=None):
    """Prepare constant-rate pitch-shifting files for subjective evaluation"""
    # Setup output directory
    original_directory = output_directory / 'original'
    original_directory.mkdir(exist_ok=True, parents=True)

    # Iterate over utterances
    for file in files:

        # Write original audio
        original_file = \
            original_directory / \
            f'constant-pitch_original_{file.stem.replace("_", "-")}.wav'
        soundfile.write(original_file,
                        clpcnet.load.audio(file),
                        clpcnet.SAMPLE_RATE)

        # Constant shifting with lpcnet
        pool.apply_async(generate_pitch_lpcnet,
                         (file, output_directory, run, checkpoint))
        # generate_pitch_lpcnet(file, output_directory, run, checkpoint)


def generate_pitch_lpcnet(file,
                          output_directory,
                          run='clpcnet',
                          checkpoint=clpcnet.DEFAULT_CHECKPOINT):
    """Generate examples using lpcnet"""
    for ratio in PITCH_RATIOS:

        # Get run name
        name = f'{run}-{ratio:03d}'

        # Make output directory
        directory = output_directory / name
        directory.mkdir(exist_ok=True, parents=True)

        # Generate
        output_file = \
            directory / \
            f'constant-pitch_{name}_{file.stem.replace("_", "-")}.wav'
        clpcnet.from_file_to_file(file,
                                  output_file,
                                  constant_shift=ratio / 100.,
                                  checkpoint_file=checkpoint,
                                  verbose=False)


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=Path,
        default=clpcnet.DATA_DIR / 'daps-segmented',
        help='The root directory of the segmented daps dataset')
    parser.add_argument(
        '--run',
        default='clpcnet',
        help='The evaluation run')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=clpcnet.DEFAULT_CHECKPOINT,
        help='The checkpoint to use')
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='The gpu to use for pitch estimation')

    return parser.parse_args()


if __name__ == '__main__':
    generate(**vars(parse_args()))
