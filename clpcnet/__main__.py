import argparse
from pathlib import Path

import clpcnet


###############################################################################
# Entry point
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()

    # Audio I/O
    parser.add_argument(
        '--audio_files',
        type=Path,
        nargs='+',
        help='The audio files to process')
    parser.add_argument(
        '--output_files',
        type=Path,
        nargs='+',
        required=True,
        help='The files to write the output audio')

    # Time-stretching
    parser.add_argument(
        '--source_alignment_files',
        type=Path,
        nargs='+',
        help='The original alignments on disk. Used only for time-stretching.')
    parser.add_argument(
        '--target_alignment_files',
        type=Path,
        nargs='+',
        help='The target alignments on disk. Used only for time-stretching.')
    parser.add_argument(
        '--constant_stretch',
        type=float,
        help='A constant value for time-stretching')

    # Pitch shifting
    parser.add_argument(
        '--source_pitch_files',
        type=Path,
        nargs='+',
        help='The file containing the original pitch contours')
    parser.add_argument(
        '--source_periodicity_files',
        type=Path,
        nargs='+',
        help='The file containing the original periodicities')
    parser.add_argument(
        '--target_pitch_files',
        type=Path,
        nargs='+',
        help='The files containing the desired pitch contours')
    parser.add_argument(
        '--constant_shift',
        type=float,
        help='A constant value for pitch-shifting')

    # Model checkpoint
    parser.add_argument(
        '--checkpoint_file',
        type=Path,
        default=clpcnet.DEFAULT_CHECKPOINT,
        help='The checkpoint file to load')

    return parser.parse_args()


if __name__ == '__main__':
    clpcnet.from_files_to_files(**vars(parse_args()))
