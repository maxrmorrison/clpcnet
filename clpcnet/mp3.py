import argparse
import glob
import multiprocessing as mp
import os
import shutil
import subprocess
from pathlib import Path


###############################################################################
# Convert audio to mp3
###############################################################################


def convert_file(input_file, output_file=None, verbose=False):
    """Convert audio file to mp3"""
    # Handle input files starting with hyphen
    clean_input = False
    if input_file.stem.startswith('-'):
        dummy_file = input_file.parent / input_file.name[1:]
        shutil.copyfile(input_file, dummy_file)
        input_file = dummy_file
        clean_input = True

    # Handle output files starting with hyphen
    clean_output = False
    if output_file.stem.startswith('-'):
        output_file = output_file.parent / output_file.name[1:]
        clean_output = True

    # Default output filename is same as input but with MP3 extension
    if output_file is None:
        output_file = input_file.with_suffix('.mp3')

    # Convert
    args = [
        'ffmpeg',
        '-y',
        '-i',
        str(input_file),
        '-b:a',
        '320k',
        str(output_file)]
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)
    stdout, stderr = process.communicate()

    # Maybe print
    if verbose or process.returncode != 0:
        print(stdout)
        print(stderr)

    # Clean-up input files starting with hyphen
    if clean_input:
        os.remove(input_file)

    # Clean-up output files starting with hyphen
    if clean_output:
        os.replace(output_file, output_file.parent / ('-' + output_file.name))


def convert_files(input_files, output_files=None):
    """Convert audio files to mp3"""
    # Convert to paths
    input_files = [Path(file) for file in input_files]

    # Default output filename is same as input but with MP3 extension
    if output_files is None:
        output_files = [file.with_suffix('.mp3') for file in input_files]

    # Multiprocess conversion
    with mp.Pool() as pool:
        pool.starmap(convert_file, zip(input_files, output_files))

    # for input_file, output_file in zip(input_files, output_files):
    #     convert_file(input_file, output_file)


###############################################################################
# Entry point
###############################################################################


def expand_files(files):
    """Expands a wildcard to a list of paths for Windows compatibility"""
    # Split at whitespace
    files = files.split()

    # Handle wildcard expansion
    if len(files) == 1 and '*' in files[0]:
        files = glob.glob(files[0])

    # Convert to Path objects
    return files


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()

    # Handle wildcards across platforms
    if os.name == 'nt':
        parser.add_argument(
            '--input_files',
            type=expand_files,
            help='The audio files to convert to mp3')
    else:
        parser.add_argument(
            '--input_files',
            nargs='+',
            help='The audio files to convert to mp3')

    parser.add_argument(
        '--output_files',
        type=Path,
        nargs='+',
        help='The corresponding output files. ' +
             'Uses same filename with mp3 extension by default')
    return parser.parse_args()


if __name__ == '__main__':
    convert_files(**vars(parse_args()))
