# Controllable LPCNet

Code corresponding to the paper "Neural Pitch-Shifting and Time-Stretching
with Controllable LPCNet" [1]. Audio examples can be found
[here](https://maxrmorrison.com/sites/controllable-lpcnet). The original
LPCNet [2] code can be found [here](https://github.com/mozilla/LPCNet).


## Table of contents

- [Installation](#installation)
  * [Without Docker](#without-docker)
  * [With Docker](#with-docker)
- [Inference](#inference)
  * [Library inference](#library-inference)
  * [Command-line inference](#command-line-inference)
- [Replicating results](#replicating-results)
  * [Partition the dataset](#partition-the-dataset)
  * [Preprocess the dataset](#preprocess-the-dataset)
  * [Train the model](#train-the-model)
  * [Evaluate the model](#evaluate-the-model)
- [API](#api)
  * [`clpcnet.from_audio`](#clpcnetfrom_audio)
  * [`clpcnet.from_features`](#clpcnetfrom_features)
  * [`clpcnet.from_file`](#clpcnetfrom_file)
  * [`clpcnet.from_file_to_file`](#clpcnetfrom_file_to_file)
  * [`clpcnet.from_files_to_files`](#clpcnetfrom_files_to_files)
  * [`clpcnet.to_file`](#clpcnetto_file)
- [CLI](#cli)
  * [`clpcnet`](#clpcnet)
  * [`clpcnet.evaluate.gather`](#clpcnetevaluategather)
  * [`clpcnet.evaluate.objective.constant`](#clpcnetevaluateobjectiveconstant)
  * [`clpcnet.evaluate.objective.variable`](#clpcnetevaluateobjectivevariable)
  * [`clpcnet.evaluate.subjective.constant`](#clpcnetevaluatesubjectiveconstant)
  * [`clpcnet.evaluate.subjective.variable`](#clpcnetevaluatesubjectivevariable)
  * [`clpcnet.partition`](#clpcnetpartition)
  * [`clpcnet.pitch`](#clpcnetpitch)
  * [`clpcnet.preprocess`](#clpcnetpreprocess)
  * [`clpcnet.preprocess.augment`](#clpcnetpreprocessaugment)
  * [`clpcnet.train`](#clpcnet.train)
- [Citation](#citation)
- [References](#references)


## Installation

### With Docker

Docker installation assumes recent versions of Docker and NVidia Docker are
installed.

In order to perform variable-ratio time-stretching, you must first
download HTK 3.4.0 (see
[download instructions](https://github.com/maxrmorrison/pyfoal#hidden-markov-model-toolkit-htk)),
which is used for forced phoneme alignment.
Note that you only have to download HTK. You do not have to install it locally.
HTK must be downloaded to within this directory in order to be considered part of
the build context.

Next, build the image.

```bash
docker build --tag clpcnet --build-arg HTK=<path_to_htk> .
```

Now we can run a command within the Docker image.

```bash
docker run -itd --rm --name "clpcnet" --shm-size 32g --gpus all \
  -v <absolute_path_of_runs_directory>:/clpcnet/runs \
  -v <absolute_path_of_data_directory>:/clpcnet/data \
  clpcnet:latest \
  <command>
```

Where `<command>` is the command you would like to execute within the
container, prefaced with the correct Python path within the Docker
image (e.g., `/opt/conda/envs/clpcnet/bin/python -m clpcnet.train --gpu 0`).


### Without Docker

Installation assumes we start from a clean install of Ubuntu 18 or 20 with a
recent CUDA driver and `conda`.

Install the apt-get dependencies.
```bash
sudo apt-get update && \
sudo apt-get install -y \
    ffmpeg \
    gcc-multilib \
    libsndfile1 \
    sox
```

Build the C preprocessing code
```bash
make
```

Create a new conda environment and install conda dependencies
```bash
conda create -n clpcnet python=3.7 cudatoolkit=10.0 cudnn=7.6 -y
conda activate clpcnet
```

Finally, install the Python dependencies
```bash
pip install -e .
```

If you would like to perform variable-ratio time-stretching, you must also
download and install HTK 3.4.0 (see
[download instructions](https://github.com/maxrmorrison/pyfoal#hidden-markov-model-toolkit-htk)),
which is used for forced phoneme alignment.


## Inference

`clpcnet` can be used as a library (via `import clpcnet`) or
as an application accessed via the command-line.


### Library inference

To perform pitch-shifting or time-stretching on audio already loaded into
memory, use `clpcnet.from_audio`. To do this with audio saved in a file, use
`clpcnet.from_file`. You can use `clpcnet.to_file` or
`clpcnet.from_file_to_file` to save the results to a file. To process many
files at once with multiprocessing, use `clpcnet.from_files_to_files`. To
perform vocoding from acoustic features, use `clpcnet.from_features`.
See [the `clpcnet` API](#api) for full argument lists. Below is an example of
performing constant-ratio pitch-shifting with a ratio of 0.8 and
constant-ratio time-stretching with a ratio of 1.2.

```python
import clpcnet

# Load audio from disk and resample to 16 kHz
audio_file = 'audio.wav'
audio = clpcnet.load.audio(audio_file)

# Perform constant-ratio pitch-shifting and time-stretching
generated = clpcnet.from_audio(audio, constant_stretch=1.2, constant_shift=0.8)
```


### Command-line inference

The command-line interface for inference wraps the arguments of
[`clpcnet.from_files_to_files`](#clpcnet.from_files_to_files).
To run inference using a pretrained model, use the module entry point. For
example, to resynthesize speech without modification, use the following.

```bash
python -m clpcnet --audio_files audio.wav --output_files output.wav
```

See [the command-line interface documentation](#clpcnet) for full list and
description of arguments.


## Replicating results

Here we demonstrate how to replicate the results of the paper "Neural
Pitch-Shifting and Time-Stretching with Controllable LPCNet" on the VCTK
dataset [3]. Time estimates are given using a 12-core 12-thread CPU with a 2.1 GHz
clock and a NVidia V100 GPU. First,
[download VCTK](https://datashare.ed.ac.uk/handle/10283/3443) so that the root
directory of VCTK is `./data/vctk` (Time estimate: ~10 hours).


### Partition the dataset

Partitions for each dataset are saved in
`./clpcnet/assets/partition/`. The VCTK partition file can be explicitly
recomputed as follows.

```bash
# Time estimate: < 10 seconds
python -m clpcnet.partition
```


### Preprocess the dataset

```bash
# Compute YIN pitch and periodicity, BFCCs, and LPC coefficients.
# Time estimate: ~15 minutes
python -m clpcnet.preprocess

# Compute CREPE pitch and periodicity (Section 3.1).
# Time estimate: ~2.5 hours
python -m clpcnet.pitch --gpu 0

# Perform data augmentation (Section 3.2).
# Time estimate: ~17 hours
python -m clpcnet.preprocess.augment --gpu 0
```

All files are saved to `./runs/cache/vctk` by default.


### Train the model

```bash
# Time estimate: ~75 hours
python -m clpcnet.train --gpu 0
```

Checkpoints are saved to `./runs/checkpoints/clpcnet/`. Log files are saved to `./runs/logs/clpcnet`.


### Evaluate the model

We perform evaluation on VCTK [3], as well as modified versions of DAPS [4] and
RAVDESS [5]. We modify DAPS by segmenting the dataset into sentences, and
providing transcripts for each sentence. To reproduce results on DAPS,
download [our modified DAPS dataset on Zenodo](https://zenodo.org/record/4783456#.YVjN07hKguU)
and decompress the tarball within `data/`. We modify RAVDESS by performing speech enhancement with
HiFi-GAN [6]. To reproduce results on RAVDESS, download
[our modified RAVDESS dataset on Zenodo](https://zenodo.org/record/4783521#.YVjN7rhKguU)
and decompress the tarball within `data/`.

To create the DAPS partition file for evaluation, run the following.

```bash
# Partition the modified DAPS dataset
# Time estimate: < 10 seconds
python -m clpcnet.partition --dataset daps-segmented
```

We create two partition files for RAVDESS. The first is used for variable-ratio
pitch-shifting and time-stretching, and creates pairs of audio files. The
second samples files from those pairs for constant-ratio evaluation.

```bash
# Create pairs for variable-ratio evaluation from modified RAVDESS dataset
# Time estimate: ~1 hour
python -m clpcnet.partition --dataset ravdess-variable --gpu 0

# Sample files for constant-ratio evaluation
# Time estiamte: ~3 seconds
python -m clpcnet.partition --dataset ravdess-hifi
```


#### Constant-ratio objective evaluation

We perform constant-ratio objective evaluation on VCTK [3], as well as our
modified DAPS [4] and RAVDESS [5] datasets.

```bash
# Prepare files for constant-ratio objective evaluation
# Files are saved to ./runs/eval/objective/constant/vctk/data/
# Time estimate:
#  - vctk: ~2 minutes
#  - daps-segmented: ~3 minutes
#  - ravdess-hifi: ~3 minutes
python -m clpcnet.evaluate.gather --dataset <dataset> --gpu 0

# Evaluate
# Results are written to ./runs/eval/objective/constant/vctk/results.json
# Time estimate:
#  - vctk: ~2 hours
#  - daps-segmented: ~4.5 hours
#  - ravdess-hifi: ~3.5 hours
python -m clpcnet.evaluate.objective.constant \
    --checkpoint ./runs/checkpoints/clpcnet/clpcnet-103.h5 \
    --dataset <dataset>
    --gpu 0
```

`<dataset>` can be one of `vctk` (default), `daps-segmented`, or
`ravdess-hifi`.


#### Variable-ratio objective evaluation

We perform variable-ratio objective evaluation on our modified RAVDESS [5]
dataset.

```bash
# Results are written to ./runs/eval/objective/variable/ravdess-hifi/results.json
# Time estimate: ~1.5 hours
python -m clpcnet.evaluate.objective.variable \
    --checkpoint ./runs/checkpoints/clpcnet/clpcnet-103.h5 \
    --gpu 0
```


#### Constant-ratio subjective evaluation

We perform constant-ratio subjective evaluation on our modified DAPS [4]
dataset.

```bash
# Files are written to ./runs/eval/subjective/constant/daps-segmented
# Time estimate: ~10 hours
python -m clpcnet.evaluate.subjective.constant \
    --checkpoint ./runs/checkpoints/clpcnet/clpcnet-103.h5 \
    --gpu 0
```


#### Variable-ratio subjective evaluation

We perform variable-ratio subjective evaluation on our modified RAVDESS [5]
dataset.

```bash
# Files are written to ./runs/eval/subjective/variable/ravdess-hifi
# Time estimate: ~1.5 hours
python -m clpcnet.evaluate.subjective.variable \
    --checkpoint ./runs/checkpoints/clpcnet/clpcnet-103.h5 \
    --gpu 0
```


#### Baselines

##### Original LPCNet

To perform inference using original LPCNet settings, set `ORIGINAL_LPCNET` to
`True` in `./clpcnet/config.py`.


##### TD-PSOLA

The implementation of TD-PSOLA [7] used as a baseline in our paper is released
under a GPL license and can be found
[here](https://github.com/maxrmorrison/psola).


##### WORLD

The implementation of WORLD [8] used as a baseline in our paper can be found
[here](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder). We provide
a wrapper for pitch-shifting and time-stretching in `clpcnet/world.py`.


## API

### `clpcnet.from_audio`

```python
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
```


### `clpcnet.from_features`

```python
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
```


### `clpcnet.from_file`

```python
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
```


### `clpcnet.from_file_to_file`

```python
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
```


### `clpcnet.from_files_to_files`

```python
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
```


### `clpcnet.to_file`

```python
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
```


## CLI

### `clpcnet`

Perform pitch-shifting and time-stretching with a pretrained model.

```
usage: python -m clpcnet [-h]
                         [--audio_files AUDIO_FILES [AUDIO_FILES ...]]
                         --output_files OUTPUT_FILES [OUTPUT_FILES ...]
                         [--source_alignment_files SOURCE_ALIGNMENT_FILES [SOURCE_ALIGNMENT_FILES ...]]
                         [--target_alignment_files TARGET_ALIGNMENT_FILES [TARGET_ALIGNMENT_FILES ...]]
                         [--constant_stretch CONSTANT_STRETCH]
                         [--source_pitch_files SOURCE_PITCH_FILES [SOURCE_PITCH_FILES ...]]
                         [--source_periodicity_files SOURCE_PERIODICITY_FILES [SOURCE_PERIODICITY_FILES ...]]
                         [--target_pitch_files TARGET_PITCH_FILES [TARGET_PITCH_FILES ...]]
                         [--constant_shift CONSTANT_SHIFT]
                         [--checkpoint_file CHECKPOINT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --audio_files AUDIO_FILES [AUDIO_FILES ...]
                        The audio files to process
  --output_files OUTPUT_FILES [OUTPUT_FILES ...]
                        The files to write the output audio
  --source_alignment_files SOURCE_ALIGNMENT_FILES [SOURCE_ALIGNMENT_FILES ...]
                        The original alignments on disk. Used only for time-
                        stretching.
  --target_alignment_files TARGET_ALIGNMENT_FILES [TARGET_ALIGNMENT_FILES ...]
                        The target alignments on disk. Used only for time-
                        stretching.
  --constant_stretch CONSTANT_STRETCH
                        A constant value for time-stretching
  --source_pitch_files SOURCE_PITCH_FILES [SOURCE_PITCH_FILES ...]
                        The file containing the original pitch contours
  --source_periodicity_files SOURCE_PERIODICITY_FILES [SOURCE_PERIODICITY_FILES ...]
                        The file containing the original periodicities
  --target_pitch_files TARGET_PITCH_FILES [TARGET_PITCH_FILES ...]
                        The files containing the desired pitch contours
  --constant_shift CONSTANT_SHIFT
                        A constant value for pitch-shifting
  --checkpoint_file CHECKPOINT_FILE
                        The checkpoint file to load
```


### `clpcnet.evaluate.gather`

Gather files for constant-ratio objective evaluation.

```
usage: python -m clpcnet.evaluate.gather [-h]
                                         [--dataset DATASET]
                                         [--directory DIRECTORY]
                                         [--output_directory OUTPUT_DIRECTORY]
                                         [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     The dataset to gather evaluation files from
  --directory DIRECTORY
                        The root directory of the dataset
  --output_directory OUTPUT_DIRECTORY
                        The output evaluation directory
  --gpu GPU             The gpu to use for pitch estimation
```


### `clpcnet.evaluate.objective.constant`

Perform constant-ratio objective evaluation on a trained model.

```
usage: python -m clpcnet.evaluate.objective.constant [-h]
                                                     [--checkpoint CHECKPOINT]
                                                     [--run RUN]
                                                     [--directory DIRECTORY]
                                                     [--dataset DATASET]
                                                     [--skip_generation]
                                                     [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        The lpcnet checkpoint file
  --run RUN             The name of the experiment
  --directory DIRECTORY
                        The evaluation directory
  --dataset DATASET     The dataset to evaluate
  --skip_generation     Whether to skip generation and eval a previously
                        generated run
  --gpu GPU             The gpu to run pitch tracking on
```


### `clpcnet.evaluate.objective.variable`

```
usage: python -m clpcnet.evaluate.objective.variable [-h]
                                                     [--directory DIRECTORY]
                                                     [--run RUN]
                                                     [--checkpoint CHECKPOINT]
                                                     [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        Root directory of the ravdess dataset
  --run RUN             The evaluation run
  --checkpoint CHECKPOINT
                        The model checkpoint
  --gpu GPU             The index of the gpu to use
```


### `clpcnet.evaluate.subjective.constant`

```
usage: python -m clpcnet.evaluate.subjective.constant [-h]
                                                      [--directory DIRECTORY]
                                                      [--run RUN]
                                                      [--checkpoint CHECKPOINT]
                                                      [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        The root directory of the segmented daps dataset
  --run RUN             The evaluation run
  --checkpoint CHECKPOINT
                        The checkpoint to use
  --gpu GPU             The gpu to use for pitch estimation
```


### `clpcnet.evaluate.subjective.variable`

```
usage: clpcnet.evaluate.subjective.variable [-h]
                                            [--directory DIRECTORY]
                                            [--output_directory OUTPUT_DIRECTORY]
                                            [--run RUN]
                                            [--checkpoint CHECKPOINT]
                                            [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        Root directory of the ravdess dataset
  --output_directory OUTPUT_DIRECTORY
                        The location to store files for subjective evaluation
  --run RUN             The evaluation run
  --checkpoint CHECKPOINT
                        The checkpoint to use
  --gpu GPU             The index of the gpu to use
```


### `clpcnet.partition`

Partition a dataset.

```
usage: python -m clpcnet.partition [-h]
                                   [--dataset DATASET]
                                   [--directory DIRECTORY]
                                   [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     The name of the dataset
  --directory DIRECTORY
                        The data directory
  --gpu GPU             The gpu to use
```


### `clpcnet.pitch`

Compute CREPE pitch and periodicity on a dataset (Section 3.1).

```
usage: python -m clpcnet.pitch [-h]
                               [--dataset DATASET]
                               [--directory DIRECTORY]
                               [--cache CACHE]
                               [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     The dataset to perform pitch tracking on
  --directory DIRECTORY
                        The data directory
  --cache CACHE         The cache directory
  --gpu GPU             The gpu to use for pitch tracking
```


### `clpcnet.preprocess`

Compute YIN pitch and periodicity, BFCCs, and LPC coefficients on a dataset.

```
usage: python -m clpcnet.preprocess [-h]
                                    [--dataset DATASET]
                                    [--directory DIRECTORY]
                                    [--cache CACHE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     The dataset to preprocess
  --directory DIRECTORY
                        The data directory
  --cache CACHE         The cache directory
```


### `clpcnet.preprocess.augment`

Perform resampling data augmentation (Section 3.2).

```
usage: python -m clpcnet.preprocess.augment [-h]
                                            [--dataset DATASET]
                                            [--directory DIRECTORY]
                                            [--cache CACHE]
                                            [--allowed_scales ALLOWED_SCALES [ALLOWED_SCALES ...]]
                                            [--passes PASSES]
                                            [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     The name of the dataset
  --directory DIRECTORY
                        The data directory
  --cache CACHE         The cache directory
  --allowed_scales ALLOWED_SCALES [ALLOWED_SCALES ...]
                        The allowable scale values for resampling
  --passes PASSES       The number of augmentation passes to make over the
                        dataset
  --gpu GPU             The index of the gpu to use
```


### `clpcnet.train`

Train a model.

```
usage: python -m clpcnet.train [-h]
                               [--name NAME]
                               [--dataset DATASET]
                               [--cache CACHE]
                               [--checkpoint_directory CHECKPOINT_DIRECTORY]
                               [--resume_from RESUME_FROM]
                               [--log_directory LOG_DIRECTORY]
                               [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           The name for this experiment
  --dataset DATASET     The name of the dataset to train on
  --cache CACHE         The directory containing features and targets for
                        training
  --checkpoint_directory CHECKPOINT_DIRECTORY
                        The location on disk to save checkpoints
  --resume_from RESUME_FROM
                        Checkpoint file to resume training from
  --log_directory LOG_DIRECTORY
                        The log directory
  --gpu GPU             The gpu to use
```


## Citation

### IEEE
M. Morrison, Z. Jin, N. J. Bryan, J. Caceres, and B. Pardo, "Neural Pitch-Shifting and Time-Stretching with Controllable LPCNet," Submitted to Interspeech 2021, August 2021.


### BibTex

```
@inproceedings{morrison2021neural,
    title={Neural Pitch-Shifting and Time-Stretching with Controllable LPCNet},
    author={Morrison, Max and Jin, Zeyu and Bryan, Nicholas J and Caceres, Juan-Pablo and Pardo, Bryan},
    booktitle={Submitted to Interspeech 2021},
    month={August},
    year={2021}
}
```


## References
- [1] M. Morrison, Z. Jin, N. J. Bryan, J. Caceres, and B. Pardo, "Neural Pitch-Shifting and Time-Stretching with Controllable LPCNet," Submitted to Interspeech 2021, August 2021.
- [2] J. M. Valin and J. Skoglund, "LPCNet: Improving neural speech
      synthesis through linear prediction," International
      Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019.
- [3] J. Yamagishi, C. Veaux, and K. MacDonald, "CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)," 2019.
- [4] G. J. Mysore, "Can we automatically transform speech recorded on common consumer devices in real-world environments into professional production quality speech?—a dataset, insights, and challenges." IEEE Signal Processing Letters 22.8. 2014.
- [5] S. R. Livingstone and F. A. Russo. "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English." PloS one 13.5. 2018.
- [6] J. Su, Z. Jin, and A. Finkelstein. "HiFi-GAN: High-fidelity denoising and dereverberation based on speech deep features in adversarial networks." Interspeech. October 2020.
- [7] E. Moulines and F. Charpentier. "Pitch-synchronous waveform processing techniques for text-to-speech synthesis using diphones." Speech communication 9.5-6. 1990.
- [8] M. Morise, F. Yokomori, and K. Ozawa, "WORLD: a vocoder-based high-quality speech synthesis system for real-time applications," IEICI Transactions on Information and Systems, 2016.
