'''Copyright (c) 2018 Mozilla
   Modified by Max Morrison

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import argparse
import os
import random
import sys
from pathlib import Path

# Import keras without printing backend
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

import numpy as np
import tensorflow as tf
import tqdm

import clpcnet


###############################################################################
# Data loading utilities
###############################################################################


def load_features(directory, stems):
    """Load features from directory"""
    # Get feature filenames
    sample_files = [directory / f'{stem}-samples.u8' for stem in stems]
    frame_files = [directory / f'{stem}-frames.f32' for stem in stems]
    pitch_files = [directory / f'{stem}-pitch.npy' for stem in stems]
    periodicity_files = \
        [directory / f'{stem}-periodicity.npy' for stem in stems]

    # Iterate over files and load
    samples, frames, pitches, periodicities = [], [], [], []
    iterator = zip(sample_files, frame_files, pitch_files, periodicity_files)
    iterator = tqdm.tqdm(iterator, desc='loading files', dynamic_ncols=True)
    for sample_file, frame_file, pitch_file, periodicity_file in iterator:

        # Load sample rate features and target excitation
        samples.append(np.fromfile(sample_file, dtype=np.uint8))

        # Load frame rate features
        frames.append(np.fromfile(frame_file, dtype=np.float32))

        # Load pitch
        pitches.append(np.load(pitch_file))

        # Load periodicity
        periodicities.append(np.load(periodicity_file))

    # Concatenate
    samples = np.concatenate(samples)
    frames = np.concatenate(frames)
    pitches = np.concatenate(pitches)
    periodicities = np.concatenate(periodicities)

    # Truncate samples and extract targets
    samples, targets = prepare_sample_features(samples)

    # Truncate frames and extract pitch
    frames, pitches = prepare_frame_features(frames,
                                             len(samples),
                                             pitches,
                                             periodicities)

    return [samples, frames, pitches], targets


def load_fit_loop(directory, model, epoch=0, callbacks=None):
    """Loading all data at once takes up a lot of RAM. Loading only one batch
       at a time is too slow for clpcnet. We compromise and load a different
       large chunk of training features per epoch.
    """
    # Get training data stems
    files = sorted(directory.glob('*-samples.u8'))
    random.seed(0)
    random.shuffle(files)

    # Make a small validation set
    train_files, valid_files = files[1750:], files[:1750]

    # Load validation set into memory
    valid_stems = [file.stem[:-8] for file in valid_files]
    valid_inputs, valid_outputs = load_features(directory, valid_stems)

    index = 0
    steps = epoch * clpcnet.AVERAGE_STEPS_PER_EPOCH
    while steps < clpcnet.STEPS:

        # Select files for batch
        mem_mb = 0.
        batch = []
        # NOTE - The 4000 refers to 4 gb of just sample-rate features. The
        #        actual memory consumption is significantly (~4x) higher.
        while mem_mb < 4000.:

            # Finished dataset epoch
            if index == len(train_files):
                random.shuffle(train_files)
                index = 0

            # Add a file to next batch
            batch.append(train_files[index])
            mem_mb += train_files[index].stat().st_size / 2. ** 20
            index += 1

        # Load
        stems = [file.stem[:-8] for file in batch]
        inputs, outputs = load_features(directory, stems)

        # Fit
        model.fit(inputs,
                  outputs,
                  initial_epoch=epoch,
                  epochs=epoch + 1,
                  batch_size=clpcnet.BATCH_SIZE,
                  validation_split=0.,
                  callbacks=callbacks)
        results = model.evaluate(valid_inputs,
                                 valid_outputs,
                                 batch_size=clpcnet.BATCH_SIZE)
        print(f'Loss: {results[0]}, Accuracy: {results[1]}')
        epoch += 1
        steps += len(outputs)

        # Free memory
        del inputs
        del outputs


def pad(item, size=2):
    """Pad features via repetition for same-shape convolution"""
    left = np.concatenate([item[0:1, 0:size], item[:-1, -size:]])
    right = np.concatenate([item[1:, :size], item[0:1, -size:]])
    return np.concatenate([left, item, right], axis=1)


def prepare_frame_features(features, frames, pitch=None, periodicity=None):
    """Truncate frame and extract targets"""
    # Truncate features at a discrete number of blocks
    features = features[
        :frames * clpcnet.FEATURE_CHUNK_SIZE * clpcnet.TOTAL_FEATURE_SIZE]
    features = np.reshape(
        features,
        (frames, clpcnet.FEATURE_CHUNK_SIZE, clpcnet.TOTAL_FEATURE_SIZE))

    features = features[:, :, :clpcnet.SPECTRAL_FEATURE_SIZE]
    features[:, :, 18:36] = 0.

    # Optionally replace pitch
    if pitch is not None and not clpcnet.ABLATE_CREPE:

        # Truncate to a discrete number of blocks
        pitch = pitch[:frames * clpcnet.FEATURE_CHUNK_SIZE]

        # Convert to epochs
        epochs = clpcnet.convert.hz_to_epochs(pitch)

        # Replace Yin pitch
        features[:, :, clpcnet.PITCH_IDX] = \
            epochs.reshape(frames, clpcnet.FEATURE_CHUNK_SIZE)

    if pitch is not None and not clpcnet.ABLATE_CREPE:
        if clpcnet.ABLATE_PITCH_REPR:
            # Non-uniform pitch resolution
            pitch_bins = clpcnet.convertn.hz_to_length(pitch).reshape(
                frames, clpcnet.FEATURE_CHUNK_SIZE)[:, :, None]
        else:
            # Uniform pitch resolution
            pitch_bins = clpcnet.convert.hz_to_bins(pitch).reshape(
                frames, clpcnet.FEATURE_CHUNK_SIZE)[:, :, None]
    else:
        if clpcnet.ABLATE_PITCH_REPR:
            # Non-uniform pitch resolution
            pitch_bins = clpcnet.convert.epochs_to_length(
                features[:, :, clpcnet.PITCH_IDX:clpcnet.PITCH_IDX + 1])
        else:
            # Uniform pitch resolution
            pitch_bins = clpcnet.convert.epochs_to_bins(
                features[:, :, clpcnet.PITCH_IDX:clpcnet.PITCH_IDX + 1])

    # Optionally replace periodicity
    if periodicity is not None and not clpcnet.ABLATE_CREPE:

        # Scale periodicity
        periodicity = .8 * periodicity - .4

        # Truncate to a discrete number of blocks
        periodicity = periodicity[:frames * clpcnet.FEATURE_CHUNK_SIZE]

        # Replace Yin aperiodicity
        features[:, :, clpcnet.CORRELATION_IDX] = \
            periodicity.reshape(frames, clpcnet.FEATURE_CHUNK_SIZE)

    # Pad features via repetition for same-shape convolution
    return pad(features), pad(pitch_bins)


def prepare_sample_features(samples):
    """Truncate samples and extract targets"""
    # Truncate at discrete number of blocks
    frames = len(samples) // (4 * clpcnet.PCM_CHUNK_SIZE)
    samples = samples[:frames * 4 * clpcnet.PCM_CHUNK_SIZE]

    # Break data into signal, prediction, and excitations
    signal = np.reshape(samples[0::4], (frames, clpcnet.PCM_CHUNK_SIZE, 1))
    prediction = np.reshape(samples[1::4], (frames, clpcnet.PCM_CHUNK_SIZE, 1))
    input_excitation = np.reshape(samples[2::4],
                                  (frames, clpcnet.PCM_CHUNK_SIZE, 1))
    output_excitation = np.reshape(samples[3::4],
                                   (frames, clpcnet.PCM_CHUNK_SIZE, 1))

    # Concatenate input features
    features = np.concatenate([signal, prediction, input_excitation], axis=-1)

    return features, output_excitation


###############################################################################
# Model loading utilities
###############################################################################


def load_model(use_gpu, resume_from=None):
    """Setup the LPCNet model for training"""
    # Get the lpcnet model
    model = clpcnet.model(training=True, use_gpu=use_gpu)[0]

    # Compile model
    optimizer = keras.optimizers.Adam(clpcnet.LEARNING_RATE,
                                      amsgrad=True,
                                      decay=clpcnet.WEIGHT_DECAY)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    # Resume training from checkpoint
    if resume_from is not None:
        model.load_weights(resume_from)

    return model


###############################################################################
# Entry point
###############################################################################


def main():
    """Train a model"""
    # Parse command-line arguments
    args = parse_args()

    # Device management
    device = 'CPU' if args.gpu is None else 'GPU'
    number = '0' if args.gpu is None else str(args.gpu)
    with tf.device(f'/{device}:{number}'):

        # Tensorflow graph
        with tf.compat.v1.get_default_graph().as_default():

            # Tensorflow session
            gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
            config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
            session = tf.compat.v1.Session(config=config)
            keras.backend.set_session(session)

            # Setup model
            model = load_model(args.gpu is not None, args.resume_from)

            # Setup checkpointing
            args.checkpoint_directory.mkdir(parents=True, exist_ok=True)
            checkpoint_file = args.checkpoint_directory / \
                              (args.name + '-{epoch:03d}.h5')
            callbacks = [keras.callbacks.ModelCheckpoint(str(checkpoint_file))]

            # (Optional callback) - logging
            if args.log is not None:
                args.log.parent.mkdir(parents=True, exist_ok=True)
                callbacks.append(keras.callbacks.CSVLogger(str(args.log)))

            # Train
            initial_epoch = 0 if args.resume_from is None \
                else int(args.resume_from.stem[-3:])
            load_fit_loop(args.cache,
                          model,
                          initial_epoch,
                          callbacks=callbacks)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        '--name',
        default='clpcnet',
        help='The name for this experiment')
    parser.add_argument(
        '--dataset',
        default='vctk',
        help='The name of the dataset to train on')
    parser.add_argument(
        '--cache',
        type=Path,
        default=clpcnet.CACHE_DIR,
        help='The directory containing features and targets for training')

    # Checkpointing
    parser.add_argument(
        '--checkpoint_directory',
        type=Path,
        default=clpcnet.CHECKPOINT_DIR,
        help='The location on disk to save checkpoints')
    parser.add_argument(
        '--resume_from',
        type=Path,
        help='Checkpoint file to resume training from')

    # Logging
    parser.add_argument(
        '--log_directory',
        type=Path,
        default=clpcnet.LOG_DIR,
        help='The log directory')

    # Device placement
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='The gpu to use')

    # Extend directories with experiment name
    args = parser.parse_args()
    if args.cache:
        args.cache = args.cache / args.dataset
    if args.log_directory:
        args.log = args.log_directory / args.name
    if args.checkpoint_directory:
        args.checkpoint_directory = args.checkpoint_directory / args.name

    return args


if __name__ == '__main__':
    main()
