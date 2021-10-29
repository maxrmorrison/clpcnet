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
import functools
import math
import os
import sys

# Import keras without printing backend
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

import numpy as np
from keras import backend as K
from keras.layers import Concatenate, Input, Reshape

import clpcnet


###############################################################################
# LPCNet model construction
###############################################################################


def model(training=False, use_gpu=True):
    """Build the LPCNet model"""

    ###########################################################################
    # Inputs
    ###########################################################################

    # Signal, prediction, and excitation inputs
    sample_rate_feats = Input(shape=(None, 3))

    # Bark-scale coefficients and pitch correlation
    spectral_feats = Input(shape=(None, clpcnet.SPECTRAL_FEATURE_SIZE))

    # Pitch period
    pitch = Input(shape=(None, 1))

    ###########################################################################
    # Create graph
    ###########################################################################

    # Build and link frame-rate network
    frame_rate_feats = frame_rate_network(spectral_feats, pitch, training)

    # Build and add sample-rate network
    probabilities, decoder_model = sample_rate_network(
        frame_rate_feats, sample_rate_feats, use_gpu)

    # Build lpcnet model
    model = keras.models.Model([sample_rate_feats, spectral_feats, pitch],
                               probabilities)

    # Build encoder model
    encoder_model = encoder(spectral_feats, pitch, frame_rate_feats)

    return model, encoder_model, decoder_model


###############################################################################
# Model components
###############################################################################


def decoder(sample_rate_feats,
            sample_rate_embedding,
            gru_a,
            gru_b,
            dual_dense):
    """Build the LPCNet decoder"""

    ###########################################################################
    # Inputs
    ###########################################################################

    # Frame-rate features upsampled to the sampling rate
    upsampled = Input(shape=(None, 128))

    # GRU A initial state
    gru_a_init = Input(shape=(clpcnet.GRU_A_SIZE,))

    # GRU B initial state
    gru_b_init = Input(shape=(clpcnet.GRU_B_SIZE,))

    ###########################################################################
    # Link
    ###########################################################################

    # Concatenate sample-rate and upsampled frame-rate features
    all_sample_rate_feats = Concatenate()([sample_rate_embedding, upsampled])

    # Add sample-rate gru A to graph
    activation, gru_a_state = gru_a(all_sample_rate_feats,
                                    initial_state=gru_a_init)

    # Residual connection between upsampled features and rnn output
    # Note: this is NOT in the original LPCNet paper, but is in the code
    activation = Concatenate()([activation, upsampled])

    # Add sample-rate gru B to graph
    activation, gru_b_state = gru_b(activation,
                                    initial_state=gru_b_init)

    # Add dual fully-connected layer to graph
    probabilities = dual_dense(activation)

    # Specify model start and end points
    inputs = [sample_rate_feats, upsampled, gru_a_init, gru_b_init]
    outputs = [probabilities, gru_a_state, gru_b_state]

    return keras.models.Model(inputs, outputs)


def encoder(spectral_feats, pitch, frame_rate_feats):
    """Create the LPCNet encoder"""
    return keras.models.Model([spectral_feats, pitch], frame_rate_feats)


def frame_rate_network(spectral_feats, pitch, training=False):
    """Create the LPCNet frame-rate network"""

    ###########################################################################
    # Build
    ###########################################################################

    # Pitch embedding table
    pitch_embedding_table = keras.layers.Embedding(
        clpcnet.PITCH_BINS, 64, name='embed_pitch')

    # 1d convolutions
    conv_fn = functools.partial(keras.layers.Conv1D,
                                128,
                                3,
                                padding='valid' if training else 'same',
                                activation='tanh')
    conv1, conv2 = conv_fn(name='feature_conv1'), conv_fn(name='feature_conv2')

    # Dense layers
    dense_fn = functools.partial(keras.layers.Dense, 128, activation='tanh')
    dense1 = dense_fn(name='feature_dense1')
    dense2 = dense_fn(name='feature_dense2')

    ###########################################################################
    # Link
    ###########################################################################

    # Embed pitch
    pitch_embedding = Reshape((-1, 64))(pitch_embedding_table(pitch))

    # Join frame-rate features
    features = Concatenate()([spectral_feats, pitch_embedding])

    # Convolution layer forward pass
    activation = conv2(conv1(features))

    # Dense layer forward pass
    # Note: The residual connection shown in the paper was later found
    # to be harmful. Therefore, it is omitted.
    return dense2(dense1(activation))


def sample_rate_network(frame_rate_feats, sample_rate_feats, use_gpu=True):
    """Create the LPCNet sample-rate network"""

    ###########################################################################
    # Build
    ###########################################################################

    # PCM sample embedding table
    sample_rate_embedding_table = keras.layers.Embedding(
        clpcnet.PCM_LEVELS,
        clpcnet.EMBEDDING_SIZE,
        embeddings_initializer=sample_rate_embedding_initializer,
        name='embed_sig')

    # Upsampler
    repeat = keras.layers.Lambda(
        lambda x: K.repeat_elements(x, clpcnet.HOPSIZE, 1))

    # Get gru function based on compute
    if use_gpu:
        gru_fn = functools.partial(
            keras.layers.CuDNNGRU, return_sequences=True, return_state=True)
    else:
        gru_fn = functools.partial(keras.layers.GRU,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_activation='sigmoid',
                                   reset_after='true')

    # Gru layers
    gru_a = gru_fn(clpcnet.GRU_A_SIZE, name='gru_a')
    gru_b = gru_fn(clpcnet.GRU_B_SIZE, name='gru_b')

    # Dual fully-connected layer
    dual_dense = DualDense(
        clpcnet.PCM_LEVELS, activation='softmax', name='dual_fc')

    ###########################################################################
    # Link
    ###########################################################################

    # Embed Audio
    sample_rate_embedding = sample_rate_embedding_table(sample_rate_feats)
    sample_rate_embedding = \
        Reshape((-1, 3 * clpcnet.EMBEDDING_SIZE))(sample_rate_embedding)

    # Upsample the frame-rate features to the sampling rate
    upsampled = repeat(frame_rate_feats) # Residual connection ---------------
                                                                           # |
    # Concatenate sample-rate and upsampled frame-rate features            # |
    all_sample_rate_feats = Concatenate()(                                 # |
        [sample_rate_embedding, upsampled])                                # |
                                                                           # |
    # Add sample-rate gru A to graph.                                      # |
    activation = gru_a(all_sample_rate_feats)[0]                           # |
                                                                           # |
    # Residual connection between upsampled features and rnn output        # |
    # Note: this is NOT in the original LPCNet paper                       # |
    activation = Concatenate()([activation, upsampled]) # <-------------------

    # Add sample-rate gru B to graph
    activation = gru_b(activation)[0]

    # Add dual fully-connected layer to graph
    probabilities = dual_dense(activation)

    # Reuse components to build decoder model
    decoder_model = decoder(sample_rate_feats,
                            sample_rate_embedding,
                            gru_a,
                            gru_b,
                            dual_dense)

    return probabilities, decoder_model


###############################################################################
# Custom keras layer
###############################################################################


class DualDense(keras.layers.Layer):
    """Dual fully-connected layer"""

    channels = 2

    def __init__(self, output_size, activation=None, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.activation = keras.activations.get(activation)

        # Network weights
        self.kernel, self.bias, self.factor = None, None, None

    def build(self, input_shape):
        """Initialize the DualDense layer weights"""
        assert len(input_shape) >= 2

        # Kernel
        kernel_shape = (self.output_size, input_shape[-1], self.channels)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=keras.initializers.get('glorot_uniform'),
            regularizer=keras.regularizers.get(None),
            constraint=keras.constraints.get(None))

        # Bias
        bias_shape = (self.output_size, self.channels)
        self.bias = self.add_weight(
            name='bias',
            shape=bias_shape,
            initializer=keras.initializers.get('zeros'),
            regularizer=keras.regularizers.get(None),
            constraint=keras.constraints.get(None))

        # Learned scale factor
        self.factor = self.add_weight(
            name='factor',
            shape=bias_shape,
            initializer=keras.initializers.get('ones'),
            regularizer=keras.regularizers.get(None),
            constraint=keras.constraints.get(None))

    def call(self, inputs):
        """Forward pass through the DualDense layer"""
        # Pass through two linear maps
        output = K.dot(inputs, self.kernel) + self.bias

        # Scaled tanh nonlinearity
        output = K.tanh(output) * self.factor

        # Sum over the two channels of dual-dense layer
        output = K.sum(output, axis=-1)

        # Apply optional output activation
        return self.activation(output)


###############################################################################
# Custom embedding initializer
###############################################################################


def sample_rate_embedding_initializer(shape, dtype=None):
    """Initializer for the sample-rate feature embedding table"""
    # Get output shape
    shape = (np.prod(shape[:-1]), shape[-1])

    # Initialize as uniform noise in [-sqrt(3), sqrt(3)]
    weights = np.random.uniform(-1.7321, 1.7321, shape)

    # Add a unique offset to each weight such that the embedding
    # table is encouraged to be ordered
    line = np.arange(-.5 * shape[0] + .5, .5 * shape[0] - .4)
    line *= math.sqrt(12) / shape[0]
    return weights + np.reshape(line, (shape[0], 1))
