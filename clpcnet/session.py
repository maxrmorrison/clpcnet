import contextlib
import os
import sys

# Import keras without printing backend
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

import tensorflow as tf

import clpcnet


###############################################################################
# Tensorflow session management
###############################################################################


class Session:

    def __init__(self, file, gpu=None):
        self.file = file
        self.gpu = gpu

        # Tensorflow setup
        if gpu is None:
            config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        else:
            gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
            config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

        self.session = tf.compat.v1.Session(config=config)
        self.graph = tf.compat.v1.get_default_graph()

        # Keras setup
        keras.backend.set_session(self.session)

        # Device management
        device = 'CPU' if gpu is None else 'GPU'
        number = '0' if gpu is None else str(gpu)
        self.device = f'/{device}:{number}'

        # Build LPCNet
        model, encoder, decoder = clpcnet.model(use_gpu=gpu is not None)
        optimizer = keras.optimizers.Adam(clpcnet.LEARNING_RATE,
                                          amsgrad=True,
                                          decay=clpcnet.WEIGHT_DECAY)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])

        # Load pretrained weights
        model.load_weights(file)

        # Bind model components for inference
        self.encoder = encoder
        self.decoder = decoder

    @contextlib.contextmanager
    def context(self):
        """Context manager for tensorflow setup"""
        with tf.device(self.device):
            with self.graph.as_default():
                yield
