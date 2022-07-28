# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Deep learning-based NPRACH synchronization.
"""


import tensorflow as tf
from tensorflow.keras.layers import Layer, SeparableConv1D, Dense, LayerNormalization
from tensorflow.keras.activations import relu
import sionna as sn


class ResnetBlock(Layer):
    # pylint: disable=line-too-long
    """
    ResnetBlock(num_kernels, kernel_size, normalization_axis)

    ResNet block. See Figure 2 of [AIT].

    [AIT] https://arxiv.org/abs/2205.10805

    Parameters:
    ------------
    num_kernels : int
        Number of kernels forming the separable convolutional layers.

    kernel_size : int
        Kernel size for the separable convolutional layers.

    normalization_axis : int or list of int
        Axes along which to perform layer normalization.

    Input
    ------
    : [..., s, n], tf.float
        Input sequences of length ``s`` and with ``n`` features.

    Output
    -------
    : [..., s, num_kernels], tf.float
        Processed input.
    """

    def __init__(self, num_kernels, kernel_size, normalization_axis):
        super().__init__()

        self._kernel_size = kernel_size
        self._num_kernels = num_kernels

        self._norm_1 = LayerNormalization(axis=normalization_axis)
        self._conv_1 = SeparableConv1D(num_kernels, kernel_size, padding='same')

        self._norm_2 = LayerNormalization(axis=normalization_axis)
        self._conv_2 = SeparableConv1D(num_kernels, kernel_size, padding='same')

    def call(self, inp):

        z = inp

        z = self._norm_1(z)
        z = relu(z)
        z = self._conv_1(z)

        z = self._norm_2(z)
        z = relu(z)
        z = self._conv_2(z)

        z = z + inp
        return z

class DeepNSynch(Layer):
    # pylint: disable=line-too-long
    """
    DeepNSynch(nprach_gen)

    Implementation of the model proposed in [AIT].

    [AIT] https://arxiv.org/abs/2205.10805

    Parameters:
    ------------
    nprach_gen : NPRACH
        NPRACH generator.

    Input
    ------
    y : [batch size, number of time steps], tf.complex
        Received samples.

    Output
    -------
    n_ue_hat : [batch size, max number of preambles], tf.float
        Tensor where for each preamble, a soft decision that a device using this
        preamble was detected is provided.
        The soft decisions are given as logits, i.e., unnormalized
        log-probabilities.

    toa_est : [batch size, max number of preambles], tf.float
        Tensor of estimated ToAs.

    f_off_est : [batch size, max number of preambles], tf.float
        Tensor of estimated frequency offsets.
    """

    def __init__(self, nprach_gen):
        super().__init__()

        self._nprach_gen = nprach_gen
        self._config = nprach_gen.config

        self._input_norm = LayerNormalization(axis=(2,3), scale=False,
                                                                center=False)

        ################################################
        # Detection network
        ################################################
        self._act_conv_rg = SeparableConv1D(128, 3, padding='same')
        self._act_rnb_rg_1 = ResnetBlock(128, 3, normalization_axis=(1,2))
        self._act_rnb_rg_2 = ResnetBlock(128, 3, normalization_axis=(1,2))
        self._act_rnb_rg_3 = ResnetBlock(128, 3, normalization_axis=(1,2))
        #
        self._dense_act_1 = Dense(1024, activation='relu')
        self._dense_act_2 = Dense(512, activation='relu')
        self._dense_act_3 = Dense(256, activation='relu')
        self._dense_act_4 = Dense(1, activation=None)

        ################################################
        # CFO and ToA estimation
        ################################################
        self._est_conv_rg = SeparableConv1D(128, 3, padding='same')
        self._est_rnb_rg_1 = ResnetBlock(128, 3, normalization_axis=(1,2))
        self._est_rnb_rg_2 = ResnetBlock(128, 3, normalization_axis=(1,2))
        self._est_rnb_rg_3 = ResnetBlock(128, 3, normalization_axis=(1,2))
        # ToA
        self._dense_toa_1 = Dense(1024, activation='relu')
        self._dense_toa_2 = Dense(512, activation='relu')
        self._dense_toa_3 = Dense(256, activation='relu')
        self._dense_toa_4 = Dense(1, activation=None)
        # CFO
        self._dense_cfo_1 = Dense(1024, activation='relu')
        self._dense_cfo_2 = Dense(512, activation='relu')
        self._dense_cfo_3 = Dense(256, activation='relu')
        self._dense_cfo_4 = Dense(1, activation=None)

    def call(self, inputs):

        y = inputs
        batch_size = tf.shape(y)[0]
        config = self._config
        nprach_dft_size = self._config.nprach_dft_size
        num_sg = config.nprach_sg_per_rep*config.nprach_num_rep

        #############################################################
        # Time to frequency domain to build the resource grid
        #############################################################
        # Gathering the OFDM symbols without CP (sequences)
        # [batch_size, num of seq, dft size]
        y_seq = tf.gather(y, self._nprach_gen.seq_indices, axis=1)
        # Unitary DFT
        # [batch_size, num of seq, dft size]
        y_seq_freq = tf.signal.fft(y_seq)\
            /tf.complex(tf.cast(config.nprach_dft_size, tf.float32), 0.0)

        ################################################################
        # Combining over sequences by averaging
        ################################################################
        # [batch size, number of sg, dft size]
        z = tf.reshape(y_seq_freq, [batch_size, num_sg,
                                config.nprach_seq_per_sg, config.nprach_num_sc])
        z = tf.reduce_mean(z, axis=2)

        ################################################################
        # C2R
        ################################################################
        # [batch_size, number of sg, dft size, 2]
        z = tf.stack([tf.math.real(z), tf.math.imag(z)], axis=-1)

        ################################################################
        # Extracting REs according to hop patterns for normalization
        ################################################################
        # [batch_size, num preamble, number of sg, 2]
        z = self._extract_preamble_seq(z, self._nprach_gen.freq_patterns)

        #############################################################
        # Normalization
        #############################################################
        # [batch_size, num preamble, 1, 1]
        energy = tf.reduce_mean(tf.square(z), axis=(2,3), keepdims=True)
        # [batch_size, num preamble, number of sg, 2]
        z = self._input_norm(z)

        #############################################################
        # Log-scale for concatenation
        #############################################################
        # [batch_size, num preamble, number of sg, 3]
        energy = tf.tile(energy, [1, 1, num_sg, 1])
        z = tf.concat([z, sn.utils.log10(energy)], axis=-1)

        #############################################################
        # Scattering the normalized REs back to the RG
        #############################################################
        # [batch_size, number of sg, dft size, 3]
        z = self._preamble_seq_to_rg(z, self._nprach_gen.freq_patterns)

        #############################################################
        # User detection
        #############################################################
        # [batch_size, number of sg, dft size, dim]
        z_a =  z
        z_a = tf.reshape(z_a, [batch_size*num_sg, nprach_dft_size, 3])
        z_a = self._act_conv_rg(z_a)
        z_a = self._act_rnb_rg_1(z_a)
        z_a = self._act_rnb_rg_2(z_a)
        z_a = self._act_rnb_rg_3(z_a)
        z_a = tf.reshape(z_a, [batch_size, num_sg, nprach_dft_size,
                            tf.shape(z_a)[-1]])
        # Extracting REs according to hop pattern
        # [batch_size, num preamble, number of sg, dim]
        z_a = self._extract_preamble_seq(z_a, self._nprach_gen.freq_patterns)
        # [batch_size, num preamble]
        z_a = tf.reshape(z_a, [batch_size, config.nprach_num_sc, -1])
        z_a = self._dense_act_1(z_a)
        z_a = self._dense_act_2(z_a)
        z_a = self._dense_act_3(z_a)
        z_a = self._dense_act_4(z_a)
        z_a = z_a[...,0]

        ##############################################################
        # Computing ToA and estimate
        #############################################################
        p = z
        p = tf.reshape(p, [batch_size*num_sg, nprach_dft_size, 3])
        p = self._est_conv_rg(p)
        p = self._est_rnb_rg_1(p)
        p = self._est_rnb_rg_2(p)
        p = self._est_rnb_rg_3(p)
        p = tf.reshape(p, [batch_size, num_sg, nprach_dft_size,
                        tf.shape(p)[-1]])
        # Extracting REs according to hop pattern
        # [batch_size, num preamble, number of sg, dim]
        p = self._extract_preamble_seq(p, self._nprach_gen.freq_patterns)
        # ToA
        z_toa = p
        z_toa = tf.reshape(z_toa, [batch_size, config.nprach_num_sc, -1])
        z_toa = self._dense_toa_1(z_toa)
        z_toa = self._dense_toa_2(z_toa)
        z_toa = self._dense_toa_3(z_toa)
        z_toa = self._dense_toa_4(z_toa)
        z_toa = z_toa[...,0]
        # CFO
        z_cfo = p
        z_cfo = tf.reshape(z_cfo, [batch_size, config.nprach_num_sc, -1])
        z_cfo = self._dense_cfo_1(z_cfo)
        z_cfo = self._dense_cfo_2(z_cfo)
        z_cfo = self._dense_cfo_3(z_cfo)
        z_cfo = self._dense_cfo_4(z_cfo)
        z_cfo = z_cfo[...,0]

        return z_a, z_toa, z_cfo

    def _extract_preamble_seq(self, y, indices):
        # pylint: disable=line-too-long
        """
        Extract from `y` and for every possible preamble the sequence of REs
        following the pattern defined by `indices`.

        Input
        -----
        y : [batch_size, num_sg, dft size, dim], tf.complex or tf.float
            Received resource grid.

        indices : [num preambles, num_sg], tf.int
            For every possible preamble, the sequence of subcarrier indices
            for the corresponding sequence of SGs.

        Output
        -------
        y_pr : [batch_size, num preamble, num_sg, dim], tf.complex
            For every possible preamble, the corresponding received
            sequence of SGs.
        """
        # Expand to sequences
        indices = tf.transpose(indices)
        y = tf.transpose(y, [1, 2, 0, 3])
        y_pr = tf.gather(y, indices, batch_dims=1)
        y_pr = tf.transpose(y_pr, [2, 1, 0, 3])
        return y_pr

    def _preamble_seq_to_rg(self, y_pr, indices):
        # pylint: disable=line-too-long
        """
        Extract from `y` and for every possible preamble the sequence of REs
        following the pattern defined by `indices`.

        Input
        -----
        y_pr : [batch_size, num preamble, num_sg, dim], tf.complex
            For every possible preamble, the corresponding received
            sequence of SGs.

        indices : [num preambles, num_sg], tf.int
            For every possible preamble, the sequence of subcarrier
            indices for the corresponding sequence of SGs.

        Output
        -------
        y : [batch_size, num_sg, dft size, dim], tf.complex or tf.float
            Resource grid.
        """
        # Expand to sequences
        num_sg = tf.shape(indices)[1]
        num_pr = tf.shape(indices)[0]
        batch_size = tf.shape(y_pr)[0]
        dim = tf.shape(y_pr)[-1]

        indices_2 = tf.transpose(indices)
        indices_1 = tf.tile(tf.expand_dims(tf.range(num_sg, dtype=tf.int32),
                                                            axis=1), [1, num_pr])
        indices = tf.stack([indices_1, indices_2], axis=-1)
        y_pr = tf.transpose(y_pr, [2, 1, 0, 3])
        y = tf.scatter_nd(indices, y_pr,
                        [num_sg, self._config.nprach_dft_size, batch_size, dim])
        y = tf.transpose(y, [2, 0, 1, 3])
        return y
