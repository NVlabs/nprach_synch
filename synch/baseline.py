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
Implements the baseline from [CHO].

[CHO] H. Chougrani, S. Kisseleff and S. Chatzinotas, "Efficient Preamble Detection and Time-of-Arrival Estimation for Single-Tone Frequency Hopping Random Access in NB-IoT," in IEEE Internet of Things Journal, vol. 8, no. 9, pp. 7437-7449, 1 May1, 2021, doi: 10.1109/JIOT.2020.3039004.
"""


import tensorflow as tf
from tensorflow.keras.layers import Layer
import sionna as sn
import numpy as np


class NPRACHSynch(Layer):
    # pylint: disable=line-too-long
    r"""NPRACHSynch(nprach_gen, fft_size, pfa, no)

    Implements NPRACH detection and synchronization algorithm from [CHO].

    This layer performs NPRACH detection: It identifies the UE from the preamble it transmits,
    estimates the time of arrival (ToA) and the channel frequency offset (CFO).

    [CHO] H. Chougrani, S. Kisseleff and S. Chatzinotas, "Efficient Preamble Detection and Time-of-Arrival Estimation for Single-Tone Frequency Hopping Random Access in NB-IoT," in IEEE Internet of Things Journal, vol. 8, no. 9, pp. 7437-7449, 1 May1, 2021, doi: 10.1109/JIOT.2020.3039004.

    Parameters
    -----------
    nprach_gen : NPRACH
        NPRACH generator.

    fft_size : int
        FFT size used for detection. Note that this is different from the DFT
        size used for the NPRACH waveform on the transmitter side.

    pfa : float
        Target probability of false positive.

    no : float
        Noise power spectral density

    Input
    ------
    y : [batch size, number of time steps], tf.complex
        Received samples.

    Output
    -------
    n_ue_hat : [batch size, max number of preambles], tf.float
        Tensor where for each preamble, 1 indicates that a device using this
        preamble was detected, and 0 indicates that no divice was detected.

    toa_est : [batch size, max number of preambles], tf.float
        Tensor of estimated ToA for detected devices.
        When no device is detected, i.e., when ``n_ue_hat`` is set to 0 for the
        corresponding entry, it contains 0.

    f_off_est : [batch size, max number of preambles], tf.float
        Tensor of estimated frequency offset for detected devices.
        When no device is detected, i.e., when ``n_ue_hat`` is set to 0 for the
        corresponding entry, it contains 0.
    """

    def __init__(self, nprach_gen, fft_size, pfa, no):
        super().__init__()

        self._nprach_gen = nprach_gen
        self._config = nprach_gen.config
        self._fft_size = fft_size

        self._build_detection_threshold(pfa, no)

    def call(self, y):

        batch_size = tf.shape(y)[0]
        config = self._config
        nprach_dft_size = config.nprach_dft_size
        num_sg = config.nprach_num_rep*config.nprach_sg_per_rep

        # Gathering the OFDM symbols without CP (sequences)
        # [batch_size, num of seq, dft size]
        y_seq = tf.gather(y, self._nprach_gen.seq_indices, axis=1)
        # Unitary DFT
        # [batch_size, num of seq, dft size]
        y_seq_freq = tf.signal.fft(y_seq)\
            /tf.complex(tf.cast(nprach_dft_size, tf.float32), 0.0)
        # Combine the sequences from a same SG
        # [batch_size, num_sg, dft size]
        y_sg = tf.reshape(y_seq_freq, [batch_size, num_sg,
                                       config.nprach_seq_per_sg,
                                       nprach_dft_size])
        y_sg = tf.reduce_sum(y_sg, axis=2)
        # Extract for every possible preamble the sequence of SGs
        # [batch_size, num preamble, num_sg]
        y_pr = self._extract_preamble_sg(y_sg, self._nprach_gen.freq_patterns)

        # Differential processing of neighboring SGs(III.A.1 of [2])
        # Build Y_{m+1}
        # Extract for every possible preamble the sequence of SGs
        # [batch_size, num preamble, num_sg]
        y_pr_del = tf.roll(y_pr, shift=-1, axis=2)
        # Differential processing of neighboring SGs
        # [batch_size, num preamble, num_sg-1]
        z = y_pr*tf.math.conj(y_pr_del)
        z = z[:,:,:-1]

        # Constructing vector v indexed by the hopping steps
        # [batch size, num preamble, max freq hop]
        freq_hop_steps = self._nprach_gen.freq_hop_steps
        max_hop = tf.reduce_max(tf.abs(freq_hop_steps))
        v_len = max_hop*2+1
        v = tf.zeros([config.nprach_num_sc*v_len, batch_size], tf.complex64)
        v_indices = freq_hop_steps + max_hop
        v_indices += tf.expand_dims(tf.range(config.nprach_num_sc)*v_len,
                                    axis=-1)
        v_indices = tf.reshape(v_indices, [-1, 1])
        z_ = tf.transpose(z, [1, 2, 0])
        z_ = tf.reshape(z_, [-1, batch_size])
        v = tf.tensor_scatter_nd_add(v, v_indices, z_)
        v = tf.reshape(v, [config.nprach_num_sc, v_len, batch_size])
        v = tf.transpose(v, [2, 0, 1])

        # v in frequency domain
        # [batch size, num preamble, fft size]
        v_freq = tf.concat([v, tf.zeros([batch_size, config.nprach_num_sc,
                            self._fft_size - v_len], tf.complex64)], axis=-1)
        v_freq = tf.signal.fft(v_freq)\
            /tf.complex(tf.constant(self._fft_size, tf.float32), 0.0)
        # Max frequency response
        # k_max and x_max: [batch size, num preamble]
        v_freq_abs = tf.abs(v_freq)
        k_max = tf.argmax(v_freq_abs, axis=-1)
        x_max = tf.gather(v_freq_abs, k_max, batch_dims=2)

        # Threshold-based device detection
        # [batch size, num preambles]
        tx_ue_hat = tf.greater(x_max, self._tau)

        # ToA estimation
        # [batch size, num preambles]
        toa_est = tf.cast(k_max, dtype=tf.float32)\
            /(tf.cast(self._fft_size, tf.float32)*config.delta_f_ra)
        # Quadratic interpolation to improve precision
        x_m = tf.gather(v_freq_abs, k_max - 1, batch_dims=2)
        x_p = tf.gather(v_freq_abs, k_max + 1, batch_dims=2)
        epsilon = 0.5*(x_p - x_m)/(2*x_max-x_m-x_p)
        toa_est = toa_est\
            + epsilon/(config.delta_f_ra*tf.cast(self._fft_size, tf.float32))

        # CFO estimation
        # CFO estimation is achieved by generating an estimate of v' from the
        # estimated ToA, without considering the amplitude factor (Q and |h|^2).
        # The scalar product of the estimate of v' and the vector v computed
        # previously is then calculated, which should results into a scalar
        # having 2\pi x f_off x samples_per_sg as phase (+ noise), where f_off
        # is normalized by the bandwidth.
        freq_pattern = tf.cast(self._nprach_gen.freq_hop_steps, tf.float32)
        freq_pattern = tf.expand_dims(freq_pattern, 0)
        z_hat_phase = 2.*sn.PI*tf.expand_dims(toa_est*config.bandwidth,axis=-1)\
            *freq_pattern/nprach_dft_size
        z_hat = tf.exp(tf.complex(0.0,z_hat_phase))
        z_hat = tf.transpose(z_hat, [1, 2, 0])
        z_hat = tf.reshape(z_hat, [-1, batch_size])
        v_prime_hat = tf.zeros([config.nprach_num_sc*v_len, batch_size],
                               tf.complex64)
        v_prime_hat = tf.tensor_scatter_nd_add(v_prime_hat, v_indices, z_hat)
        v_prime_hat = tf.reshape(v_prime_hat, [config.nprach_num_sc, v_len,
                                               batch_size])
        v_prime_hat = tf.transpose(v_prime_hat, [2, 0, 1])

        samples_per_seq = int(config.nprach_seq_duration*config.bandwidth)
        samples_per_cp = int(config.nprach_cp_duration*config.bandwidth)
        samples_per_sg = samples_per_cp+config.nprach_seq_per_sg*samples_per_seq
        f_off_est = tf.math.angle(tf.reduce_sum(v_prime_hat*tf.math.conj(v),
                                                axis=-1))/\
                                    (2.*sn.PI*samples_per_sg)

        return tx_ue_hat, toa_est, f_off_est

    def _build_detection_threshold(self, pfa, no):
        # pylint: disable=line-too-long
        """Builds the detection threshold value for a given noise power spectral
        density ``no`` and target probability of false positive ``pfa``.

        Parameters
        ----------
        pfa : float
            Target probability of false positive. Must be in (0,1).

        no : float
            Noise power spectral density.
        """
        batch_size = 10000
        num_it = 10
        noise_real_dev = tf.cast(tf.sqrt(0.5*no), tf.float32)
        x = []
        for _ in range(num_it):
            # Generate white noise as received signal
            y_pr_real = tf.random.normal([batch_size,
                self._config.nprach_num_sc,
                self._config.nprach_num_rep*self._config.nprach_sg_per_rep],
                                            stddev=noise_real_dev)
            y_pr_im = tf.random.normal([batch_size,
                self._config.nprach_num_sc,
                self._config.nprach_num_rep*self._config.nprach_sg_per_rep],
                                            stddev=noise_real_dev)
            y_pr = tf.complex(y_pr_real, y_pr_im)
            # Differential processing of neighboring SGs
            # Build Y_{m+1}
            # Extract for every possible preamble the sequence of SGs
            # [batch_size, num preamble, num_sg]
            y_pr_del = tf.roll(y_pr, shift=-1, axis=2)
            # Differential processing of neighboring SGs
            # [batch_size, num preamble, num_sg-1]
            z = y_pr*tf.math.conj(y_pr_del)
            z = z[:,:,:-1]
            # Constructing vector v indexed by the hopping steps
            # [batch size, num preamble, max freq hop]
            freq_hop_steps = self._nprach_gen.freq_hop_steps
            max_hop = tf.reduce_max(tf.abs(freq_hop_steps))
            v_len = max_hop*2+1
            v = tf.zeros([self._config.nprach_num_sc*v_len, batch_size],
                            tf.complex64)
            v_indices = freq_hop_steps + max_hop
            v_indices += tf.expand_dims(
                tf.range(self._config.nprach_num_sc)*v_len, axis=-1)
            v_indices = tf.reshape(v_indices, [-1, 1])
            z_ = tf.transpose(z, [1, 2, 0])
            z_ = tf.reshape(z_, [-1, batch_size])
            v = tf.tensor_scatter_nd_add(v, v_indices, z_)
            v = tf.reshape(v, [self._config.nprach_num_sc, v_len,
                                batch_size])
            v = tf.transpose(v, [2, 0, 1])
            # Compute Xmax
            v_freq = tf.concat([v,
                                tf.zeros([batch_size,
                                            self._config.nprach_num_sc,
                                            self._fft_size - v_len],
                                            tf.complex64)], axis=-1)
            v_freq = tf.signal.fft(v_freq)\
                /tf.complex(tf.constant(256, tf.float32), 0.0)
            v_freq_abs = tf.abs(v_freq)
            k_max = tf.argmax(v_freq_abs, axis=-1)
            x_max = tf.gather(v_freq_abs, k_max, batch_dims=2)
            x_max = x_max.numpy()
            x.append(x_max)
        x = np.concatenate(x, axis=0)
        tau = np.quantile(x, pfa, axis=0)
        self._tau = tf.constant(tau, tf.float32)

    def _extract_preamble_sg(self, y, indices):
        # pylint: disable=line-too-long
        """Extract from ``y`` and for every possible preamble the sequence of
        SGs following the pattern defined by `indices`.

        Input
        -----
        y : [batch_size, num_sg, dft size], tf.complex
            Received resource grid of SGs.
            The sequences froming a same SG are assumed to be already combined.

        indices : [num preambles, num_sg], tf.int
            For every possible preamble, the sequence of subcarrier indices
            for the corresponding sequence of SGs.

        Output
        -------
        y_pr : [batch_size, num preamble, num_sg], tf.complex
            For every possible preamble, the corresponding received sequence
            of SGs.
        """
        y = tf.transpose(y, [1, 2, 0])
        indices = tf.transpose(indices)
        y_pr = tf.gather(y, indices, batch_dims=1)
        y_pr = tf.transpose(y_pr, [2, 1, 0])
        return y_pr
