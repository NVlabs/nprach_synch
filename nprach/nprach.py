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
NPRACH waveform.
Only preamble configuration 0 is handled.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import sionna as sn
import numpy as np
import scipy.signal


class NPRACHConfig:
    # pylint: disable=line-too-long
    r"""NPRACHConfig(cell_id=1, nprach_sc_offset=0, nprach_num_sc=48, nprach_num_rep=1)

    Convenient class for storing the NPRACH configuration.
    Only preamble configuration 0 is handled.

    Parameters
    -----------
    cell_id : int
        Cell ID. Must be positive. Defaults to 1.

    nprach_sc_offset : int
        Frequency location (subcarrier index) of the first subcarrier allocated
        to NPRACH. Defaults to 0.

    nprach_num_sc : int
        Number of subcarriers allocated to NPRACH.

    nprach_num_rep : int
        Number of preamble repetitions forming the NPRACH.
        Must be one of 1, 2, 4, 8, 16, 32, 64, or 128.
        Defaults to 1.
    """

    def __init__(self, cell_id=1, nprach_sc_offset=0, nprach_num_sc=48,
                    nprach_num_rep=1):

        assert cell_id > 0, "Invalid value for ``cell_id``"

        assert nprach_num_rep in (1, 2, 4, 8, 16, 32, 64, 128),\
            "Invalid value for ``nprach_num_rep``"

        assert nprach_sc_offset + nprach_num_sc <= 48,\
            "``nprach_sc_offset + nprach_num_sc`` must be lower or equal to 48"

        self._cell_id = cell_id
        self._nprach_sc_offset = nprach_sc_offset
        self._nprach_num_rep = nprach_num_rep
        self._nprach_num_sc = nprach_num_sc

    @property
    def ts(self):
        "Basic time unit [s]"
        return 1./(15000*2048)

    @property
    def bandwidth(self):
        "Bandwidth [Hz]"
        return 180e3

    @property
    def cell_id(self):
        """Cell Id"""
        return self._cell_id

    @property
    def delta_f_ra(self):
        "Subcarrier spacing for NPRACH [Hz]"
        return 3.75e3

    @property
    def nprach_dft_size(self):
        """Number of subcarriers for NPRACH"""
        return 48

    @property
    def nprach_cp_duration(self):
        "NPRACH CP duration [s]"
        return 2048.*self.ts

    @property
    def nprach_seq_duration(self):
        "NPRACH symbol duration [s]"
        return 8192.*self.ts

    @property
    def nprach_seq_per_sg(self):
        """Number of symbols per symbol group (SG)"""
        return 5

    @property
    def nprach_sg_per_rep(self):
        """Number pf symbol groups (SGs) per repetition"""
        return 4

    @property
    def nprach_sg_duration(self):
        """Duration of a symbol group [s]"""
        return (self.nprach_cp_duration
                    + self.nprach_seq_per_sg*self.nprach_seq_duration)

    @property
    def nprach_sc_offset(self):
        """First subcarrier allocated to the NPRACH"""
        return self._nprach_sc_offset

    @property
    def nprach_num_rep(self):
        """Number of repetitions forming the preamble"""
        return self._nprach_num_rep

    @property
    def nprach_num_sc(self):
        """Number of subcarriers allocated to the NPRACH"""
        return self._nprach_num_sc

class GoldSequence:
    # pylint: disable=line-too-long
    r"""GoldSequence(c_init)

    Gold-sequence generator as in TS 36.211, Section 7.2.

    Parameters
    -----------
    c_init: int
        Seed

    Input
    ------
    length : int
        Length of the binary sequence to generate.

    Output
    -------
    seq : np.array
        Binary array of size `length`.
    """

    def __init__(self, c_init):

        self._c_init = c_init
        self.reset()

    def reset(self):
        """Reset the internal state."""

        x1_init = [1] + [0]*30 # x1 m-sequence initial state
        x2_init = np.flip([int(i) for i in np.binary_repr(self._c_init,
                                                            width=31)])

        _, state_x1 = scipy.signal.max_len_seq(31, x1_init,
                                                1600, [0, 3, 31])
        _, state_x2 = scipy.signal.max_len_seq(31, x2_init,
                                                1600, [0, 1, 2, 3, 31])
        self._state_x1 = state_x1
        self._state_x2 = state_x2

    def __call__(self, length):
        x1, state_x1 = scipy.signal.max_len_seq(31, self._state_x1,
                                                    length, [0, 3, 31])
        x2, state_x2 = scipy.signal.max_len_seq(31, self._state_x2,
                                                    length, [0, 1, 2, 3, 31])
        self._state_x1 = state_x1
        self._state_x2 = state_x2

        seq = np.bitwise_xor(x1, x2)
        return seq

class NPRACHPattern:
    # pylint: disable=line-too-long
    r"""NPRACHPattern(config)

    Compute the frequency hopping pattern for the symbol
    groups (SGs) forming an NPRACH from an initial subcarrier index `n_init`.

    Only preamble format 0 is supported.

    Parameters
    -----------
    config : NPRACHConfig
        NPRACH configuration.

    Input
    ------
    n_init : int
        Initial subcarrier for the frequency hopping pattern.
        This value determines the frequency hopping pattern, and
        therefore the preamble structure.
        Different devices accessing the base station concurrently must use
        different initial subcarrier to avoid colliding.

    Output
    -------
    freq_pattern: [number of SG], int
        Subcarrier index for all SGs.
    """

    def __init__(self, config):

        self._config = config
        self._N_C_RA = 12 # Constant from the standard
        # Gold-sequence generator
        self._gold_seq = GoldSequence(config.cell_id)

    def _f_t(self):
        """
        Function f(t) as required in TS 36.211, section 10.1.6.1
        """
        # Goes from 10*t+1 to 10*t+9 (included) -> skip 10*t
        c = self._gold_seq(10)[1:]
        p = np.sum(c*np.power(2, np.arange(9))) % (self._N_C_RA-1)
        f_t = (self._f_t_prev + p + 1) % self._N_C_RA
        self._f_t_prev = f_t

        return f_t

    def _n_ra_sc(self, i):
        """
        Returns the frequency hop offset for the SG with index ``i``.
        See TS 36.211.

        Input
        ------
        i : int
            SG index

        Output
        -------
        : int
            Subcarrier index for the SG with index ``i``
        """

        if (i % 4) == 0 :
            if i > 0:
                n_ra_sc_tilde = ( (self._n_nc_ra_tilde_0 + self._f_t())
                                    % self._N_C_RA )
            else:
                n_ra_sc_tilde = self._n_nc_ra_tilde_0
        elif (i % 4) in (1,3):
            prev = self._n_ra_sc_tilde_prev
            if (prev % 2) == 0:
                n_ra_sc_tilde  = prev + 1
            else:
                n_ra_sc_tilde = prev - 1
        elif (i % 4) == 2:
            prev = self._n_ra_sc_tilde_prev
            if prev < 6:
                n_ra_sc_tilde  = prev + 6
            else:
                n_ra_sc_tilde = prev - 6

        self._n_ra_sc_tilde_prev = n_ra_sc_tilde
        return n_ra_sc_tilde

    def __call__(self, n_init):

        ###################################
        # Reset internal state
        ###################################
        # Initial frequency hop
        self._n_nc_ra_tilde_0 = n_init % self._N_C_RA
        # Starting subcarriers before initial hopping
        n_start = (self._config.nprach_sc_offset
                    + (n_init//self._N_C_RA)*self._N_C_RA)
        # Setting-up some internal state variables
        self._f_t_prev = 0
        # Reset Gold-sequence
        self._gold_seq.reset()

        ###################################
        # Build the frequency pattern
        ###################################
        num_sg = self._config.nprach_sg_per_rep*self._config.nprach_num_rep
        freq_pattern = np.zeros(num_sg, int)
        freq_hops = np.zeros(num_sg-1, int)
        for i in range(num_sg):
            freq_pattern[i] = n_start + self._n_ra_sc(i)
            if i > 0:
                freq_hops[i-1] = freq_pattern[i] - freq_pattern[i-1]
        return freq_pattern, freq_hops

class NPRACH(Layer):
    # pylint: disable=line-too-long
    r"""NPRACH(config)

    This layer implements the NPRACH waveform.

    Only the frequency hopping pattern of preamble format 0 is supported.

    Parameters
    -----------
    config : NPRACHConfig
        NPRACH configuration

    Input
    ------
    tx_power : [batch_size, max number of users], tf.float
        Tensor of user transmit powers.
        Set to 0 for inactive users.

    Output
    ------
    : [batch_size, num_time_samples, max num users]
        NPRACH waveform time-domain samples.
    """

    def __init__(self, config):
        super().__init__()

        self._config = config

        ## Generate the frequency patterns
        freq_patterns = []
        freq_hops = []
        freq_pattern_generator = NPRACHPattern(config)
        for n_init in range(config.nprach_num_sc):
            p,h = freq_pattern_generator(n_init)
            freq_patterns.append(p)
            freq_hops.append(h)
        freq_patterns = tf.stack(freq_patterns, axis=0)
        freq_hops = tf.stack(freq_hops, axis=0)
        self._freq_patterns = freq_patterns
        self._freq_hops = freq_hops

    @property
    def config(self):
        return self._config

    @property
    def seq_indices(self):
        # pylint: disable=line-too-long
        r"""
        Tensor with shape [number of sequences, DFT size] of indices
        of time samples corresponding to sequences of the NPRACH.
        Useful to extract the sequences forming the NPRACH without the CP.
        """
        config = self._config
        samples_per_seq = int(config.nprach_seq_duration*config.bandwidth)
        seq_indices = tf.range(samples_per_seq*config.nprach_seq_per_sg)
        # Shift due to CP and repetitions
        samples_per_cp = int(config.nprach_cp_duration*config.bandwidth)
        samples_per_sg = samples_per_cp\
                                    + config.nprach_seq_per_sg*samples_per_seq
        indices_shift= tf.range(config.nprach_num_rep*config.nprach_sg_per_rep)\
                            *samples_per_sg + samples_per_cp
        # Combining seq indices and shifts
        seq_indices = tf.expand_dims(seq_indices, 0)\
                                        + tf.expand_dims(indices_shift, 1)
        seq_indices = tf.reshape(seq_indices, [-1, samples_per_seq])
        return seq_indices

    @property
    def freq_patterns(self):
        return self._freq_patterns

    @property
    def freq_hop_steps(self):
        return self._freq_hops

    def call(self, tx_power):

        batch_size = tf.shape(tx_power)[0]

        # NRPACH config
        config = self._config

        # DFT size
        dft_size = tf.cast(config.nprach_dft_size, tf.float32)
        # Subcarrier spacing
        delta_f = config.delta_f_ra
        # CP length normalized by DFT size
        cp_length_norm = config.nprach_cp_duration*delta_f*dft_size
        # Number of time steps per SG
        num_time_steps_sg = tf.cast(config.nprach_sg_duration*config.bandwidth,
                                tf.int32)

        # Time steps
        # [1, num_time_steps_sg, 1]
        time_steps = tf.range(num_time_steps_sg, dtype=tf.float32)
        time_steps = tf.expand_dims(tf.expand_dims(time_steps, axis=0), axis=2)
        # [1, 1, num_time_steps_sg, max number of users]
        phase = (time_steps-cp_length_norm)/dft_size
        phase = tf.expand_dims(phase, axis=1)
        # [1, number of SGs, 1, max number of users]
        freq_pattern = tf.transpose(self._freq_patterns, [1, 0])
        freq_pattern = tf.cast(freq_pattern, tf.float32)
        freq_pattern = tf.expand_dims(freq_pattern, 0)
        freq_pattern = tf.expand_dims(freq_pattern, 2)
        # [1, number of SGs, num_time_steps_sg, max number of users]
        phase = 2.*sn.PI*phase*freq_pattern
        s = tf.exp(tf.complex(0.0, phase))
        # [batch_size, preamble_length = number of SGs*num_time_steps_sg,
        #   max num users]
        s = tf.reshape(s, [1, -1, config.nprach_num_sc])
        s = tf.expand_dims(tf.complex(tf.sqrt(tx_power), 0.0), axis=1)*s
        return s
