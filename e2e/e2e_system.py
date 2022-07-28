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
Model for simulating the end-to-end system: NPRACH generation, channel, and
detection.

This model is used for both training and evaluation.
"""


import sys
sys.path.append('..')
import tensorflow as tf
import sionna as sn
from tensorflow.keras import Model
from . import CFO
from nprach import NPRACHConfig, NPRACH
from synch import DeepNSynch, NPRACHSynch
from parameters import *


class E2E(Model):
    # pylint: disable=line-too-long
    """
    E2E(system, training, nprach_num_rep, nprach_num_sc, fft_size=256, pfa=0.999)

    Model for implementing the end-to-end model. This model uses the 3GPP UMi
    channel model with CFO, and is used for both training and evaluating the
    NPRACH synchronization methods.

    Parameters:
    ------------
    system : one of ('baseline', 'dl'), str
        The synchronization method to use.
        'baseline' ->  synch.NPRACHSynch
        'dl' -> synch.DeepNSynch

    training : boolean
        Set to `true` if training the deep learning-based method.
        In this case, the carrier frequency offset (CFO) and transmit
        probabilites are randomly sampled and do not need to be provided.

    nprach_num_rep : int
        Number of repetitions forming the preamble.

    nprach_num_sc : int
        Number of subcarriers allocated to the preamble.

    fft_size : int
        FFT size for the internal computation of the baseline.
        Unused by the deep learning method.
        Defaults to 256.

    pfa : float
        Target probability of false positive for the baseline. Must be in (0,1).
        Unused by the deep learning method.
        Defaults to 0.999.

    Input
    ------
    batch_size : int
        Batch size for simulation.

    max_cfo_ppm : float
        CFO [in PPM] are uniformly and randomly sampled from
        ``(-max_cfo_ppm, max_cfo_ppm)`` during evaluation, i.e.,
        when ``training = False``.
        At training, i.e., when ``training = True``, the CFO is randomly and
        uniformly sampled from ``(-MAX_CFO_PPM_TRAIN, MAX_CFO_PPM_TRAIN)``,
        where ``MAX_CFO_PPM_TRAIN`` is specified in the ``parameters`` module.

    ue_prob : float, must be in (0,1)
        Probability that a preamble is used by a user during evaluation.
        Must be in (0,1). At training, i.e.,  when ``training = True``, the
        probability for a user to transmit is randomly and uniformly sampled
        from `(0,1)`.

    Output
    -------
    During training (i.e., ``training = True``):

    loss_tx_ue : float
        Loss function for training the detection of active preambles.
        See (10) in [AIT].

    loss_toa : float
        Loss function for training the estimation of time-of-arrivals.
        See (11) in [AIT].

    loss_cfo : float
        Loss function for training the estimation of CFO.
        See (11) in [AIT].

    During evaluation (i.e., ``training = False``):

    snr : [batch_size, max_num_users], tf.float
        SNR (linear) for every user, defined in (12) of [AIT].
        SNR equals 0 for inactive users.

    toa : [batch_size, max_num_users], tf.float
        Time-of-arrival normalized by the cyclic prefix ducation for every user.
        Set to 0 for non-active users.

    f_off : [batch_size, max_num_users], tf.float
        Frequency offset normalized by the maximum allowed value.
        set to 0 for non-active users.

    ue_prob : [batch_size, max_num_users], tf.float
        Probability of transmission.

    fpr : [batch_size, max_num_users], tf.float
        False positives. Set to 1 if there is a false positive,
        0 if there is not, and -1 if not applicable, i.e., the user
        did transmit.

    fnr : [batch_size, max_num_users], tf.float
        False negatives. Set to 1 if there is a false negative, 0 if there
        is not, and -1 if not applicable, i.e., the user did not transmit.

    toa_err : [batch_size, max_num_users], tf.float
        Time-of-arrival estimation error normalized by the cyclic prefix
        duration. Set to 0 for inactive users.

    f_off_err : [batch_size, max_num_users], tf.float
        CFO estimation error normalized by the bandwidth.
        Set to 0 for inactive users.


    [AIT] https://arxiv.org/abs/2205.10805
    """

    def __init__(   self,
                    system,
                    training,
                    nprach_num_rep,
                    nprach_num_sc,
                    fft_size=256,
                    pfa=0.999):
        super().__init__()

        assert system in ('baseline', 'dl')

        self.system = system
        self.training = training
        self.no = tf.pow(10.0, N0_DB/10.0)

        # Transmitter
        self.config = NPRACHConfig( nprach_num_rep=nprach_num_rep,
                                    nprach_num_sc=nprach_num_sc)
        self.max_ues = self.config.nprach_num_sc
        self.nprach = NPRACH(self.config)

        # Channel
        self.num_time_samples = int(self.config.nprach_sg_duration\
                                        *self.config.bandwidth)*4*nprach_num_rep
        self.cfo = CFO(CARRIER_FREQ, SAMPLING_FREQUENCY)
        # UT and BS antennas
        bs_array = sn.channel.tr38901.Antenna(polarization = 'single',
                                    polarization_type  = 'V',
                                    antenna_pattern = 'omni',
                                    carrier_frequency = CARRIER_FREQ)
        ut_array = sn.channel.tr38901.Antenna(polarization = 'single',
                                    polarization_type = 'V',
                                    antenna_pattern = 'omni',
                                    carrier_frequency = CARRIER_FREQ)
        # Instantiating UMi channel model
        self.channel_model = sn.channel.tr38901.UMi(
                                 carrier_frequency = CARRIER_FREQ,
                                 o2i_model = 'low',
                                 ut_array = ut_array,
                                 bs_array = bs_array,
                                 direction = 'uplink')
        # Apply time-domain channel
        self.apply_channel = sn.channel.ApplyTimeChannel(self.num_time_samples,
                                                        MAX_L-MIN_L+1)

        if system == 'baseline':
            self.synch = NPRACHSynch(self.nprach, fft_size, pfa, self.no)
        elif system == 'dl':
            self.synch = DeepNSynch(self.nprach)

        # Losses
        if training:
            self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.mse_toa = tf.keras.losses.MeanSquaredError()
            self.mse_cfo = tf.keras.losses.MeanSquaredError()

    def call(self, batch_size, max_cfo_ppm=None, ue_prob=None):

        #############################################
        # Transmitter
        #############################################
        # Select random devices ID
        if self.training:
            ue_prob = tf.random.uniform([batch_size, 1], minval=0.0,
                                            maxval=1.0, dtype=tf.float32)
        else:
            ue_prob = tf.fill([batch_size, 1], ue_prob)
        ue_prob = tf.tile(ue_prob, [1, self.max_ues])
        tx_ue_ = tf.random.uniform([batch_size, self.max_ues], minval=0.0,
                                    maxval=1.0, dtype=tf.float32)
        tx_ue_ = tf.less(tx_ue_, ue_prob)
        tx_ue = tf.cast(tx_ue_, tf.bool)
        tx_power_db = tf.fill([batch_size, self.max_ues], TX_POWER_DB)
        tx_power = tf.pow(10.0, tx_power_db/10.0)
        tx_power = tf.where(tx_ue, tx_power, 0.0)
        # Generate NPRACH
        s = self.nprach(tx_power)

        ############################################
        # Channel
        ############################################
        # Generate random ToA
        # Minimum ToA is set to 0.1us
        toa = tf.random.uniform([batch_size,self.max_ues], minval=1e-7,
                                maxval=self.config.nprach_cp_duration) # s
        # Generate random PPM
        if self.training:
            cfo_ppm = tf.random.uniform([batch_size,self.max_ues],
                            minval=-MAX_CFO_PPM_TRAIN, maxval=MAX_CFO_PPM_TRAIN)
        else:
            cfo_ppm = tf.random.uniform([batch_size,self.max_ues],
                                    minval=-max_cfo_ppm, maxval=max_cfo_ppm)
        # Apply CFO
        s = self.cfo((s, cfo_ppm))
        # Generate and set topology
        topology = sn.channel.utils.gen_single_sector_topology(
                                            batch_size = batch_size,
                                            num_ut = self.max_ues,
                                            scenario = 'umi',
                                            max_ut_velocity=MAX_SPEED)
        self.channel_model.set_topology(*topology)
        # Generate channel CIR
        a,tau = self.channel_model( self.num_time_samples+MAX_L-MIN_L,
                                    self.config.bandwidth)
        # Add ToA to CIR
        tau = tau + tf.expand_dims(tf.expand_dims(toa, axis=1), axis=3)
        s = tf.expand_dims(tf.transpose(s, [0,2,1]), axis=-2)
        # Compute channel taps
        h_time = sn.channel.cir_to_time_channel(self.config.bandwidth,
                                                a, tau, MIN_L, MAX_L)
        y = self.apply_channel((s, h_time, self.no))
        # Keep only relevant samples
        y = tf.squeeze(y, axis=(1,2))

        ############################################
        # Receiver
        ############################################
        # Detection of the users, ToA and CFO estimation
        tx_ue_hat, toa_est, f_off_est = self.synch(y)

        ############################################
        # Losses and metric calculation
        ############################################
        if self.training:
            # Binary CE for detecting UE
            loss_tx_ue = self.bce(y_true=tf.cast(tx_ue_, tf.float32),
                                            y_pred=tx_ue_hat)
            # SNR calculation for loss weigting
            freq = sn.channel.subcarrier_frequencies(
                        self.config.nprach_dft_size, self.config.delta_f_ra)
            h_freq = sn.channel.cir_to_ofdm_channel(freq, a, tau)[:,0,0,:,0,:,:]
            h_sq = tf.reduce_mean(tf.square(tf.abs(h_freq)), axis=(-1,-2))
            w = tx_power*h_sq/self.no
            # MSE on ToA
            toa = toa/self.config.nprach_cp_duration
            toa = tf.where(tx_ue, toa, 0.0)
            toa_est = tf.where(tx_ue, toa_est, 0.0)
            toa = tf.expand_dims(toa, axis=-1)
            toa_est = tf.expand_dims(toa_est, axis=-1)
            loss_toa = self.mse_toa(y_true=toa, y_pred=toa_est, sample_weight=w)
            # MSE on CFO
            f_off = self.cfo.ppm2Foffnorm(cfo_ppm)
            MAX_CFO_TRAIN = self.cfo.ppm2Foffnorm(MAX_CFO_PPM_TRAIN)
            f_off = f_off/MAX_CFO_TRAIN
            f_off = tf.where(tx_ue, f_off, 0.0)
            f_off_est = tf.where(tx_ue, f_off_est, 0.0)
            f_off = tf.expand_dims(f_off, axis=-1)
            f_off_est = tf.expand_dims(f_off_est, axis=-1)
            loss_cfo = self.mse_cfo(y_true=f_off, y_pred=f_off_est,
                                    sample_weight=w)
            return loss_tx_ue, loss_toa, loss_cfo
        else:
            if self.system == 'dl':
                toa_est = toa_est*self.config.nprach_cp_duration
                MAX_CFO_TRAIN = self.cfo.ppm2Foffnorm(MAX_CFO_PPM_TRAIN)
                f_off_est = f_off_est*MAX_CFO_TRAIN
                tx_ue_hat = tf.greater(tf.sign(tx_ue_hat), 0.0)
            # Detection error rate
            tx_ue_float = tf.cast(tx_ue, tf.float32)
            tx_ue_hat_float = tf.cast(tx_ue_hat, tf.float32)
            # For FNR and FPR, -1 means the example should be ignored
            fpr = tf.where(tf.logical_not(tx_ue), tx_ue_hat_float, -1.0) # FPR
            fnr =  tf.where(tx_ue, 1.0-tx_ue_hat_float, -1.0) # FNR
            # ToA NMSE
            toa_err = tf.where(tx_ue,
                tf.square((toa-toa_est)/self.config.nprach_cp_duration), 0.0)
            # CFO NMSE
            f_off = self.cfo.ppm2Foffnorm(cfo_ppm)
            f_off_err = tf.where(tx_ue,
                        tf.square((f_off-f_off_est)/self.config.bandwidth), 0.0)
            # Compute the snr
            freq = sn.channel.subcarrier_frequencies(
                        self.config.nprach_dft_size, self.config.delta_f_ra)
            h_freq = sn.channel.cir_to_ofdm_channel(freq, a, tau)[:,0,0,:,0,:,:]
            h_sq = tf.reduce_mean(tf.square(tf.abs(h_freq)), axis=(-1,-2))
            snr = tx_power*h_sq/self.no
            return snr, toa, f_off, ue_prob, fpr, fnr, toa_err, f_off_err
