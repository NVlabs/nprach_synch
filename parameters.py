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
Training and evaluation parameters
"""

import tensorflow as tf
import numpy as np


#########################################################
# General
#########################################################
# Number of pseudo-random repetitions forming the NPRACH
NPRACH_NUM_REP = 1
# Max speed
MAX_SPEED = 0.0 # m/s
# Noise power spectral density
N0_DB = tf.constant(0.0, tf.float32)
# TX power
TX_POWER_DB= tf.constant(110.0, tf.float32)
# Channel time-lag
MIN_L = 0 # Min
MAX_L = 12 # Max
# Carrier frequency (Hz)
CARRIER_FREQ = 3.4e9
# Sampling frequency (Hz)
SAMPLING_FREQUENCY = 50e6
# Number of subcarrier for the NPRACH.
# Corresponds to the maximum number of UTs that can access the channel
# concurrently without colliding
NPRACH_NUM_SC = 48

##########################################################
# Training
##########################################################
# Max CFO in parts-per-million (ppm) when training
MAX_CFO_PPM_TRAIN = tf.constant(25, tf.float32)
# Number of training iterations
NUM_IT_TRAIN = 800000
# Batch size for training
BATCH_SIZE_TRAIN = tf.cast(64, tf.int32)
# Location for storing the weights of the deep learning-based synchronization
# method.
DEEPNSYNCH_WEIGHTS = 'weights.dat'

##########################################################
# Evaluation
##########################################################
# Default probability of transmission
DEFAULT_UE_PROB_EVAL = tf.constant(0.5, tf.float32)
# Number of iterations
NUM_IT_EVAL = tf.constant(100, tf.int32)
# Batch size
BATCH_SIZE_EVAL = tf.cast(64, tf.int32)
# CFO values for plotting vs SNR [ppm]
EVAL_SNR_CFO_PPMs = tf.constant([0.0, 10.0, 20.0], tf.float32)
# CFO range [ppm]
CFO_PPM_EVAL_RANGE = tf.constant(np.linspace(0.0, 20.0, 10), tf.float32)
# Range of probability of transmissions
PTX_EVAL_RANGE = tf.constant(np.linspace(0.05, 0.95, 10), tf.float32)
