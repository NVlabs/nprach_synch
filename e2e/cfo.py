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
Keras layer for simulating the carrier frequency offset (CFO).
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import sionna as sn


class CFO(Layer):
    # pylint: disable=line-too-long
    r"""
    Keras layer for simulating the carrier frequency offset (CFO).

    Let :math:`f_{\text{off}}` denotes frequency offset normalized by the
    sampling frequency. This layer computes

    .. math::
        y_n = x_n e^{j 2 \pi f_{\text{off}} n}

    where :math:`x_n` is the channel input and :math:`y_n` the layer output.

    The frequency offset normalized by the sampling frequency
    :math:`f_{\text{off}}` is computed from the frequency offset in
    parts-per-million (ppm) according to

    .. math::
        f_{\text{off}} = \frac{ f_{\text{off,ppm}} 10^{-6} f_c }{ f_s }

    where :math:`f_{\text{off,ppm}}` is the frequency offset in ppm,
    :math:`f_c` the carrier frequency in Hz, and :math:`f_s` the sampling
    frequency in Hz.

    Parameters
    ----------
    carrier_frequency : float
        Carrier frequency [Hz].

    sampling_frequency : float
        Sampling frequency [Hz]

    Input
    ------
    x : [batch size, num time steps, num users], tf.complex
        Input signal.

    f_off_ppm : [batch size, num users], tf.float
        Frequency offset in parts-per-million (ppm) for every user.

    Output
    -------
    y : [batch size, time steps, num users], tf.complex
        Input signal with CFO.
    """

    def __init__(self, carrier_frequency, sampling_frequency):
        super().__init__()

        # Multiplicative factor to go from ppm to normalized frequency offset
        self._ppm2normfoff = 1e-6*carrier_frequency/sampling_frequency

    def call(self, inputs):

        x, f_off_ppm = inputs
        real_dtype = x.dtype.real_dtype
        num_time_steps = tf.shape(x)[1]

        # Frequency offset
        # [batch size, 1, num users]
        f_off_norm = f_off_ppm*self._ppm2normfoff
        f_off_norm = tf.expand_dims(f_off_norm, axis=1)

        # Phase shift due to CFO
        # [batch size, num time steps, num users]
        time_steps = tf.range(num_time_steps, dtype=real_dtype)
        time_steps = tf.expand_dims(tf.expand_dims(time_steps, axis=0), axis=2)
        phase_shift = 2.*sn.PI*time_steps*f_off_norm

        # [batch size, num time steps, num users]
        y = x*tf.exp(tf.complex(0.,phase_shift))
        return y

    def ppm2Foffnorm(self, cfo_ppm):
        """
        Convert the CFO in ppm to the corresponding frequency offset normalized
        by the sampling frequency.

        Input
        ------
        cfo_ppm: Any shape, float
            CFO in ppm.

        Output
        -------
        CFO normalized by the sampling frequency.
        """
        return cfo_ppm*self._ppm2normfoff
