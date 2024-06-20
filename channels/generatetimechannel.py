"""
Software Name : QCSP Orange
SPDX-FileCopyrightText: Copyright (c) Orange SA
SPDX-License-Identifier: MIT

This software is distributed under the MIT License,
see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

Authors: 
    Louis-Adrien DUFRÈNE    louisadrien.dufrene@orange.com
    Guillaume LARUE         guillaume.larue@orange.com
    Quentin LAMPIN          quentin.lampin@orange.com

Software description: Orange study on the combination of CCSK and OFDM modulation. Part of the QCSP ANR project. See Deliverable D2.5b_OFDM-CCSK.pdf
"""

"""
Derived from Sionna class GenerateTimeChannel() from version 0.10 to improve input naming, remove MIMO specificities and differenciate channel and data sampling frequencies for less complexity.

SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""


import tensorflow as tf
import sionna as sn


class CustomGenerateTimeChannel:
    def __init__(self, channel_model, channel_sampling_frequency, data_sampling_frequency, num_time_samples, l_min, l_max, cir_pwr=1.):

        # Callable used to sample channel input responses
        self._cir_sampler = channel_model

        self._l_min = l_min
        self._l_max = l_max
        self._l_tot = l_max - l_min + 1
        self._channel_sampling_frequency = channel_sampling_frequency
        self._data_sampling_frequency = data_sampling_frequency
        self._num_time_steps = num_time_samples
        self._cir_pwr = tf.cast(cir_pwr, dtype=tf.complex64)

    def __call__(self, batch_size=None):

        # Sample channel impulse responses
        h, tau = self._cir_sampler(batch_size, self._num_time_steps, self._channel_sampling_frequency) # inputs = batch_size, num_time_steps, sampling_frequency

        h = h[:,0,0,0,0,:,:]
        tau = tau[:,0,0,:]

        real_dtype = tau.dtype

        # Time lags for which to compute the channel taps
        l = tf.range(self._l_min, self._l_max+1, dtype=real_dtype)

        # Bring tau and l to broadcastable shapes
        tau = tf.expand_dims(tau, axis=-1)
        l = sn.utils.expand_to_rank(l, tau.shape.rank, axis=0)

        # sinc pulse shaping
        g = tf.experimental.numpy.sinc(l - (tau*self._data_sampling_frequency))
        g = tf.complex(g, tf.constant(0., real_dtype))
        h = tf.expand_dims(h, axis=-1)

        # For every tap, sum the sinc-weighted coefficients
        # Broadcast is not supported by TF for such high rank tensors.
        # We therefore do part of it manually
        g = tf.expand_dims(g, axis=2)
        hm = tf.reduce_sum(h*g, axis=-3)

        hm = hm / self._cir_pwr

        return hm # output shape = [batch_size, num_time_steps, l_tot = l_max - l_min + 1]
