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
Derived from Sionna Class ApplyTimeChannel() from version 0.10 to remove MIMO specificities, reduce complexity and apply block fading channel.

SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import tensorflow as tf
import numpy as np
import scipy
import sionna as sn


class CustomApplyTimeChannel(tf.keras.layers.Layer):
    def __init__(self, num_time_samples, l_tot, add_awgn=True, dtype=tf.complex64, **kwargs):

        super().__init__(trainable=False, dtype=dtype, **kwargs)

        self._add_awgn = add_awgn
        self._num_time_samples = num_time_samples

        # The channel transfert function is implemented by first gathering from
        # the vector of transmitted baseband symbols
        # x = [x_0,...,x_{num_time_samples-1}]^T  the symbols that are then
        # multiplied by the channel tap coefficients.
        # We build here the matrix of indices G, with size
        # `num_time_samples + l_tot - 1` x `l_tot` that is used to perform this
        # gathering.
        # For example, if there are 4 channel taps
        # h = [h_0, h_1, h_2, h_3]^T
        # and `num_time_samples` = 10 time steps then G would be
        #       [[0, 10, 10, 10]
        #        [1,  0, 10, 10]
        #        [2,  1,  0, 10]
        #        [3,  2,  1,  0]
        #        [4,  3,  2,  1]
        #        [5,  4,  3,  2]
        #        [6,  5,  4,  3]
        #        [7,  6,  5,  4]
        #        [8,  7,  6,  5]
        #        [9,  8,  7,  6]
        #        [10, 9,  8,  7]
        #        [10,10,  9,  8]
        #        [10,10, 10,  9]
        # Note that G is a Toeplitz matrix.
        # In this example, the index `num_time_samples`=10 corresponds to the
        # zero symbol. The vector of transmitted symbols is padded with one
        # zero at the end.
        
        first_colum = np.concatenate([np.arange(0, num_time_samples), np.full([l_tot-1], num_time_samples)])
        first_row = np.concatenate([[0], np.full([l_tot-1], num_time_samples)])
        self._g = scipy.linalg.toeplitz(first_colum, first_row)

    def build(self, input_shape): #pylint: disable=unused-argument

        if self._add_awgn:
            self._awgn = sn.channel.AWGN(dtype=self.dtype)

    def call(self, inputs):
        """
        tf.shape(x) = [batch_size, num_channel_blocks (e.g. num_ofdm_symbols), num_time_samples (e.g. fft_size + cp_length)]
        tf.shape(h_time) = [batch_size, num_channel_blocks (e.g. num_ofdm_symbols), l_tot]
        """

        if self._add_awgn:
            x, h_time, no = inputs
        else:
            x, h_time = inputs

        # Preparing the channel input for broadcasting and matrix multiplication
        x = tf.pad(x, [[0,0], [0,0], [0,1]])
        x = tf.gather(x, self._g, axis=-1)

        # Apply the channel response
        y = x @ tf.expand_dims(h_time, axis=-1)
        y = y[:,:,:,0]
        y = tf.signal.overlap_and_add(signal=y, frame_step=self._num_time_samples)

        # Add AWGN if requested
        if self._add_awgn:
            y = self._awgn((y, no))

        return y # output shape = [batch_size, (num_channel_blocks * num_time_samples) + l_tot -1]
