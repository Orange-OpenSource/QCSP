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

import tensorflow as tf
import sionna as sn



class MatchedFilterOneTapEqualizer(tf.keras.layers.Layer):

    def __init__(self, resource_grid, **kwargs):
        super().__init__(**kwargs)
        self._resource_grid = resource_grid
        self._pilot_pattern = self._resource_grid.pilot_pattern
        self._removed_nulled_scs = sn.ofdm.RemoveNulledSubcarriers(self._resource_grid)

        # Precompute indices to extract data symbols
        mask = self._pilot_pattern.mask  # output shape = [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
        mask = mask[0,0,:,:] # output shape = [num_ofdm_symbols, num_effective_subcarriers]
        num_data_symbols = self._pilot_pattern.num_data_symbols
        data_ind = tf.argsort(sn.utils.flatten_last_dims(mask, num_dims=2), axis=-1, direction="ASCENDING")
        self._data_ind = data_ind[...,:num_data_symbols]

    def call(self, inputs):
        # input shape = [batch_size, num_ofdm_symb, fft_size] ; [batch_size, num_ofdm_symb, num_effective_subcarriers] ; tf.float32
        x, channel, n0 = inputs

        # out shape = [batch_size, num_ofdm_symbols, num_effective_subcarriers]
        x = self._removed_nulled_scs(x)

        # out shape = [batch_size, num_ofdm_symbols * num_effective_subcarriers = num_data_symb + num_pilots]
        x = sn.utils.flatten_last_dims(x, num_dims=2)
        channel = sn.utils.flatten_last_dims(channel, num_dims=2)

        # out shape = [batch_size, num_data_symb]
        x = tf.gather(x, indices=self._data_ind, axis=-1)
        channel = tf.gather(channel, indices=self._data_ind, axis=-1)

        # out shape = [batch_size, num_data_symb]
        # TODO: This scheme cannot work with high order constellation since the amplitude of the channel is not compensated.
        x = x * tf.math.conj(channel)
        n0_eff = n0 * tf.cast(tf.square(tf.abs(channel)), tf.float32)

        output = [x, n0_eff]

        return output