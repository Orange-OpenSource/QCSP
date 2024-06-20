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


class LeastSquareEstimator(tf.keras.layers.Layer):

    def __init__(self, resource_grid, **kwargs):
        super().__init__(**kwargs)
        self._resource_grid = resource_grid
        self._pilot_pattern = self._resource_grid.pilot_pattern
        self._pilots = self._pilot_pattern.pilots[0,0,:] # output shape = [num_tx, num_txt_ant, num_pilots] => [num_pilots]
        self._removed_nulled_scs = sn.ofdm.RemoveNulledSubcarriers(self._resource_grid)

        # Precompute indices to extract pilots symbols
        mask = self._pilot_pattern.mask[0,0,:,:]  # output shape = [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers] => [num_ofdm_symbols, num_effective_subcarriers]
        num_pilot_symbols = self._pilot_pattern.num_pilot_symbols
        pilot_ind = tf.argsort(sn.utils.flatten_last_dims(mask, num_dims=2), axis=-1, direction="DESCENDING")
        self._pilot_ind = pilot_ind[...,:num_pilot_symbols]

    def call(self, inputs):
        # input shape = [batch_size, num_ofdm_symb, fft_size] ; tf.float32
        x, n0 = inputs

        # out shape = [batch_size, num_ofdm_symbols, num_effective_subcarriers]
        x = self._removed_nulled_scs(x)

        # out shape = [batch_size, num_ofdm_symbols * num_effective_subcarriers = num_data_symb + num_pilots]
        x = sn.utils.flatten_last_dims(x, num_dims=2)

        # out shape = [batch_size, num_pilot_symb]
        x = tf.gather(x, indices=self._pilot_ind, axis=-1)

        # out shape = [batch_size, num_pilot_symb]
        h_ls = tf.math.divide_no_nan(x, self._pilots)
        
        # out shape = [1, num_pilot_symb]
        n0_eff = tf.expand_dims(tf.math.divide_no_nan(n0, tf.abs(self._pilots)**2), axis=0)

        outputs = [h_ls, n0_eff]

        return outputs
