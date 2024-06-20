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

import numpy as np
import tensorflow as tf


class AverageInterpolator(tf.keras.layers.Layer):

    def __init__(self, resource_grid, mask=None, **kwargs):
        super().__init__(**kwargs)
        self._resource_grid = resource_grid
        self._pilot_pattern = self._resource_grid.pilot_pattern
        self._dc_null = self._resource_grid.dc_null

        # Get updated mask
        if mask is None:
            mask = self._pilot_pattern.mask  # output shape = [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]
            mask = mask[0,0,:,:] # output shape = [num_ofdm_symbols, num_effective_subcarriers]
        
        if self._dc_null: # Insert DC if not already present in mask to compute correct interpolation index
            dc_ind = self._resource_grid.dc_ind
            mask = np.array(mask)
            mask = np.insert(mask, dc_ind, 0, axis=-1)
            mask = tf.convert_to_tensor(mask, dtype=tf.int32) # output shape = [num_ofdm_symbols, num_effective_subcarriers + 1]

        self._mask_dc = mask
        
        # Get [OFDM symbol (time), Subcarriers (frequency)]
        self._nb_ofdmsymb = self._resource_grid.num_ofdm_symbols
        self._nb_subcarriers = self._resource_grid.fft_size
        self._num_effective_subcarriers_dc = self._resource_grid.num_effective_subcarriers # For the correct computation of interpolation indices, we need to consider the DC

        if self._dc_null:
            self._num_effective_subcarriers_dc += 1
            self._dc_ind = int(self._num_effective_subcarriers_dc/2 - (self._num_effective_subcarriers_dc % 2 == 1)/2)

        self._data_pilot_ind = tf.range(self._num_effective_subcarriers_dc, dtype=tf.int32)

        if self._dc_null:
            self._data_pilot_ind = np.delete(self._data_pilot_ind, self._dc_ind)

        # 1D-array of the index of OFDM symbol for each pilot, ordered first by OFDM symbol and then by subcarrier
        self._pilots_ofdmsymb_index = tf.cast(tf.where(self._mask_dc == 1)[:,0], dtype=tf.int32)
        
        # 1D-array of the index of the subcarrier for each pilot, ordered first by OFDM symbol and then by subcarrier
        self._pilots_subcarrier_index = tf.cast(tf.where(self._mask_dc == 1)[:,1], dtype=tf.int32)
        
        # Store the unique OFDM symbol index containing at least one symbol
        self._unique_pilots_ofdmsymb, _ = tf.unique(self._pilots_ofdmsymb_index)
        self._nb_unique_pilots_ofdmsymb = tf.size(self._unique_pilots_ofdmsymb)

        # Get reshape size
        self._reshape_lines, self._reshape_columns = self._generate_averaging_index_()

    
    def _generate_averaging_index_(self):
        # Convert tensors to numpy arrays
        nb_unique_pilots_ofdmsymb = np.array(self._nb_unique_pilots_ofdmsymb)
        pilots_ofdmsymb_index = np.array(self._pilots_ofdmsymb_index)

        # The averaging is done in frequency. All the estimated channels within an OFDM symbol are averaged and broadcasted to the whole OFDM symbol.
        # The process inplies a simple reshape and reduce_mean of the inputs estimated channels.
        # We check that all OFDM symbols containing pilots have the same number of pilots.
        _, unique_count = np.unique(pilots_ofdmsymb_index, return_counts=True)
        tmp_nb_pilots_ofdmsymb = np.unique(unique_count, return_counts=False)

        if tmp_nb_pilots_ofdmsymb.size > 1:
            raise ValueError(f'[ERROR][Average Interpolator] The number of pilots in OFDM symbols is not constant. Distribution of pilots in OFDM symbols: {unique_count}')

        # Same number of pilots in each OFDM symbol
        # Compute the reshape number of columns, equal to the number of pilots within an OFDM symbol.
        reshape_columns = tmp_nb_pilots_ofdmsymb[0]

        # Number of lines is number of unique OFDM symbols containing at least one pilots
        reshape_lines = nb_unique_pilots_ofdmsymb

        return tf.convert_to_tensor(reshape_lines, dtype=tf.int32), tf.convert_to_tensor(reshape_columns, dtype=tf.int32)

        
    def call(self, inputs):
        """
            Average the channel estimations over the whole OFDM symbol.
            TODO: Add interpolation in time, no time interpolation at the moment.
            
            inputs: 2D-array[complex64]
                Shape[0] is the batch size.
                Shape[1] is the number of pilot symbols within the OFDM frame (order frequency then time)
        """
        estimated_channels = inputs
        
        estimated_channels = tf.reshape(estimated_channels, shape=[tf.shape(estimated_channels)[0], self._reshape_lines, self._reshape_columns])
        averaged_channels = tf.reduce_mean(estimated_channels, axis=-1, keepdims=True)

        interpolated_channels = tf.tile(averaged_channels, multiples=[1, 1, self._num_effective_subcarriers_dc])

        if self._dc_null:
            interpolated_channels = tf.gather(interpolated_channels, indices=self._data_pilot_ind, axis=-1) # get data and pilots, no DC

        return interpolated_channels # output shape = [batch_size, num_ofdm_symbols, num_effective_subcarriers]
        