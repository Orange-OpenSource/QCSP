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
import sionna as sn


class PieceWiseConstantInterpolator(tf.keras.layers.Layer):

    def __init__(self, resource_grid, interpolation_type='1D', mask=None, **kwargs):
        super().__init__(**kwargs)
        self._resource_grid = resource_grid
        self._pilot_pattern = self._resource_grid.pilot_pattern
        self._dc_null = self._resource_grid.dc_null
        self._interpolation_type = interpolation_type

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
        
        self._closest1D_freq_index, self._closest1D_time_index = self._generate_1Dclosest_pilots_()
        closest2D_freq_index, closest2D_time_index = self._generate_2Dclosest_pilots_()

        # Modification of _closest1D_freq_index for tensorflow indexing support
        pilot_ofdmsymb_index = tf.expand_dims(tf.repeat(tf.range(self._nb_unique_pilots_ofdmsymb, dtype=tf.int32), self._num_effective_subcarriers_dc, axis=0), axis=1)
        freq_index = tf.expand_dims(sn.utils.flatten_last_dims(self._closest1D_freq_index, num_dims=2), axis=1)
        freq_index = tf.concat([pilot_ofdmsymb_index, freq_index], axis=1)
        self._closest1D_freq_index = tf.reshape(freq_index, shape=[self._nb_unique_pilots_ofdmsymb, self._num_effective_subcarriers_dc, 2])

        # Modification of _closest2D_freq_index and _closest2D_time_index for tensorflow indexing support
        time_index = tf.expand_dims(sn.utils.flatten_last_dims(closest2D_time_index, num_dims=2), axis=1)
        freq_index = tf.expand_dims(sn.utils.flatten_last_dims(closest2D_freq_index, num_dims=2), axis=1)
        closest2D_index = tf.concat([time_index, freq_index], axis=1)
        self._closest2D_index = tf.reshape(closest2D_index, shape=[self._nb_ofdmsymb, self._num_effective_subcarriers_dc, 2])
        
    def _generate_1Dclosest_pilots_(self):
        # Convert tensors to numpy arrays
        nb_unique_pilots_ofdmsymb = np.array(self._nb_unique_pilots_ofdmsymb)
        nb_subcarriers = np.array(self._num_effective_subcarriers_dc) # the interpolation is limited to effective subcarrier, including the DC
        nb_ofdmsymb = np.array(self._nb_ofdmsymb)
        pilots_ofdmsymb_index = np.array(self._pilots_ofdmsymb_index)
        pilots_subcarrier_index = np.array(self._pilots_subcarrier_index)

        # The interpolation is done first in frequency in the OFDM symbols containing at least one pilot and then in time.
        ## Index used for frequency and time interpolation.
        ## First we compute the index of the closest pilots (subcarrier) in the OFDM symbol containing at least one symbol.
        ## Then we compute the index of the closest OFDM symbol containing at least one symbol for the rest of the OFDM symbols.
        closest1D_freq_index = np.empty(shape=(nb_unique_pilots_ofdmsymb, nb_subcarriers), dtype=np.int32)
        closest1D_time_index = np.empty(shape=nb_ofdmsymb, dtype=np.int32)

        # Compute the distance in terms of subcarrier index for each pilot subcarrier position from all possible subcarrier position
        subcarriers_dist = np.expand_dims(np.arange(nb_subcarriers, dtype=np.int32), axis=1)
        subcarriers_dist = np.abs(subcarriers_dist - pilots_subcarrier_index)

        # Get all unique OFDM symbol containing at least one pilot
        unique_pilots_ofdmsymb, unique_count = np.unique(pilots_ofdmsymb_index, return_counts=True)
        
        # Compute the closest pilot index (subcarrier) in each OFDM symbol with at least one pilot
        current_first_pilot_index = 0
        for (current_ofdmsymb_index, current_nb_pilots_ofdmsymb) in zip(np.arange(nb_unique_pilots_ofdmsymb), unique_count):
            current_dist_array = subcarriers_dist[:, current_first_pilot_index : current_first_pilot_index + current_nb_pilots_ofdmsymb]
            current_closest_pilots = np.argmin(current_dist_array, axis=1)
            current_pilots_subcarrier = pilots_subcarrier_index[current_first_pilot_index : current_first_pilot_index + current_nb_pilots_ofdmsymb]
            current_first_pilot_index += current_nb_pilots_ofdmsymb            
            closest1D_freq_index[current_ofdmsymb_index, :] = current_pilots_subcarrier[current_closest_pilots]

        # Compute the distance in terms of OFDM symbol between the OFDM symbol with pilots and all the OFDM symbol
        ofdmsymb_dist = np.expand_dims(np.arange(nb_ofdmsymb, dtype=np.int32), axis=1)
        ofdmsymb_dist = np.abs(ofdmsymb_dist - unique_pilots_ofdmsymb)
        
        # Compute the closest OFDM symbol index containing a pilot for all OFDM symbol in the frame
        closest_pilots_ofdmsymb = np.argmin(ofdmsymb_dist, axis=1)
        closest1D_time_index = unique_pilots_ofdmsymb[closest_pilots_ofdmsymb]

        return tf.convert_to_tensor(closest1D_freq_index, tf.int32), tf.convert_to_tensor(closest1D_time_index, tf.int32)

    def _generate_2Dclosest_pilots_(self):
        # Convert tensors to numpy arrays
        nb_subcarriers = np.array(self._num_effective_subcarriers_dc) # the interpolation is limited to effective subcarrier, including the DC
        nb_ofdmsymb = np.array(self._nb_ofdmsymb)
        frame_shape = np.array([self._nb_ofdmsymb, self._num_effective_subcarriers_dc])
        pilots_ofdmsymb_index = np.array(self._pilots_ofdmsymb_index)
        pilots_subcarrier_index = np.array(self._pilots_subcarrier_index)

        # The interpolation is done in both dimensions frequency and time.
        ## Index used for frequency and time interpolation.
        ## We compute the Euclidean distance between of (OFDM symbol, subcarrier) and all pilots positions.
        ## We store the closest one as two arrays: one for frequency and one for time position.
        closest2D_freq_index = np.empty(shape=frame_shape, dtype=np.int32)
        closest2D_time_index = np.empty(shape=frame_shape, dtype=np.int32)

        # Compute distance in time
        ofdmsymb_dist = np.expand_dims(np.arange(nb_ofdmsymb, dtype=np.int32), axis=1)
        ofdmsymb_dist = np.abs(ofdmsymb_dist - pilots_ofdmsymb_index)**2

        # Compute distance in frequency
        subcarriers_dist = np.expand_dims(np.arange(nb_subcarriers, dtype=np.int32), axis=1)
        subcarriers_dist = np.abs(subcarriers_dist - pilots_subcarrier_index)**2
        
        # Compute closest pilot and store subcarrier and ofdm symbol position
        for current_ofdmsymbol in np.arange(nb_ofdmsymb):
            current_closest_pilot_index = np.argmin(ofdmsymb_dist[current_ofdmsymbol,:] + subcarriers_dist, axis=1)
            closest2D_freq_index[current_ofdmsymbol, :] = pilots_subcarrier_index[current_closest_pilot_index]
            closest2D_time_index[current_ofdmsymbol, :] = pilots_ofdmsymb_index[current_closest_pilot_index]

        return tf.convert_to_tensor(closest2D_freq_index, tf.int32), tf.convert_to_tensor(closest2D_time_index, tf.int32)
    
    def call(self, inputs):
        """
            The Piece-wise Constant Interpolator estimated the unknown channel to the closest pilot channel available.
            - 1D: First in frequency in OFDM symbol containing pilots and then in time.
            - 2D: In both time and frequency, the closest pilot is evaluated by euclidian distance.
            
            inputs: 2D-array[complex64]
                Shape[0] is the batch size.
                Shape[1] is the number of pilot symbols within the OFDM frame (order frequency then time)
        """
        estimated_channels = inputs

        # Recreate a matrix of size [num_ofdm_symbols, num_effective_subcarriers_dc] fill with 0. and h_hat at pilots positions
        updates = tf.transpose(estimated_channels)
        indices = tf.cast(tf.where(self._mask_dc == 1), tf.int32)
        shape = tf.stack([self._nb_ofdmsymb, self._num_effective_subcarriers_dc, tf.shape(inputs)[0]], axis=0)
        estimated_channels = tf.scatter_nd(indices=indices, updates=updates, shape=shape)

        if self._interpolation_type == '1D':
            freq_index = self._closest1D_freq_index

            # Frequency interpolation
            ## Get channels
            f_estimated_channels = tf.gather(estimated_channels, indices=self._unique_pilots_ofdmsymb, axis=0)

            ## Process interpolation
            updates = tf.gather_nd(f_estimated_channels, indices=freq_index)
            indices = tf.expand_dims(self._unique_pilots_ofdmsymb, axis=1)
            shape = tf.shape(estimated_channels)
            # Needed to avoid different shapes caused by the following if condition, place zeros OFDM symbols at the missing index in time (if any) to be able to process the time interpolation
            interpolated_channels = tf.scatter_nd(indices, updates, shape)

            # Time interpolation (if needed)
            if self._nb_unique_pilots_ofdmsymb != self._nb_ofdmsymb:
                time_index = self._closest1D_time_index
                interpolated_channels = tf.gather(interpolated_channels, indices=time_index, axis=0)
            
        elif self._interpolation_type == '2D':
            # Time and Frequency interpolation
            interpolated_channels = tf.gather_nd(estimated_channels, indices=self._closest2D_index)

        interpolated_channels = tf.transpose(interpolated_channels, perm=[2,0,1]) # reset batch size at the beginning

        if self._dc_null:
            interpolated_channels = tf.gather(interpolated_channels, indices=self._data_pilot_ind, axis=-1) # get data and pilots, no DC

        return interpolated_channels # output shape = [batch_size, num_data_symbols, num_effective_subcarriers]
