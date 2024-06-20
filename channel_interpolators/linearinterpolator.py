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


class LinearInterpolator(tf.keras.layers.Layer):

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

        self._freq_interpolation_index, self._freq_interpolation_norm_factor, self._time_interpolation_index, self._time_interpolation_norm_factor = self._generate_interpolation_index_()

        # Modification of _freq_interpolation_index for tensorflow indexing support
        freq_index_array = []
        pilot_ofdmsymb_index = tf.expand_dims(tf.repeat(tf.range(self._nb_unique_pilots_ofdmsymb, dtype=tf.int32), self._num_effective_subcarriers_dc, axis=0), axis=1)
        for ind in range(2):
            freq_index = tf.expand_dims(sn.utils.flatten_last_dims(self._freq_interpolation_index[ind,:,:], num_dims=2), axis=1)
            freq_index = tf.concat([pilot_ofdmsymb_index, freq_index], axis=1)
            freq_index = tf.reshape(freq_index, shape=[1, self._nb_unique_pilots_ofdmsymb, self._num_effective_subcarriers_dc, 2])
            freq_index_array.append(freq_index)
        self._freq_interpolation_index = tf.concat(freq_index_array, axis=0)


    def _generate_interpolation_index_(self):
        # Convert tensors to numpy arrays
        nb_unique_pilots_ofdmsymb = np.array(self._nb_unique_pilots_ofdmsymb)
        nb_subcarriers = np.array(self._num_effective_subcarriers_dc) # the interpolation is limited to effective subcarrier, including the DC
        nb_ofdmsymb = np.array(self._nb_ofdmsymb)
        pilots_ofdmsymb_index = np.array(self._pilots_ofdmsymb_index)
        pilots_subcarrier_index = np.array(self._pilots_subcarrier_index)

        # The interpolation is done first in frequency in the OFDM symbols containing at least one pilot and then in time.        
        ## Index and scaling used for frequency interpolation in OFDM symbol containing at least one pilot
        ## The index represents the index of the subcarrier of the pilot used in the interpolation in the corresponding OFDM symbol
        ## Shape[0]: 0 = current pilot position | 1 = next pilot position
        freq_interpolation_index = np.empty(shape=(2, nb_unique_pilots_ofdmsymb, nb_subcarriers), dtype=np.int32)
        freq_interpolation_norm_factor = np.empty(shape=(nb_unique_pilots_ofdmsymb, nb_subcarriers), dtype=np.float32)

        ## Index and scaling used for time interpolation between OFDM symbols after frequency interpolation
        ## The index represents the index of the OFDM symbol in the frame
        ## Shape[0]: 0 = current pilot position | 1 = next pilot position
        time_interpolation_index = np.empty(shape=(2, nb_ofdmsymb), dtype=np.int32)
        time_interpolation_norm_factor = np.empty(shape=(nb_ofdmsymb), dtype=np.float32)

        # Get the index of OFDM symbols containing at least one pilot and the number of pilots in each of these OFDM symbols
        unique_pilots_ofdmsymb, unique_count = np.unique(pilots_ofdmsymb_index, return_counts=True)
        
        # First compute the index and scaling for the time interpolation (last interpolation step)
        ## Extrapolation when the full interpolation formula does not apply => handle boundaries
        ### In case the first symbol is not the first of the frame
        if unique_pilots_ofdmsymb[0] != 0:
            time_interpolation_index[:, 0:unique_pilots_ofdmsymb[0]] = unique_pilots_ofdmsymb[0]

            time_interpolation_norm_factor[0:unique_pilots_ofdmsymb[0]] = 0.

        ### Handle the last portion of the positions, works also if last pilot is at the end
        time_interpolation_index[:, unique_pilots_ofdmsymb[-1]:] = unique_pilots_ofdmsymb[-1]
        time_interpolation_norm_factor[unique_pilots_ofdmsymb[-1]:] = 0.

        ## If there is more than one OFDM symbol containing a pilot, the full interpolation formula apply in time
        if unique_pilots_ofdmsymb.size > 1:
            for current_pilot_ofdmsymb_index in np.arange(unique_pilots_ofdmsymb.size - 1):
                time_interpolation_index[0, unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index] : unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index + 1]] = unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index]
                time_interpolation_index[1, unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index] : unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index + 1]] = unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index + 1]
                
                current_ofdmsymb_distance = unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index + 1] - unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index]
                current_time_norm_factor = np.arange(current_ofdmsymb_distance) / current_ofdmsymb_distance
                time_interpolation_norm_factor[unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index] : unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index + 1]] = current_time_norm_factor
            
        ## Then compute the index and scaling for the frequency interpolation (first interpolation step)
        current_first_pilot_index = 0
        for (current_ofdmsymb_index, current_nb_pilots_in_ofdmsymb) in zip(np.arange(nb_unique_pilots_ofdmsymb), unique_count):
            current_pilots_subcarrier = pilots_subcarrier_index[current_first_pilot_index : current_first_pilot_index + current_nb_pilots_in_ofdmsymb]
            current_first_pilot_index += current_nb_pilots_in_ofdmsymb

            ## Extrapolation in case the whole formula does not apply for the frequency interpolation => handle boundaries
            ### If the first pilot in the OFDM symbol is not on the first subcarrier
            if current_pilots_subcarrier[0] != 0:
                freq_interpolation_index[:, current_ofdmsymb_index, 0:current_pilots_subcarrier[0]] = current_pilots_subcarrier[0]
                freq_interpolation_norm_factor[current_ofdmsymb_index, 0:current_pilots_subcarrier[0]] = 0.

            ### Handle the last portion of the positions, works also if last pilot is at the end
            freq_interpolation_index[:, current_ofdmsymb_index, current_pilots_subcarrier[-1]:] = current_pilots_subcarrier[-1]
            freq_interpolation_norm_factor[current_ofdmsymb_index, current_pilots_subcarrier[-1]:] = 0.

            ## In case there is more than one pilot in the OFDM symbol, the full interpolation formula apply in frequency
            if current_pilots_subcarrier.size > 1:
                for current_pilot_subcarrier_index in np.arange(current_pilots_subcarrier.size - 1):
                    freq_interpolation_index[0, current_ofdmsymb_index, current_pilots_subcarrier[current_pilot_subcarrier_index] : current_pilots_subcarrier[current_pilot_subcarrier_index + 1]] = current_pilots_subcarrier[current_pilot_subcarrier_index]
                    freq_interpolation_index[1, current_ofdmsymb_index, current_pilots_subcarrier[current_pilot_subcarrier_index] : current_pilots_subcarrier[current_pilot_subcarrier_index + 1]] = current_pilots_subcarrier[current_pilot_subcarrier_index + 1]
                    
                    current_subcarriers_distance = current_pilots_subcarrier[current_pilot_subcarrier_index + 1] - current_pilots_subcarrier[current_pilot_subcarrier_index]
                    current_freq_norm_factor = np.arange(current_subcarriers_distance) / current_subcarriers_distance
                    freq_interpolation_norm_factor[current_ofdmsymb_index, current_pilots_subcarrier[current_pilot_subcarrier_index] : current_pilots_subcarrier[current_pilot_subcarrier_index + 1]] = current_freq_norm_factor

        return tf.convert_to_tensor(freq_interpolation_index, dtype=tf.int32), tf.convert_to_tensor(freq_interpolation_norm_factor, dtype=tf.complex64), \
                tf.convert_to_tensor(time_interpolation_index, dtype=tf.int32), tf.convert_to_tensor(time_interpolation_norm_factor, dtype=tf.complex64)


    def call(self, inputs):
        """
            The linear interpolated channel at l (Lm <= l < L(m+1)): ( H(m+1) - H(m) ) * l/L + H(m)
            Where L is assumed to be the distance in terms of index between pilot(m) and pilot(m+1) 
            
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

        # Set interpolations index
        freq_index = self._freq_interpolation_index
        freq_norm = tf.expand_dims(self._freq_interpolation_norm_factor, axis=-1)

        # Frequency interpolation
        ## Get channels
        f_estimated_channels = tf.gather(estimated_channels, indices=self._unique_pilots_ofdmsymb, axis=0)

        f_estimated_channels_0 = tf.gather_nd(f_estimated_channels, indices=freq_index[0])
        f_estimated_channels_1 = tf.gather_nd(f_estimated_channels, indices=freq_index[1])        

        ## Process interpolation
        updates = (f_estimated_channels_1 - f_estimated_channels_0) * freq_norm + f_estimated_channels_0
        indices = tf.expand_dims(self._unique_pilots_ofdmsymb, axis=1)
        shape = tf.shape(estimated_channels)
        # Needed to avoid different shapes caused by the following if condition, place zeros OFDM symbols at the missing index in time (if any) to be able to process the time interpolation
        interpolated_channels = tf.scatter_nd(indices, updates, shape)
        
        # Time interpolation (if needed)
        if self._nb_unique_pilots_ofdmsymb != self._nb_ofdmsymb:
            time_index = self._time_interpolation_index
            time_norm = sn.utils.expand_to_rank(self._time_interpolation_norm_factor, 3, axis=1)

            t_interpolated_channels_0 = tf.gather(interpolated_channels, indices=time_index[0], axis=0)
            t_interpolated_channels_1 = tf.gather(interpolated_channels, indices=time_index[1], axis=0)

            interpolated_channels = (t_interpolated_channels_1 - t_interpolated_channels_0) * time_norm + t_interpolated_channels_0

        interpolated_channels = tf.transpose(interpolated_channels, perm=[2,0,1]) # reset batch size at the beginning

        if self._dc_null:
            interpolated_channels = tf.gather(interpolated_channels, indices=self._data_pilot_ind, axis=-1) # get data and pilots, no DC

        return interpolated_channels # output shape = [batch_size, num_ofdm_symbols, num_effective_subcarriers]
