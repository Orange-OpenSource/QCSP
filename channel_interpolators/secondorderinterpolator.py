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


class SecondOrderInterpolator(tf.keras.layers.Layer):

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
        for ind in range(3):
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
        ## Shape[0]: 0 = previous pilot position | 1 = current pilot position | 2 = next pilot position
        freq_interpolation_index = np.zeros(shape=(3, nb_unique_pilots_ofdmsymb, nb_subcarriers), dtype=np.int32)
        freq_interpolation_norm_factor = np.zeros(shape=(3, nb_unique_pilots_ofdmsymb, nb_subcarriers), dtype=np.float32)

        ## Index and scaling used for time interpolation between OFDM symbols after frequency interpolation
        ## The index represents the index of the OFDM symbol in the frame
        ## Shape[0]: 0 = previous pilot position | 1 = current pilot position | 2 = next pilot position
        time_interpolation_index = np.zeros(shape=(3, nb_ofdmsymb), dtype=np.int32)
        time_interpolation_norm_factor = np.zeros(shape=(3, nb_ofdmsymb), dtype=np.float32)

        # First compute the index and scaling for the time interpolation (last interpolation step)
        ## Get the OFDM symbols containing at least one pilot
        unique_pilots_ofdmsymb, unique_count = np.unique(pilots_ofdmsymb_index, return_counts=True)
        
        ## Only one OFDM symbol with pilots
        if unique_pilots_ofdmsymb.size == 1:
            ### Extrapolation case where the full interpolation formula doesn't apply
            time_interpolation_index[:, :] = unique_pilots_ofdmsymb[0]

            time_interpolation_norm_factor[0, :] = 0.
            time_interpolation_norm_factor[1, :] = 1.
            time_interpolation_norm_factor[2, :] = 0.

        ## >= 2 OFDM symbols with pilots
        else:
            ### Extrapolation case where the full interpolation formula doesn't apply => handle boundaries
            #### Before first pilot
            if unique_pilots_ofdmsymb[0] != 0:
                time_interpolation_index[:, 0:unique_pilots_ofdmsymb[0]] = unique_pilots_ofdmsymb[0]

                time_interpolation_norm_factor[0, 0:unique_pilots_ofdmsymb[0]] = 0.
                time_interpolation_norm_factor[1, 0:unique_pilots_ofdmsymb[0]] = 1.
                time_interpolation_norm_factor[2, 0:unique_pilots_ofdmsymb[0]] = 0.

            #### Between first and second pilots
            time_interpolation_index[0, unique_pilots_ofdmsymb[0]:unique_pilots_ofdmsymb[1]] = unique_pilots_ofdmsymb[0]
            time_interpolation_index[1, unique_pilots_ofdmsymb[0]:unique_pilots_ofdmsymb[1]] = unique_pilots_ofdmsymb[0]
            time_interpolation_index[2, unique_pilots_ofdmsymb[0]:unique_pilots_ofdmsymb[1]] = unique_pilots_ofdmsymb[1]

            current_ofdmsymb_distance = unique_pilots_ofdmsymb[1] - unique_pilots_ofdmsymb[0]
            alpha = np.arange(current_ofdmsymb_distance) / current_ofdmsymb_distance
            time_interpolation_norm_factor[0, unique_pilots_ofdmsymb[0]:unique_pilots_ofdmsymb[1]] = alpha * (alpha - 1) * 0.5
            time_interpolation_norm_factor[1, unique_pilots_ofdmsymb[0]:unique_pilots_ofdmsymb[1]] = (-1) * (alpha - 1) * (alpha + 1)
            time_interpolation_norm_factor[2, unique_pilots_ofdmsymb[0]:unique_pilots_ofdmsymb[1]] = alpha * (alpha + 1) * 0.5

            #### After last pilot => handle boundaries
            if unique_pilots_ofdmsymb[-1] != (nb_ofdmsymb-1):
                time_interpolation_index[0, unique_pilots_ofdmsymb[-1]:] = unique_pilots_ofdmsymb[-2]
                time_interpolation_index[1, unique_pilots_ofdmsymb[-1]:] = unique_pilots_ofdmsymb[-1]
                time_interpolation_index[2, unique_pilots_ofdmsymb[-1]:] = unique_pilots_ofdmsymb[-1]
                
                #### For the end we consider the distance between last and second from last pilots position
                current_ofdmsymb_distance = unique_pilots_ofdmsymb[-1] - unique_pilots_ofdmsymb[-2]
                remaining_distance = nb_ofdmsymb - unique_pilots_ofdmsymb[-1]
                alpha = np.arange(remaining_distance) / current_ofdmsymb_distance
                time_interpolation_norm_factor[0, unique_pilots_ofdmsymb[-1]:] = alpha * (alpha - 1) * 0.5
                time_interpolation_norm_factor[1, unique_pilots_ofdmsymb[-1]:] = (-1) * (alpha - 1) * (alpha + 1)
                time_interpolation_norm_factor[2, unique_pilots_ofdmsymb[-1]:] = alpha * (alpha + 1) * 0.5
            ### Handle the last position, otherwise it will not be processed => handle boundaries
            else:
                time_interpolation_index[:, unique_pilots_ofdmsymb[-1]] = unique_pilots_ofdmsymb[-1]

                time_interpolation_norm_factor[0, unique_pilots_ofdmsymb[-1]] = 0.
                time_interpolation_norm_factor[1, unique_pilots_ofdmsymb[-1]] = 1.
                time_interpolation_norm_factor[2, unique_pilots_ofdmsymb[-1]] = 0.
        
            ### Regular case where the full interpolation formula apply, need at least 3 OFDM symbols containing pilots
            if unique_pilots_ofdmsymb.size > 2:
                for current_pilot_ofdmsymb_index in np.arange(1, unique_pilots_ofdmsymb.size - 1):
                    time_interpolation_index[0, unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index] : unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index + 1]] = unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index - 1]
                    time_interpolation_index[1, unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index] : unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index + 1]] = unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index]
                    time_interpolation_index[2, unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index] : unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index + 1]] = unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index + 1]
                    
                    ### For the scaling factor, the distance between current pilot and next pilot is supposed equal to previous pilot
                    current_ofdmsymb_distance = unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index + 1] - unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index]
                    alpha = np.arange(current_ofdmsymb_distance) / current_ofdmsymb_distance
                    time_interpolation_norm_factor[0, unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index] : unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index + 1]] = alpha * (alpha - 1) * 0.5
                    time_interpolation_norm_factor[1, unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index] : unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index + 1]] = (-1) * (alpha - 1) * (alpha + 1)
                    time_interpolation_norm_factor[2, unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index] : unique_pilots_ofdmsymb[current_pilot_ofdmsymb_index + 1]] = alpha * (alpha + 1) * 0.5


        # Then compute the index and scaling for the frequency interpolation in valid OFDM symbols containing at least one pilot (first interpolation step)
        current_first_pilot_index = 0
        ## Each OFDM symbol containing at least one pilot is treated independently in the loop
        for (current_ofdmsymb_index, current_nb_pilots_in_ofdmsymb) in zip(np.arange(nb_unique_pilots_ofdmsymb), unique_count):
            current_pilots_subcarrier = pilots_subcarrier_index[current_first_pilot_index : current_first_pilot_index + current_nb_pilots_in_ofdmsymb]
            current_first_pilot_index += current_nb_pilots_in_ofdmsymb

            ## Only one pilot in the current OFDM symbol
            if current_pilots_subcarrier.size == 1:
                ### Extrapolation case where the full interpolation formula doesn't apply
                freq_interpolation_index[:, current_ofdmsymb_index, 0] = current_pilots_subcarrier[0]

                freq_interpolation_norm_factor[0, current_ofdmsymb_index, :] = 0.
                freq_interpolation_norm_factor[1, current_ofdmsymb_index, :] = 1.
                freq_interpolation_norm_factor[2, current_ofdmsymb_index, :] = 0.

            ## >= 2 pilots in the current OFDM symbol
            else:
                ### Extrapolation case where the full interpolation formula doesn't apply => handle boundaries
                #### Before first pilot
                if current_pilots_subcarrier[0] != 0:
                    freq_interpolation_index[:, current_ofdmsymb_index, 0:current_pilots_subcarrier[0]] = current_pilots_subcarrier[0]

                    freq_interpolation_norm_factor[0, current_ofdmsymb_index, 0:current_pilots_subcarrier[0]] = 0.
                    freq_interpolation_norm_factor[1, current_ofdmsymb_index, 0:current_pilots_subcarrier[0]] = 1.
                    freq_interpolation_norm_factor[2, current_ofdmsymb_index, 0:current_pilots_subcarrier[0]] = 0.

                #### Between first and second pilots
                freq_interpolation_index[0, current_ofdmsymb_index, current_pilots_subcarrier[0]:current_pilots_subcarrier[1]] = current_pilots_subcarrier[0]
                freq_interpolation_index[1, current_ofdmsymb_index, current_pilots_subcarrier[0]:current_pilots_subcarrier[1]] = current_pilots_subcarrier[0]
                freq_interpolation_index[2, current_ofdmsymb_index, current_pilots_subcarrier[0]:current_pilots_subcarrier[1]] = current_pilots_subcarrier[1]

                current_subcarriers_distance = current_pilots_subcarrier[1] - current_pilots_subcarrier[0]
                alpha = np.arange(current_subcarriers_distance) / current_subcarriers_distance
                freq_interpolation_norm_factor[0, current_ofdmsymb_index, current_pilots_subcarrier[0]:current_pilots_subcarrier[1]] = alpha * (alpha - 1) * 0.5
                freq_interpolation_norm_factor[1, current_ofdmsymb_index, current_pilots_subcarrier[0]:current_pilots_subcarrier[1]] = (-1) * (alpha - 1) * (alpha + 1)
                freq_interpolation_norm_factor[2, current_ofdmsymb_index, current_pilots_subcarrier[0]:current_pilots_subcarrier[1]] = alpha * (alpha + 1) * 0.5

                #### After last pilot => handle boundaries
                if current_pilots_subcarrier[-1] != (nb_subcarriers - 1):
                    freq_interpolation_index[0, current_ofdmsymb_index, current_pilots_subcarrier[-1]:] = current_pilots_subcarrier[-2]
                    freq_interpolation_index[1, current_ofdmsymb_index, current_pilots_subcarrier[-1]:] = current_pilots_subcarrier[-1]
                    freq_interpolation_index[2, current_ofdmsymb_index, current_pilots_subcarrier[-1]:] = current_pilots_subcarrier[-1]
                    
                    #### For the end we consider the distance between last and second from last pilots position
                    current_subcarriers_distance = current_pilots_subcarrier[-1] - current_pilots_subcarrier[-2]
                    remaining_distance = nb_subcarriers - current_pilots_subcarrier[-1]
                    alpha = np.arange(remaining_distance) / current_subcarriers_distance
                    freq_interpolation_norm_factor[0, current_ofdmsymb_index, current_pilots_subcarrier[-1]:] = alpha * (alpha - 1) * 0.5
                    freq_interpolation_norm_factor[1, current_ofdmsymb_index, current_pilots_subcarrier[-1]:] = (-1) * (alpha - 1) * (alpha + 1)
                    freq_interpolation_norm_factor[2, current_ofdmsymb_index, current_pilots_subcarrier[-1]:] = alpha * (alpha + 1) * 0.5
                ### Handle the last position, otherwise it will not be processed => handle boundaries
                else:
                    freq_interpolation_index[:, current_ofdmsymb_index, current_pilots_subcarrier[-1]] = current_pilots_subcarrier[-1]

                    freq_interpolation_norm_factor[0, current_ofdmsymb_index, current_pilots_subcarrier[-1]] = 0.
                    freq_interpolation_norm_factor[1, current_ofdmsymb_index, current_pilots_subcarrier[-1]] = 1.
                    freq_interpolation_norm_factor[2, current_ofdmsymb_index, current_pilots_subcarrier[-1]] = 0.
            
                ### Regular case where the full interpolation formula apply, need at least 3 pilots in the current OFDM symbol
                if current_pilots_subcarrier.size > 2:
                    for current_pilot_subcarrier_index in np.arange(1, current_pilots_subcarrier.size - 1):
                        freq_interpolation_index[0, current_ofdmsymb_index, current_pilots_subcarrier[current_pilot_subcarrier_index] : current_pilots_subcarrier[current_pilot_subcarrier_index + 1]] = current_pilots_subcarrier[current_pilot_subcarrier_index - 1]
                        freq_interpolation_index[1, current_ofdmsymb_index, current_pilots_subcarrier[current_pilot_subcarrier_index] : current_pilots_subcarrier[current_pilot_subcarrier_index + 1]] = current_pilots_subcarrier[current_pilot_subcarrier_index]
                        freq_interpolation_index[2, current_ofdmsymb_index, current_pilots_subcarrier[current_pilot_subcarrier_index] : current_pilots_subcarrier[current_pilot_subcarrier_index + 1]] = current_pilots_subcarrier[current_pilot_subcarrier_index + 1]
                        
                        ### For the scaling factor, the distance between current pilot and next pilot is supposed equal to previous pilot
                        current_subcarriers_distance = current_pilots_subcarrier[current_pilot_subcarrier_index + 1] - current_pilots_subcarrier[current_pilot_subcarrier_index]
                        alpha = np.arange(current_subcarriers_distance) / current_subcarriers_distance
                        freq_interpolation_norm_factor[0, current_ofdmsymb_index, current_pilots_subcarrier[current_pilot_subcarrier_index] : current_pilots_subcarrier[current_pilot_subcarrier_index + 1]] = alpha * (alpha - 1) * 0.5
                        freq_interpolation_norm_factor[1, current_ofdmsymb_index, current_pilots_subcarrier[current_pilot_subcarrier_index] : current_pilots_subcarrier[current_pilot_subcarrier_index + 1]] = (-1) * (alpha - 1) * (alpha + 1)
                        freq_interpolation_norm_factor[2, current_ofdmsymb_index, current_pilots_subcarrier[current_pilot_subcarrier_index] : current_pilots_subcarrier[current_pilot_subcarrier_index + 1]] = alpha * (alpha + 1) * 0.5

        return tf.convert_to_tensor(freq_interpolation_index, dtype=tf.int32), tf.convert_to_tensor(freq_interpolation_norm_factor, dtype=tf.complex64), \
                tf.convert_to_tensor(time_interpolation_index, dtype=tf.int32), tf.convert_to_tensor(time_interpolation_norm_factor, dtype=tf.complex64)

    def call(self, inputs):
        """
            The second order piecewise polynomial interpolated channel at l (Lm <= l < L(m+1)): alpha*(alpha - 1)*0.5 * H[m-1] - (alpha - 1)*(alpha + 1) * H[m] + alpha*(alpha + 1)*0.5 * H[m+1]
            With alpha = l/L and L is assumed to be the distance between pilot[m] and pilot[m+1] equal to distance between pilot[m-1] and pilot[m]
            
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
        f_estimated_channels_2 = tf.gather_nd(f_estimated_channels, indices=freq_index[2])

        ## Process interpolation
        updates = f_estimated_channels_0 * freq_norm[0] + f_estimated_channels_1 * freq_norm[1] + f_estimated_channels_2 * freq_norm[2]
        indices = tf.expand_dims(self._unique_pilots_ofdmsymb, axis=1)
        shape = tf.shape(estimated_channels)
        # Needed to avoid different shapes caused by the following if condition, place zeros OFDM symbols at the missing index in time (if any) to be able to process the time interpolation
        interpolated_channels = tf.scatter_nd(indices, updates, shape)
        
        # Time interpolation (if needed)
        if self._nb_unique_pilots_ofdmsymb != self._nb_ofdmsymb:
            time_index = self._time_interpolation_index
            time_norm = sn.utils.expand_to_rank(self._time_interpolation_norm_factor, 4, axis=2)

            t_interpolated_channels_0 = tf.gather(interpolated_channels, indices=time_index[0], axis=0)
            t_interpolated_channels_1 = tf.gather(interpolated_channels, indices=time_index[1], axis=0)
            t_interpolated_channels_2 = tf.gather(interpolated_channels, indices=time_index[2], axis=0)

            interpolated_channels = t_interpolated_channels_0 * time_norm[0] + t_interpolated_channels_1 * time_norm[1] + t_interpolated_channels_2 * time_norm[2]

        interpolated_channels = tf.transpose(interpolated_channels, perm=[2,0,1]) # reset batch size at the beginning

        if self._dc_null:
            interpolated_channels = tf.gather(interpolated_channels, indices=self._data_pilot_ind, axis=-1) # get data and pilots, no DC

        return interpolated_channels # output shape = [batch_size, num_data_symbols, num_effective_subcarriers]
