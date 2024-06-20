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


class SplineCubicInterpolator(tf.keras.layers.Layer):

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

        self._freq_knots_index, self._time_knots_index = self._generate_knots_index_()
        self._freq_knots_position, self._time_knots_position = self._generate_knots_position_()
        self._freq_range, self._time_range = self._generate_range_()
        self._freq_tridiag, self._time_tridiag = self._generate_tridiag_()
        self._old_freq_sderivative_y_position, self._time_sderivative_y_position = self._generate_sderivative_y_position_()
        self._freq_sderivative_range, self._time_sderivative_range = self._generate_sderivative_range_()
        self._freq_constant, self._time_constant = self._generate_constants_()

        # Modification of _freq_knots_index for tensorflow indexing support
        freq_index_array = []
        pilot_ofdmsymb_index = tf.expand_dims(tf.repeat(tf.range(self._nb_unique_pilots_ofdmsymb, dtype=tf.int32), self._num_effective_subcarriers_dc, axis=0), axis=1)
        for ind in range(2):
            freq_index = tf.expand_dims(sn.utils.flatten_last_dims(self._freq_knots_index[ind,:,:], num_dims=2), axis=1)
            freq_index = tf.concat([pilot_ofdmsymb_index, freq_index], axis=1)
            freq_index = tf.reshape(freq_index, shape=[1, self._nb_unique_pilots_ofdmsymb, self._num_effective_subcarriers_dc, 2])
            freq_index_array.append(freq_index)
        self._freq_knots_index = tf.concat(freq_index_array, axis=0)

        # Modification of _old_freq_sderivative_y_position for tensorflow indexing support
        freq_index_array = [] 
        max_num_pilot_in_ofdmsymb = self._old_freq_sderivative_y_position.shape[2]
        pilot_ofdmsymb_index = tf.expand_dims(tf.repeat(tf.range(self._nb_unique_pilots_ofdmsymb, dtype=tf.int32), max_num_pilot_in_ofdmsymb, axis=0), axis=1)
        for ind in range(3):
            freq_index = tf.expand_dims(sn.utils.flatten_last_dims(self._old_freq_sderivative_y_position[ind,:,:], num_dims=2), axis=1)
            freq_index = tf.concat([pilot_ofdmsymb_index, freq_index], axis=1)
            freq_index = tf.reshape(freq_index, shape=[1, self._nb_unique_pilots_ofdmsymb, max_num_pilot_in_ofdmsymb, 2])
            freq_index_array.append(freq_index)
        self._freq_sderivative_y_position = tf.concat(freq_index_array, axis=0)


    def _generate_knots_index_(self):
        # Convert tensors to numpy arrays
        nb_unique_pilots_ofdmsymb = np.array(self._nb_unique_pilots_ofdmsymb)
        nb_subcarriers = np.array(self._num_effective_subcarriers_dc) # the interpolation is limited to effective subcarrier, including the DC
        nb_ofdmsymb = np.array(self._nb_ofdmsymb)
        pilots_ofdmsymb_index = np.array(self._pilots_ofdmsymb_index)
        pilots_subcarrier_index = np.array(self._pilots_subcarrier_index)

        # Generate the knots index for all interpolated positions.
        ## For instance, in [X[i];X[i+1]] the current knots index is i and next knots index is i+1
        ## The values will be used in the final interpolation formula to get the right second derivatives and Y[i] values
        ## Shape[0] represents the current/next knots index.
        ## 0 => i, 1 => i+1
        freq_knots_index = np.empty(shape=(2, nb_unique_pilots_ofdmsymb, nb_subcarriers), dtype=np.int32)
        time_knots_index = np.empty(shape=(2, nb_ofdmsymb), dtype=np.int32)

        ## Get the OFDM symbols containing at least one pilot
        unique_pilots_ofdmsymb, unique_count = np.unique(pilots_ofdmsymb_index, return_counts=True)

        # Compute the index for time interpolation: knot = OFDM symbol
        ## First, process the boundary cases where the full interpolation formula does not apply
        ## For the extrapolation, we consider each extrapolated point as a knot with value corresponding to the first or last knot (depending on the closest)
        ## Hence the second derivative will be zero at all the extrapolated points and the Y[i] values will be equal to Y[0] or Y[-1].
        ### Before the first knot
        if unique_pilots_ofdmsymb[0] != 0:
            time_knots_index[:, 0:unique_pilots_ofdmsymb[0]] = 0

        ### After the last knot (or just the last knot)
        time_knots_index[:, unique_pilots_ofdmsymb[-1]:] = unique_pilots_ofdmsymb.size - 1
        
        if unique_pilots_ofdmsymb.size > 1:
            for current_pilot_index in np.arange(0, unique_pilots_ofdmsymb.size - 1):
                time_knots_index[0, unique_pilots_ofdmsymb[current_pilot_index] : unique_pilots_ofdmsymb[current_pilot_index + 1]] = current_pilot_index
                time_knots_index[1, unique_pilots_ofdmsymb[current_pilot_index] : unique_pilots_ofdmsymb[current_pilot_index + 1]] = current_pilot_index + 1

        # Compute the index for the frequency interpolation: knot = subcarrier
        current_first_pilot_index = 0
        ## Each OFDM symbol containing at least one pilot is treated independently in the loop
        for (current_ofdmsymb_index, current_nb_pilots_in_ofdmsymb) in zip(np.arange(nb_unique_pilots_ofdmsymb), unique_count):
            current_pilots_subcarrier = pilots_subcarrier_index[current_first_pilot_index : current_first_pilot_index + current_nb_pilots_in_ofdmsymb]
            current_first_pilot_index += current_nb_pilots_in_ofdmsymb

            ## First, process the boundary cases where the full interpolation formula does not apply
            ## For the extrapolation, we consider each extrapolated point as a knot with value corresponding to the first or last knot (depending on the closest)
            ## Hence the second derivative will be zero at all the extrapolated points and the Y[i] values will be equal to Y[0] or Y[-1].
            ### Before the first knot
            if current_pilots_subcarrier[0] != 0:
                freq_knots_index[:, current_ofdmsymb_index, 0:current_pilots_subcarrier[0]] = 0
            
            ### After the last knot (or just the last knot)
            freq_knots_index[:, current_ofdmsymb_index, current_pilots_subcarrier[-1]:] = current_pilots_subcarrier.size - 1
        
            if current_pilots_subcarrier.size > 1:
                for current_pilot_index in np.arange(0, current_pilots_subcarrier.size - 1):
                    freq_knots_index[0, current_ofdmsymb_index, current_pilots_subcarrier[current_pilot_index] : current_pilots_subcarrier[current_pilot_index + 1]] = current_pilot_index
                    freq_knots_index[1, current_ofdmsymb_index, current_pilots_subcarrier[current_pilot_index] : current_pilots_subcarrier[current_pilot_index + 1]] = current_pilot_index + 1

        return tf.convert_to_tensor(freq_knots_index, dtype=tf.int32), tf.convert_to_tensor(time_knots_index, dtype=tf.int32)

    def _generate_knots_position_(self):
        # Convert tensors to numpy arrays
        nb_unique_pilots_ofdmsymb = np.array(self._nb_unique_pilots_ofdmsymb)
        nb_subcarriers = np.array(self._num_effective_subcarriers_dc) # the interpolation is limited to effective subcarrier, including the DC
        nb_ofdmsymb = np.array(self._nb_ofdmsymb)
        pilots_ofdmsymb_index = np.array(self._pilots_ofdmsymb_index)
        pilots_subcarrier_index = np.array(self._pilots_subcarrier_index)

        # Compute the positions of the knots (pilots) in the whole frame.
        ## The position is noted X[i] and represents the OFDM symbol or subcarrier index in the whole frame array
        ## Shape[0] represents previous/current/next knots index.
        ## 0 => i-1, 1 => i, 2 => i+1
        freq_knots_position = np.empty(shape=(3, nb_unique_pilots_ofdmsymb, nb_subcarriers), dtype=np.int32)
        time_knots_position = np.empty(shape=(3, nb_ofdmsymb), dtype=np.int32)

        ## Get the OFDM symbols containing at least one pilot
        unique_pilots_ofdmsymb, unique_count = np.unique(pilots_ofdmsymb_index, return_counts=True)
        
        # Compute the position for time interpolation : knot = OFDM symbol index
        ## First, process the boundary cases where the full interpolation formula does not apply
        ## For the extrapolation, we consider each extrapolated point as a knot so the distance between successive knots is one
        ## This will allow us to directly compute the range as 1, so that (x - X[i]) = 0 and (X[i+1] - x) = 1
        ### Before first knot
        if unique_pilots_ofdmsymb[0] != 0:
            time_knots_position[0, 0:unique_pilots_ofdmsymb[0]] = np.arange(unique_pilots_ofdmsymb[0]) - 1
            time_knots_position[1, 0:unique_pilots_ofdmsymb[0]] = np.arange(unique_pilots_ofdmsymb[0])
            time_knots_position[2, 0:unique_pilots_ofdmsymb[0]] = np.arange(unique_pilots_ofdmsymb[0]) + 1

        ### After last knot
        if unique_pilots_ofdmsymb[-1] != (nb_ofdmsymb - 1):
            time_knots_position[0, unique_pilots_ofdmsymb[-1]:] = np.arange(unique_pilots_ofdmsymb[-1], nb_ofdmsymb) - 1
            time_knots_position[1, unique_pilots_ofdmsymb[-1]:] = np.arange(unique_pilots_ofdmsymb[-1], nb_ofdmsymb)
            time_knots_position[2, unique_pilots_ofdmsymb[-1]:] = np.arange(unique_pilots_ofdmsymb[-1], nb_ofdmsymb) + 1
        
        ### Process the last knot, otherwise it will not be processed correctly
        if unique_pilots_ofdmsymb.size == 1:
            time_knots_position[0, unique_pilots_ofdmsymb[-1]] = unique_pilots_ofdmsymb[-1] - 1
            time_knots_position[1, unique_pilots_ofdmsymb[-1]] = unique_pilots_ofdmsymb[-1]
            time_knots_position[2, unique_pilots_ofdmsymb[-1]] = unique_pilots_ofdmsymb[-1] + 1
        else:
            time_knots_position[0, unique_pilots_ofdmsymb[-1]] = unique_pilots_ofdmsymb[-2]
            time_knots_position[1, unique_pilots_ofdmsymb[-1]] = unique_pilots_ofdmsymb[-1]
            time_knots_position[2, unique_pilots_ofdmsymb[-1]] = unique_pilots_ofdmsymb[-1] + 1
        
        if unique_pilots_ofdmsymb.size > 1:
            ### Between first and second knots
            time_knots_position[0, unique_pilots_ofdmsymb[0] : unique_pilots_ofdmsymb[1]] = unique_pilots_ofdmsymb[0] - 1
            time_knots_position[1, unique_pilots_ofdmsymb[0] : unique_pilots_ofdmsymb[1]] = unique_pilots_ofdmsymb[0]
            time_knots_position[2, unique_pilots_ofdmsymb[0] : unique_pilots_ofdmsymb[1]] = unique_pilots_ofdmsymb[1]

            ## Full interpolation formula can operate here
            if unique_pilots_ofdmsymb.size > 2:
                for current_pilot_position in np.arange(1, unique_pilots_ofdmsymb.size - 1):
                    time_knots_position[0, unique_pilots_ofdmsymb[current_pilot_position] : unique_pilots_ofdmsymb[current_pilot_position + 1]] = unique_pilots_ofdmsymb[current_pilot_position - 1]
                    time_knots_position[1, unique_pilots_ofdmsymb[current_pilot_position] : unique_pilots_ofdmsymb[current_pilot_position + 1]] = unique_pilots_ofdmsymb[current_pilot_position]
                    time_knots_position[2, unique_pilots_ofdmsymb[current_pilot_position] : unique_pilots_ofdmsymb[current_pilot_position + 1]] = unique_pilots_ofdmsymb[current_pilot_position + 1]

        # Compute the position for the frequency interpolation: knot = subcarrier position 
        current_first_pilot_position = 0
        ## Each OFDM symbol containing at least one pilot is treated independently in the loop
        for (current_ofdmsymb_index, current_nb_pilots_in_ofdmsymb) in zip(np.arange(nb_unique_pilots_ofdmsymb), unique_count):
            current_pilots_subcarrier = pilots_subcarrier_index[current_first_pilot_position : current_first_pilot_position + current_nb_pilots_in_ofdmsymb]
            current_first_pilot_position += current_nb_pilots_in_ofdmsymb

            ## First, process the boundary cases where the full interpolation formula does not apply
            ### Before first knot
            if current_pilots_subcarrier[0] != 0:
                freq_knots_position[0, current_ofdmsymb_index, 0:current_pilots_subcarrier[0]] = np.arange(current_pilots_subcarrier[0]) - 1
                freq_knots_position[1, current_ofdmsymb_index, 0:current_pilots_subcarrier[0]] = np.arange(current_pilots_subcarrier[0])
                freq_knots_position[2, current_ofdmsymb_index, 0:current_pilots_subcarrier[0]] = np.arange(current_pilots_subcarrier[0]) + 1

            ### After the last knot
            if current_pilots_subcarrier[-1] != (nb_subcarriers - 1):
                freq_knots_position[0, current_ofdmsymb_index, current_pilots_subcarrier[-1]:] = np.arange(current_pilots_subcarrier[-1], nb_subcarriers) - 1
                freq_knots_position[1, current_ofdmsymb_index, current_pilots_subcarrier[-1]:] = np.arange(current_pilots_subcarrier[-1], nb_subcarriers)
                freq_knots_position[2, current_ofdmsymb_index, current_pilots_subcarrier[-1]:] = np.arange(current_pilots_subcarrier[-1], nb_subcarriers) + 1
            
            ### Process the last knot independently, otherwise it will not be processed correctly
            if current_pilots_subcarrier.size == 1:
                freq_knots_position[0, current_ofdmsymb_index, current_pilots_subcarrier[-1]] = current_pilots_subcarrier[-1] - 1
                freq_knots_position[1, current_ofdmsymb_index, current_pilots_subcarrier[-1]] = current_pilots_subcarrier[-1]
                freq_knots_position[2, current_ofdmsymb_index, current_pilots_subcarrier[-1]] = current_pilots_subcarrier[-1] + 1
            else:
                freq_knots_position[0, current_ofdmsymb_index, current_pilots_subcarrier[-1]] = current_pilots_subcarrier[-2]
                freq_knots_position[1, current_ofdmsymb_index, current_pilots_subcarrier[-1]] = current_pilots_subcarrier[-1]
                freq_knots_position[2, current_ofdmsymb_index, current_pilots_subcarrier[-1]] = current_pilots_subcarrier[-1] + 1
        
            if current_pilots_subcarrier.size > 1:
                ### Between first and second knots
                freq_knots_position[0, current_ofdmsymb_index, current_pilots_subcarrier[0] : current_pilots_subcarrier[1]] = current_pilots_subcarrier[0] - 1
                freq_knots_position[1, current_ofdmsymb_index, current_pilots_subcarrier[0] : current_pilots_subcarrier[1]] = current_pilots_subcarrier[0]
                freq_knots_position[2, current_ofdmsymb_index, current_pilots_subcarrier[0] : current_pilots_subcarrier[1]] = current_pilots_subcarrier[1]

                ## Full interpolation formula can apply here
                if current_pilots_subcarrier.size > 2:
                    for current_pilot_position in np.arange(1, current_pilots_subcarrier.size - 1):
                        freq_knots_position[0, current_ofdmsymb_index, current_pilots_subcarrier[current_pilot_position] : current_pilots_subcarrier[current_pilot_position + 1]] = current_pilots_subcarrier[current_pilot_position - 1]
                        freq_knots_position[1, current_ofdmsymb_index, current_pilots_subcarrier[current_pilot_position] : current_pilots_subcarrier[current_pilot_position + 1]] = current_pilots_subcarrier[current_pilot_position]
                        freq_knots_position[2, current_ofdmsymb_index, current_pilots_subcarrier[current_pilot_position] : current_pilots_subcarrier[current_pilot_position + 1]] = current_pilots_subcarrier[current_pilot_position + 1]
        
        return tf.convert_to_tensor(freq_knots_position, dtype=tf.int32), tf.convert_to_tensor(time_knots_position, dtype=tf.int32)

    def _generate_range_(self):
        # Convert tensors to numpy arrays
        nb_unique_pilots_ofdmsymb = np.array(self._nb_unique_pilots_ofdmsymb)
        nb_subcarriers = np.array(self._num_effective_subcarriers_dc) # the interpolation is limited to effective subcarrier, including the DC
        nb_ofdmsymb = np.array(self._nb_ofdmsymb)
        time_knots_position = np.array(self._time_knots_position)
        freq_knots_position = np.array(self._freq_knots_position)

        # Compute the distance between successive knots
        # 0 => X[i-1]-X[i], 1 => X[i]-X[i+1]
        freq_range = np.empty(shape=(2, nb_unique_pilots_ofdmsymb, nb_subcarriers), dtype=np.int32)
        time_range = np.empty(shape=(2, nb_ofdmsymb), dtype=np.int32)

        time_range[0, :] = time_knots_position[1, :] - time_knots_position[0, :]
        time_range[1, :] = time_knots_position[2, :] - time_knots_position[1, :]

        freq_range[0, :, :] = freq_knots_position[1, :, :] - freq_knots_position[0, :, :]
        freq_range[1, :, :] = freq_knots_position[2, :, :] - freq_knots_position[1, :, :]
        
        return tf.convert_to_tensor(freq_range, dtype=tf.complex64), tf.convert_to_tensor(time_range, dtype=tf.complex64)

    def _generate_tridiag_(self):
        # Convert tensors to numpy arrays
        nb_unique_pilots_ofdmsymb = np.array(self._nb_unique_pilots_ofdmsymb)
        pilots_ofdmsymb_index = np.array(self._pilots_ofdmsymb_index)
        pilots_subcarrier_index = np.array(self._pilots_subcarrier_index)
        time_range = np.real(np.array(self._time_range))
        freq_range = np.real(np.array(self._freq_range))

        # Generate the tridiagonal matrix used to compute the second derivative on each knot
        ## We assume natural condition, so second derivatives at the edges are 0
        ## The square matrix size equals to the number of knots in the interpolation sequence
        ## In frequency, this means that the size is related to the number of pilots in each OFDM symbol
        ## Since the number of pilots may vary, we force the matrix to the max size
        ## The additional lines/columns are set to identity, while the correspond local second derivatives will be 0
        
        # Maximum number of pilots in one OFDM symbol
        unique_pilots_ofdmsymb, unique_count = np.unique(pilots_ofdmsymb_index, return_counts=True)
        max_pilots_ofdmsymb = np.max(unique_count)

        freq_tridiag = np.zeros(shape=(nb_unique_pilots_ofdmsymb, max_pilots_ofdmsymb, max_pilots_ofdmsymb), dtype=np.float32)
        time_tridiag = np.zeros(shape=(nb_unique_pilots_ofdmsymb, nb_unique_pilots_ofdmsymb), dtype=np.float32)

        # Compute the time tridiagonal matrix.
        time_tridiag[0,0] = 1.
        time_tridiag[-1,-1] = 1.

        # Take the values of the range h[i] for each interval (between X[i] and X[i+1])
        # The last value (at X[-1]) will not be used.
        current_interval_time_range = time_range[1,:][unique_pilots_ofdmsymb]

        if unique_pilots_ofdmsymb.size > 2:
            # Do not consider first and last lines, set the diagonal, upper and lower diagonal
            for row in np.arange(1, nb_unique_pilots_ofdmsymb - 1):
                for col in np.arange(nb_unique_pilots_ofdmsymb):
                    if row == col:
                        time_tridiag[row,col] = (current_interval_time_range[row - 1] + current_interval_time_range[row]) / 3
                    elif row == (col + 1):
                        time_tridiag[row,col] = current_interval_time_range[row] / 6
                    elif row == (col - 1):
                        time_tridiag[row,col] = current_interval_time_range[row - 1] / 6

        # Compute the inverse, the matrix cannot be singular by construction
        time_tridiag = np.linalg.inv(time_tridiag)
        
        # Compute the frequency tridiagonal matrix
        freq_tridiag[:,0,0] = 1.

        # Proxy to the current subcarrier range for all considered OFDM symbols
        interval_freq_range = freq_range[1, :, :]

        current_first_pilot_index = 0
        ## Each OFDM symbol containing at least one pilot is treated independently in the loop
        for (current_ofdmsymb_index, current_nb_pilots_in_ofdmsymb) in zip(np.arange(nb_unique_pilots_ofdmsymb), unique_count):
            current_pilots_subcarrier = pilots_subcarrier_index[current_first_pilot_index : current_first_pilot_index + current_nb_pilots_in_ofdmsymb]
            current_first_pilot_index += current_nb_pilots_in_ofdmsymb

            # Fill the end of the diagonal with ones, from last pilot to extended index
            for diag_index in np.arange(current_nb_pilots_in_ofdmsymb-1, max_pilots_ofdmsymb):
                freq_tridiag[current_ofdmsymb_index, diag_index, diag_index] = 1.

            if current_nb_pilots_in_ofdmsymb > 2:
                # Take the values of the range h[i] for each interval (between X[i] and X[i+1])
                # The last value (at X[-1]) will not be used.
                current_interval_freq_range = interval_freq_range[current_ofdmsymb_index, current_pilots_subcarrier]

                for row in np.arange(1, current_nb_pilots_in_ofdmsymb - 1):
                    for col in np.arange(current_nb_pilots_in_ofdmsymb):
                        if row == col:
                            freq_tridiag[current_ofdmsymb_index, row, col] = (current_interval_freq_range[row - 1] + current_interval_freq_range[row]) / 3
                        elif row == (col + 1):
                            freq_tridiag[current_ofdmsymb_index, row, col] = current_interval_freq_range[row] / 6
                        elif row == (col - 1):
                            freq_tridiag[current_ofdmsymb_index, row, col] = current_interval_freq_range[row - 1] / 6

        # Compute the inverse, the matrix cannot be singular by construction
        freq_tridiag = np.linalg.inv(freq_tridiag)

        return tf.convert_to_tensor(freq_tridiag, dtype=tf.complex64), tf.convert_to_tensor(time_tridiag, dtype=tf.complex64)

    def _generate_sderivative_y_position_(self):
        # Convert tensors to numpy arrays
        nb_unique_pilots_ofdmsymb = np.array(self._nb_unique_pilots_ofdmsymb)
        pilots_ofdmsymb_index = np.array(self._pilots_ofdmsymb_index)
        pilots_subcarrier_index = np.array(self._pilots_subcarrier_index)

        # Compute the positions of the knots in the frame to get the Y[i] values. 
        # From the Y[i] values, we will compute the first step of the second derivative vector, based on the Y[i-1], Y[i], Y[i+1] and the range h[i] and h[i-1].
        # ( Y[i+1] - Y[i] / h[i] ) - ( Y[i] - Y[i-1] / h[i-1] )
        # For index i, we need knots positions of i-1, i and i+1. The vector length = the number of pilots in the OFDM symbol / frame.
        # For frequency, since the number of pilots in one OFDM symbol may vary, the size is fixed to the maximum number of pilots in one OFDM symbol.
        # The extra values should be zero after the computation of the previous formula.
        # We assume second derivative for first and last knot are null (said as natural condition). 
        # Shape[0] position of Y[i]: 0 => i-1, 1 => i, 2 => i+1
        
        # Maximum number of pilots in one OFDM symbol
        unique_pilots_ofdmsymb, unique_count = np.unique(pilots_ofdmsymb_index, return_counts=True)
        max_pilots_ofdmsymb = np.max(unique_count)

        freq_sderivative_y_position = np.empty(shape=(3, nb_unique_pilots_ofdmsymb, max_pilots_ofdmsymb), dtype=np.int32)
        time_sderivative_y_position = np.empty(shape=(3, nb_unique_pilots_ofdmsymb), dtype=np.int32)

        # Compute the index for time interpolation: knot = OFDM symbol index
        ## For the first and last knots, the second derivative should be zero after computation of the formula
        time_sderivative_y_position[:,0] = unique_pilots_ofdmsymb[0]
        time_sderivative_y_position[:,-1] = unique_pilots_ofdmsymb[-1]
        
        if unique_pilots_ofdmsymb.size > 2:
            for current_pilot_index in np.arange(1, unique_pilots_ofdmsymb.size - 1):
                time_sderivative_y_position[0, current_pilot_index] = unique_pilots_ofdmsymb[current_pilot_index - 1]
                time_sderivative_y_position[1, current_pilot_index] = unique_pilots_ofdmsymb[current_pilot_index]
                time_sderivative_y_position[2, current_pilot_index] = unique_pilots_ofdmsymb[current_pilot_index + 1]

        # Compute the index for the frequency interpolation: knot = subcarrier index
        current_first_pilot_index = 0
        ## Each OFDM symbol containing at least one pilot is treated independently in the loop
        for (current_ofdmsymb_index, current_nb_pilots_in_ofdmsymb) in zip(np.arange(nb_unique_pilots_ofdmsymb), unique_count):
            current_pilots_subcarrier = pilots_subcarrier_index[current_first_pilot_index : current_first_pilot_index + current_nb_pilots_in_ofdmsymb]
            current_first_pilot_index += current_nb_pilots_in_ofdmsymb

            ## For the first knot, the second derivative should be zero after computation of the formula
            freq_sderivative_y_position[:, current_ofdmsymb_index, 0] = current_pilots_subcarrier[0]

            if current_nb_pilots_in_ofdmsymb > 1:
                ## For the last knot and the extra values, the second derivative should be zero
                for current_pilot_index in np.arange(current_nb_pilots_in_ofdmsymb-1, max_pilots_ofdmsymb):
                    freq_sderivative_y_position[:, current_ofdmsymb_index, current_pilot_index] = current_pilots_subcarrier[-1]
        
            if current_nb_pilots_in_ofdmsymb > 2:
                for current_pilot_index in np.arange(1, current_pilots_subcarrier.size - 1):
                    freq_sderivative_y_position[0, current_ofdmsymb_index, current_pilot_index] = current_pilots_subcarrier[current_pilot_index - 1]
                    freq_sderivative_y_position[1, current_ofdmsymb_index, current_pilot_index] = current_pilots_subcarrier[current_pilot_index]
                    freq_sderivative_y_position[2, current_ofdmsymb_index, current_pilot_index] = current_pilots_subcarrier[current_pilot_index + 1]

        return tf.convert_to_tensor(freq_sderivative_y_position, dtype=tf.int32), tf.convert_to_tensor(time_sderivative_y_position, dtype=tf.int32)

    def _generate_sderivative_range_(self):
        # Convert tensors to numpy arrays
        nb_unique_pilots_ofdmsymb = np.array(self._nb_unique_pilots_ofdmsymb)
        pilots_ofdmsymb_index = np.array(self._pilots_ofdmsymb_index)
        pilots_subcarrier_index = np.array(self._pilots_subcarrier_index)
        time_range = np.real(np.array(self._time_range))
        freq_range = np.real(np.array(self._freq_range))

        # Compute the range values used in the second derivative: h[i] and h[i-1]
        # ( Y[i+1] - Y[i] / h[i] ) - ( Y[i] - Y[i-1] / h[i-1] )
        # The range are already computed for the whole interpolation formula, just need to pick the right index to match the knots index
        unique_pilots_ofdmsymb, unique_count = np.unique(pilots_ofdmsymb_index, return_counts=True)
        max_pilots_ofdmsymb = np.max(unique_count)

        # In frequency, the extra values are set to one
        freq_sderivative_range = np.ones(shape=(2, nb_unique_pilots_ofdmsymb, max_pilots_ofdmsymb), dtype=np.int32)
        time_sderivative_range = np.empty(shape=(2, nb_unique_pilots_ofdmsymb), dtype=np.int32)

        # In time, take the values at OFDM symbols with pilots
        time_sderivative_range = time_range[:, unique_pilots_ofdmsymb]
        
        current_first_pilot_index = 0
        ## Each OFDM symbol containing at least one pilot is treated independently in the loop
        for (current_ofdmsymb_index, current_nb_pilots_in_ofdmsymb) in zip(np.arange(nb_unique_pilots_ofdmsymb), unique_count):
            current_pilots_subcarrier = pilots_subcarrier_index[current_first_pilot_index : current_first_pilot_index + current_nb_pilots_in_ofdmsymb]
            current_first_pilot_index += current_nb_pilots_in_ofdmsymb

            freq_sderivative_range[:, current_ofdmsymb_index, 0:current_pilots_subcarrier.size] = freq_range[:, current_ofdmsymb_index, current_pilots_subcarrier]
        
        return tf.convert_to_tensor(freq_sderivative_range, dtype=tf.complex64), tf.convert_to_tensor(time_sderivative_range, dtype=tf.complex64)

    def _generate_constants_(self):
        # Convert tensors to numpy arrays
        nb_unique_pilots_ofdmsymb = np.array(self._nb_unique_pilots_ofdmsymb)
        nb_subcarriers = np.array(self._num_effective_subcarriers_dc) # the interpolation is limited to effective subcarrier, including the DC
        nb_ofdmsymb = np.array(self._nb_ofdmsymb)
        time_range = np.real(np.array(self._time_range))
        freq_range = np.real(np.array(self._freq_range))
        time_knots_position = np.array(self._time_knots_position)
        freq_knots_position = np.array(self._freq_knots_position)

        # Compute the constants values that are used in the interpolation formula
        # See the interpolate() function for more information
        # Shape[0]: 0 => c0, 1 => c1, 2 => c2, 3 => c3, 4 => c4
        freq_constant_array = np.empty(shape=(5, nb_unique_pilots_ofdmsymb, nb_subcarriers), dtype=np.float32)
        time_constant_array = np.empty(shape=(5, nb_ofdmsymb), dtype=np.float32)
        
        # The interpolated index, the x values for time
        time_x_index = np.arange(nb_ofdmsymb, dtype=np.int32)

        time_constant_array[0, :] = np.power(time_x_index - time_knots_position[1, :], 3) / (6 * time_range[1, :])
        time_constant_array[1, :] = np.power(time_knots_position[2, :] - time_x_index, 3) / (6 * time_range[1, :])
        time_constant_array[2, :] = time_range[1, :] / 6
        time_constant_array[3, :] = time_x_index - time_knots_position[1, :]
        time_constant_array[4, :] = time_knots_position[2, :] - time_x_index

        # The interpolated index, the x values for frequency
        freq_x_index = np.arange(nb_subcarriers, dtype=np.int32)
        freq_x_index = np.tile(freq_x_index, (nb_unique_pilots_ofdmsymb, 1))

        freq_constant_array[0, :, :] = np.power(freq_x_index - freq_knots_position[1, :, :], 3) / (6 * freq_range[1, :, :])
        freq_constant_array[1, :, :] = np.power(freq_knots_position[2, :, :] - freq_x_index, 3) / (6 * freq_range[1, :, :])
        freq_constant_array[2, :, :] = freq_range[1, :, :] / 6
        freq_constant_array[3, :, :] = freq_x_index - freq_knots_position[1, :, :]
        freq_constant_array[4, :, :] = freq_knots_position[2, :, :] - freq_x_index

        return tf.convert_to_tensor(freq_constant_array, dtype=tf.complex64), tf.convert_to_tensor(time_constant_array, dtype=tf.complex64)
        
    def call(self, inputs):
        """
        The interpolation occurs first in frequency, in OFDM symbols containing at least one pilot and then in time.
        The general formula that we used for spline cubic is the following:

        S[i](x) = z[i+1] * c0 + z[i] * c1 + ( (Y[i+1]/h[i]) - (z[i+1]*c2) ) * c3 + ( (Y[i]/h[i]) - (z[i]*c2) ) * c4
        
        with the following definition:
            x is the current interpolated index in section i, between the knots of index i and i+1, value of x matches subcarrier index or OFDM symbol index
            X[i] is the position of knot i as a subcarrier index or as an OFDM symbol
            Y[i] is the value at index X[i]
            h[i] is the distance between knots i and i+1, so that h[i] = X[i+1] - X[i]
            z[i] is the second derivative at knot i, it is derived as follow:
                - first compute for each knot the vector F[i] = ((Y[i+1]-Y[i]) / h[i]) - ((Y[i]-Y[i-1]) / h[i-1])
                - we consider natural condition, so that F[0] = F[-1] = 0
                - compute R, the inverse of a tridiagonal matrix (see https://fr.wikipedia.org/wiki/Spline#Algorithme_de_calcul)
                - z = matmul(R,F)
        
        X[i] values can be found in the _knots_position attributes
        h[i] values can be found in the _range attributes
        z[i] values can be computed based on _sderivative_ attributes. 
        The size of z depends on the number of pilots. To efficiently broadcast the value on the whole frame, we have to map an index of z for each interpolated position.
            The mapped index will match the index i of the knot. For all interpolated positions in interval [X[i],X[i+1]], the second derivative values from z would be at index i and i+1 to get z[i] and z[i+1].
        c0 -> c4 are the constants computed in _constant attributes. 
        Corresponding expressions are:
            c0 = ((x - X[i])**3) / (6*h[i])
            c1 = ((X[i+1] - x)**3) / (6*h[i])
            c2 = h[i] / 6
            c3 = x - X[i]
            c4 = X[i+1] - x

        Parameters
        ----------
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
        freq_sderivative_y_position = self._freq_sderivative_y_position
        freq_sderivative_range = tf.expand_dims(self._freq_sderivative_range, axis=-1)
        freq_tridiag = self._freq_tridiag
        old_freq_sderivative_y_position = self._old_freq_sderivative_y_position
        freq_knots_index = self._freq_knots_index
        freq_constant = tf.expand_dims(self._freq_constant, axis=-1)
        freq_range = tf.expand_dims(self._freq_range, axis=-1)

        # Frequency interpolation
        ## Get channels
        f_estimated_channels = tf.gather(estimated_channels, indices=self._unique_pilots_ofdmsymb, axis=0)

        f_estimated_channels_0 = tf.gather_nd(f_estimated_channels, indices=freq_sderivative_y_position[0])
        f_estimated_channels_1 = tf.gather_nd(f_estimated_channels, indices=freq_sderivative_y_position[1])
        f_estimated_channels_2 = tf.gather_nd(f_estimated_channels, indices=freq_sderivative_y_position[2])

        # Frequency interpolation
        current_sderivative = ((f_estimated_channels_2 - f_estimated_channels_1) / (freq_sderivative_range[1])) - ((f_estimated_channels_1 - f_estimated_channels_0) / (freq_sderivative_range[0]))
        current_sderivative = tf.matmul(freq_tridiag, current_sderivative)

        sderivative_0 = tf.gather_nd(old_freq_sderivative_y_position[1], indices=freq_knots_index[0])
        sderivative_1 = tf.gather_nd(old_freq_sderivative_y_position[1], indices=freq_knots_index[1])

        pilot_ofdmsymb_index = tf.expand_dims(tf.repeat(tf.range(self._nb_unique_pilots_ofdmsymb, dtype=tf.int32), self._num_effective_subcarriers_dc, axis=0), axis=1)
        
        updates = tf.expand_dims(sn.utils.flatten_last_dims(sderivative_0, num_dims=2), axis=1)
        sderivative_0 = tf.concat([pilot_ofdmsymb_index, updates], axis=1)
        sderivative_0 = tf.reshape(sderivative_0, shape=[self._nb_unique_pilots_ofdmsymb, self._num_effective_subcarriers_dc, 2])
        
        updates = tf.expand_dims(sn.utils.flatten_last_dims(sderivative_1, num_dims=2), axis=1)
        sderivative_1 = tf.concat([pilot_ofdmsymb_index, updates], axis=1)
        sderivative_1 = tf.reshape(sderivative_1, shape=[self._nb_unique_pilots_ofdmsymb, self._num_effective_subcarriers_dc, 2])

        y_i = tf.gather_nd(f_estimated_channels, indices=sderivative_0)
        y_ip1 = tf.gather_nd(f_estimated_channels, indices=sderivative_1)

        k_sderivative_0 = tf.gather_nd(current_sderivative, indices=freq_knots_index[0])
        k_sderivative_1 = tf.gather_nd(current_sderivative, indices=freq_knots_index[1])

        ## Process interpolation
        updates = k_sderivative_1 * freq_constant[0] + k_sderivative_0 * freq_constant[1] + ((y_ip1 / freq_range[1]) - (k_sderivative_1 * freq_constant[2])) * freq_constant[3] + \
                                ((y_i / freq_range[1]) - (k_sderivative_0 * freq_constant[2])) * freq_constant[4]
        indices = tf.expand_dims(self._unique_pilots_ofdmsymb, axis=1)
        shape = tf.shape(estimated_channels)
        # Needed to avoid different shapes caused by the following if condition, place zeros OFDM symbols at the missing index in time (if any) to be able to process the time interpolation
        interpolated_channels = tf.scatter_nd(indices, updates, shape)


        # Time interpolation (if needed)
        if self._nb_unique_pilots_ofdmsymb != self._nb_ofdmsymb:
            time_sderivative_y_position = self._time_sderivative_y_position
            time_sderivative_range = sn.utils.expand_to_rank(self._time_sderivative_range, 4, axis=2)
            time_tridiag = self._time_tridiag
            time_knots_index = self._time_knots_index
            time_constant = sn.utils.expand_to_rank(self._time_constant, 4, axis=2)
            time_range = sn.utils.expand_to_rank(self._time_range, 4, axis=2)

            t_current_sderivative_0 = tf.gather(interpolated_channels, indices=time_sderivative_y_position[0], axis=0)
            t_current_sderivative_1 = tf.gather(interpolated_channels, indices=time_sderivative_y_position[1], axis=0)
            t_current_sderivative_2 = tf.gather(interpolated_channels, indices=time_sderivative_y_position[2], axis=0)

            current_sderivative = ((t_current_sderivative_2 - t_current_sderivative_1) / (time_sderivative_range[1])) - ((t_current_sderivative_1 - t_current_sderivative_0) / (time_sderivative_range[0]))
            current_sderivative = tf.transpose(current_sderivative, perm=[2,0,1])
            current_sderivative = tf.matmul(time_tridiag, current_sderivative)
            current_sderivative = tf.transpose(current_sderivative, perm=[1,2,0])

            sderivative_0 = tf.gather(time_sderivative_y_position[1], indices=time_knots_index[0], axis=0)
            sderivative_1 = tf.gather(time_sderivative_y_position[1], indices=time_knots_index[1], axis=0)

            y_i = tf.gather(interpolated_channels, indices=sderivative_0, axis=0)
            y_ip1 = tf.gather(interpolated_channels, indices=sderivative_1, axis=0)

            k_sderivative_0 = tf.gather(current_sderivative, indices=time_knots_index[0], axis=0)
            k_sderivative_1 = tf.gather(current_sderivative, indices=time_knots_index[1], axis=0)

            interpolated_channels = k_sderivative_1 * time_constant[0] + k_sderivative_0 * time_constant[1] + ((y_ip1 / time_range[1]) - (k_sderivative_1 * time_constant[2])) * time_constant[3] + \
                                    ((y_i / time_range[1]) - (k_sderivative_0 * time_constant[2])) * time_constant[4]


        interpolated_channels = tf.transpose(interpolated_channels, perm=[2,0,1]) # reset batch size at the beginning

        if self._dc_null:
            interpolated_channels = tf.gather(interpolated_channels, indices=self._data_pilot_ind, axis=-1) # get data and pilots, no DC

        return interpolated_channels # output shape = [batch_size, num_ofdm_symbols, num_effective_subcarriers]
