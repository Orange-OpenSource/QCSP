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

from ccsk import cross_correlation_c


class DFTLinkEqualizer(tf.keras.layers.Layer):

    def __init__(self, root_sequence, resource_grid, normalization_mode='as_proba', **kwargs):
        super().__init__(**kwargs)
        
        self._root_sequence = root_sequence
        self._normalization_mode = normalization_mode
        self._removed_nulled_scs = sn.ofdm.RemoveNulledSubcarriers(resource_grid)


    def call(self, inputs):
        # input shape = [batch_size, num_ofdm_symb, fft_size] || [batch_size, num_ofdm_symb, fft_size]
        x, dft_h_est = inputs

        # For LS channel estimation, the relative_shift_probability_matrix can be equivalently computed that way, only reliying on the values of x
        # In this case, there is no need for "channel estimation" as there is no channel estimation and we simply correlate each OFDM symbol with each other
        # relative_shift_probability_matrix = tf.abs(tf.signal.fft(x[:,tf.newaxis,:,:] * tf.math.conj(x[:,:,tf.newaxis,:])))


        # ======================================================================================================================================================================== #
        # Equalize each OFDM symbol with the channel estimated during each OFDM symbol.
        # Hence, each OFDM symbol is equalized num_ofdm_symb times.
        # Perform normalized MRC equalization.
        ## Ex. index [None, i, :, :] represents the whole frame equalized with the channel estimated during OFDM symbol i.
        dft_h_est = tf.expand_dims(dft_h_est, axis=2) # output shape = [batch_size, num_ofdm_symb, 1, fft_size]
        x = tf.expand_dims(x, axis=1) # output shape = [batch_size, 1, num_ofdm_symb, fft_size]
        x = x * tf.math.conj(dft_h_est) # output shape = [batch_size, num_ofdm_symb, num_ofdm_symb, fft_size]
        x = self._removed_nulled_scs(x) # output shape = [batch_size, num_ofdm_symb, num_ofdm_symb, num_effective_subcarriers = ccsk_length = fft_size]
        
        # ======================================================================================================================================================================== #
        # For each OFDM symbol (all equalization), compute the correlation with the root sequence zc0.
        # The result will be a complex, because of channel estimation mismatch, uncorrelated channel in time, noise, and constant term of the ZC depending on the actual shift.
        # We take the absolute value. If perfect, it should be fill with 1. and 0.
        # The correlation will represent the shift of the current OFDM symbol, relatively to the shift of the CCSK in the OFDM symbol used for channel estimation.
        ## Ex. index [None, i, :, :] will represent the shift of all OFDM symbols relatively to the shift in symbol i (x/i). 
        ## Hence [None, i, i, :] should look like [1., 0., 0., ...], because the shift relatively to itself is 0 (i/i).
        ## And [None, i, m, :] should have a 1. at the index pm-pi.
        ## The shift should be always read as positive, modulo fft_size. So shift of 1 relative to 0 is 1 (=1-0). But shift of 0 relative to 1 is fft_size-1 (because 1 + fft_size-1 [fft_size] = 0)
        relative_shift_probability_matrix = cross_correlation_c(x, self._root_sequence, out_operator='abs') # output shape = [batch_size, num_ofdm_symb, num_ofdm_symb, fft_size]


        # Normalization step
        if self._normalization_mode == 'as_proba':
            # This scaling is used to make the cross-correlation looks like a probability
            relative_shift_probability_matrix = relative_shift_probability_matrix / tf.reduce_sum(relative_shift_probability_matrix, axis=-1, keepdims=True) # output shape = [batch_size, num_ofdm_symb, num_ofdm_symb, fft_size]
        
        return relative_shift_probability_matrix