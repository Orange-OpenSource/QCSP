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
import numpy as np

from ccsk import cross_correlation
from initializers import ComplexIdentity


class TimeChannelEstimation(tf.keras.layers.Layer):

    def __init__(self, resource_grid, root_sequence, time_est_output=False, submode='LS', references_shift_value=0, **kwargs): # u=1, oversampling=1, l_tot=1, l_min=1, covmatrix_cir=None
        super().__init__() # **kwargs
        self._resource_grid = resource_grid
        self._root_sequence = root_sequence
        self._ifft_zc0 = tf.expand_dims(sn.signal.ifft(root_sequence), axis=0)

        self._time_est_output = time_est_output

        self._references_shift_value = references_shift_value
        # [WIP] Compute the gather list to obtain the complete shift values, including the shifts of the pilots/refecence symbols.
        # Not fully implemented, see the call for more details. The same references_shift_value is used for every pilot/reference OFDM symbol.
        # No iterative mapping supported.
        self._gather_index_list = self._compute_gather_index_list

        # Time Channel Estimation Submode
        ## LS
        ## Perfect
        ## LMMSE_TrueP
        ## expBCOMPDP_TrueP
        ## uniformPDP_TrueP 
        ## expPDP_TrueP
        self._submode = submode
        self.LMMSE_TrueP_mode = False

        if self._submode == 'Perfect':
            # args needed: u, oversampling, l_tot, l_min
            self._N = tf.size(root_sequence)
            self._fft_size = resource_grid.fft_size
            self._sqrt_fft_size_complex = tf.complex(tf.sqrt(tf.cast(self._resource_grid.fft_size, dtype=tf.float32)), 0.)
            self._u = kwargs.get('u', 1)
            self._oversampling = kwargs.get('oversampling', 1)
            self._l_tot = kwargs.get('l_tot', 1)
            self._l_min = kwargs.get('l_min', 1)
            
            # Compute the shift mapping between a shift of the CCSK sequence and the shift of the IDFT of the CCSK sequence.
            ## Considering the root ZC0[k] sequence, root IDFT sequence is IDFT(ZC0)
            ## Considering a shift i of the root CCSK sequence ZC0[k-i], the IDFT(ZC0[k-i]) correspond to a shifted version of IDFT(ZC0). 
            ## The shift value is computed by this method, located at index i.
            self._ifft_shift_mapping = self._compute_ifft_shift_mapping

        elif self._submode in ['LMMSE_TrueP','expBCOMPDP_TrueP','uniformPDP_TrueP','expPDP_TrueP']:
            # args needed: covmatrix_cir
            self.LMMSE_TrueP_mode = True # True for all LMMSE submodes
            self._u = kwargs.get('u', 1)
            self._fft_size = resource_grid.fft_size
            self._fft_size_f = tf.cast(resource_grid.fft_size, tf.float32)

            # Construct a tensor of all possible covariance block, depending on the future actual value of the CCSK shift within the OFDM symbol.
            # This assume no iterative mapping and block pilots.
            basis_cov = tf.tile(tf.expand_dims(tf.range(start=0, limit=self._fft_size, delta=1, dtype=tf.float32), axis=1), multiples=[1, self._fft_size]) - tf.tile(tf.expand_dims(tf.range(start=0, limit=self._fft_size, delta=1, dtype=tf.float32), axis=0), multiples=[self._fft_size, 1]) # out shape = [fft_size, fft_size]
            self._ccsk_cov = tf.math.exp(tf.complex(0., 2*np.pi * self._u * tf.reshape(tf.range(start=0, limit=self._fft_size, dtype=tf.float32), shape=[self._fft_size, 1, 1]) * tf.expand_dims(basis_cov, axis=0) / self._fft_size_f)) # out shape = [fft_size=nb_possible_ccsk_shift, fft_size, fft_size]

        self._w_n0 = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.),
            name='w_n0',
            trainable=False,
            dtype=tf.float32
        )

        self._w_covariance_matrix = self.add_weight(
            shape=(resource_grid.fft_size, resource_grid.fft_size),
            initializer=ComplexIdentity(),
            name='w_covariance_matrix',
            trainable=False,
            dtype=tf.complex64
        )

        self._w_covariance_tensor_block = self.add_weight(
            shape=(resource_grid.fft_size, resource_grid.fft_size, resource_grid.fft_size),
            initializer=tf.zeros_initializer,
            name='w_covariance_tensor_block',
            trainable=False,
            dtype=tf.complex64
        )

    @property
    def w_n0(self):
        return self._w_n0

    @property
    def w_covariance_matrix(self):
        return self._w_covariance_matrix

    @property
    def w_covariance_tensor_block(self):
        return self._w_covariance_tensor_block
    
    @w_n0.setter
    def w_n0(self, new_weight):
        self._w_n0.assign(tf.constant(new_weight, shape=(1,)))
        self._update_covariance_block_

    @w_covariance_matrix.setter
    def w_covariance_matrix(self, new_weight):
        self._w_covariance_matrix.assign(new_weight)
        self._compute_covariance_block_
    
    @w_covariance_tensor_block.setter
    def w_covariance_tensor_block(self, new_weight):
        self._w_covariance_tensor_block.assign(new_weight)
    
    @property
    def _compute_covariance_block_(self):
        updated_covmatrix_tensor = self._ccsk_cov * tf.expand_dims(self._w_covariance_matrix, axis=0) # out shape = [fft_size=nb_possible_ccsk_shift, fft_size, fft_size]

        self._eigenvalues, self._unitary_matrix = tf.linalg.eigh(updated_covmatrix_tensor)  # out shape = [fft_size=nb_possible_ccsk_shift, fft_size] || [fft_size=nb_possible_ccsk_shift, fft_size, fft_size]
        self._eigenvalues = tf.math.real(self._eigenvalues)  # out shape = [fft_size=nb_possible_ccsk_shift, fft_size]
        tmp = self._unitary_matrix @ tf.cast(tf.linalg.diag(tf.math.divide_no_nan(self._eigenvalues, self._eigenvalues + self._w_n0)), dtype=tf.complex64) @ tf.linalg.adjoint(self._unitary_matrix) # out shape = [fft_size=nb_possible_ccsk_shift, fft_size, fft_size]
        
        self._w_covariance_tensor_block.assign(tmp) # out shape = [fft_size=nb_possible_ccsk_shift, fft_size, fft_size]

    @property
    def _update_covariance_block_(self):
        tmp = self._unitary_matrix @ tf.cast(tf.linalg.diag(tf.math.divide_no_nan(self._eigenvalues, self._eigenvalues + self._w_n0)), dtype=tf.complex64) @ tf.linalg.adjoint(self._unitary_matrix)  # out shape = [fft_size=nb_possible_ccsk_shift, fft_size, fft_size]
        self._w_covariance_tensor_block.assign(tmp) # out shape = [fft_size=nb_possible_ccsk_shift, fft_size, fft_size]

    @property
    def _compute_ifft_shift_mapping(self):
        a = np.arange(0, self._N)
        b = np.mod(-self._u * a, self._N)
        b = b.astype(np.int32)
        return b

    @property
    def _compute_gather_index_list(self):
        gather_index_list = []
        shift_val_index = 1
        for ofdm_symb_index in np.arange(self._resource_grid.num_ofdm_symbols):
            if ofdm_symb_index in self._resource_grid.pilot_pattern._ofdm_symb_list:
                gather_index_list.append(0)
            else:
                gather_index_list.append(shift_val_index)
                shift_val_index += 1
        return gather_index_list
    

    def call(self, inputs):
        if self._submode == 'Perfect':
            # input shape = [batch_size, num_ofdm_symbols, l_tot] || [batch_size, num_ccsk_per_frame]
            h_time, shift_val = inputs
        else:
            # input shape = [batch_size, num_ofdm_symb, fft_size] || [batch_size, num_ccsk_per_frame]
            # x is the downsampled time signal
            x, shift_val = inputs 
        
        ## TODO: watchout, current implementation fill the provided shift values with references_shift_value at pilot/reference OFDM symbols positions.
        ## This suppose no iterative mapping.
        complete_shift_val = tf.pad(shift_val, paddings=[[0,0],[1,0]], constant_values=self._references_shift_value) # out shape = [batch_size, num_data_symbols + 1]
        complete_shift_val = tf.gather(complete_shift_val, self._gather_index_list, axis=-1) # out shape = [batch_size, num_ofdm_symbols]

        # Compute the shifted CIR (in time)
        if self._submode != 'Perfect':
            h_time_est = cross_correlation(x, tf.expand_dims(self._ifft_zc0, axis=1), axis=-1, normalized=True, out_operator=None) # out shape = [batch_size, num_ofdm_symb, fft_size]
                   
        else:
            # Get the corresponding IDFT shift value, 
            ifft_shift_val = tf.gather(self._ifft_shift_mapping, indices=complete_shift_val, axis=-1) # out shape = [batch_size, num_ofdm_symbols]
            
            # Create a one hot vector corresponding to the IDFT shift value
            one_hot_ifft_shift_val = tf.roll(tf.reverse(tf.one_hot(indices=ifft_shift_val, depth=self._N, axis=None, dtype=tf.complex64), axis=[-1]), shift=[1], axis=[-1]) # out shape = [batch_size, num_ofdm_symbols, fft_size]
            
            # Compute the cross-correlation to right-shift the padded CIR by the expected amount
            # With oversampling, to compute the exact channel at sample time, with downsample in frequency.
            h_time_lag = tf.roll(tf.pad(h_time, [[0,0],[0,0],[0,self._fft_size * self._oversampling - self._l_tot]]), shift=self._l_min, axis=-1) # out shape = [batch_size, num_ofdm_symbols, fft_size * oversampling]
            h_freq_true = tf.signal.fft(h_time_lag) # out shape = [batch_size, num_ofdm_symbols, fft_size * oversampling]
            h_freq_true = tf.concat([h_freq_true[:,:,:self._N//2], h_freq_true[:,:,-self._N//2:]], axis=-1) # out shape = [batch_size, num_ofdm_symbols, fft_size]
            h_time_lag = tf.signal.ifft(h_freq_true) # out shape = [batch_size, num_ofdm_symbols, fft_size]
            h_time_lag = cross_correlation(h_time_lag, one_hot_ifft_shift_val, out_operator=None) # out shape = [batch_size, num_ofdm_symbols, fft_size]
            
            # Add scaling and constant phase shift based on the CCSK and shift value
            h_time_est = h_time_lag * self._sqrt_fft_size_complex * tf.expand_dims(tf.gather(self._root_sequence, indices=complete_shift_val, axis=-1), axis=-1) # out shape = [batch_size, num_ofdm_symb, fft_size]

        # Compute the channel estimate in frequency
        h_freq_est = tf.signal.fft(h_time_est) # out shape = [batch_size, num_ofdm_symb, fft_size]
        
        ## For equivalent LS estimation, one can also compute directly the channel estimate from the time signal x
        # h_freq_est = tf.signal.fft(x) * tf.math.conj(self._root_sequence)[tf.newaxis,tf.newaxis,:]
                
        if self.LMMSE_TrueP_mode:
            selected_covariance_block = tf.gather(self._w_covariance_tensor_block, indices=complete_shift_val, axis=0, batch_dims=0) # out shape = [batch_size, num_ofdm_symb, fft_size, fft_size]
            h_freq_est = tf.transpose(tf.matmul(selected_covariance_block, tf.expand_dims(h_freq_est, axis=-1)), perm=[0,1,3,2])[:,:,0,:] # out shape = [batch_size, num_ofdm_symb, fft_size]
    	
        if self._time_est_output:
            return h_freq_est, h_time_est
        else:
            return h_freq_est