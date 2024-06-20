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

from initializers import ComplexIdentity


class LMMSEEstimator(tf.keras.layers.Layer):

    def __init__(self, resource_grid, **kwargs):
        super().__init__(**kwargs)
        self._resource_grid = resource_grid
        self._pilot_pattern = self._resource_grid.pilot_pattern
        self._pilots = self._pilot_pattern.pilots[0,0,:] # output shape = [num_tx, num_txt_ant, num_pilots] => [num_pilots]
        self._removed_nulled_scs = sn.ofdm.RemoveNulledSubcarriers(self._resource_grid)

        # Precompute indices to construct the LS channel estimates matrix
        mask = self._pilot_pattern.mask[0,0,:,:]  # output shape = [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers] => [num_ofdm_symbols, num_effective_subcarriers]
        self._mask_shape = tf.shape(mask)
        ## 1D-array of the index of OFDM symbol for each pilot, ordered first by OFDM symbol and then by subcarrier
        self._pilots_ofdmsymb_index = tf.cast(tf.where(mask == 1)[:,0], dtype=tf.int32)
        ## 1D-array of the index of the subcarrier for each pilot, ordered first by OFDM symbol and then by subcarrier
        self._pilots_subcarrier_index = tf.cast(tf.where(mask == 1)[:,1], dtype=tf.int32)
        ## 2D-array with the pilots OFDM symbol index on the rows and the pilots subcarrier index on the columns
        self._pilots_index = tf.concat([tf.expand_dims(self._pilots_ofdmsymb_index, axis=1), tf.expand_dims(self._pilots_subcarrier_index, axis=1)], axis=1)
        ## Store the unique OFDM symbol index containing at least one symbol and the number of uniquer OFDM symbol index
        self._unique_pilots_ofdmsymb, _, unique_count = tf.unique_with_counts(self._pilots_ofdmsymb_index)
        self._nb_unique_pilots_ofdmsymb = tf.size(self._unique_pilots_ofdmsymb)
        ## Store the total number of pilots within a frame
        self._nb_pilots_symbols = resource_grid.pilot_pattern.num_pilot_symbols
        ## In case of comb-pilot type, the LMMSE interpolates the channel in frequency. The pilot mask should be updated and provided to the time interpolator.
        updated_mask = tf.tensor_scatter_nd_update(tf.zeros(shape=[self._mask_shape[0]], dtype=tf.int32), indices=tf.expand_dims(self._unique_pilots_ofdmsymb, axis=1), updates=tf.ones(shape=[self._nb_unique_pilots_ofdmsymb], dtype=tf.int32))
        updated_mask = tf.tile(tf.expand_dims(updated_mask, axis=1), multiples=[1, resource_grid.num_effective_subcarriers])
        self.updated_mask = updated_mask

        self._nb_pilots_per_ofdmsymb, _ = tf.unique(unique_count)
        self._pilots_subcarrier_position, _ = tf.unique(self._pilots_subcarrier_index)
        if tf.size(self._nb_pilots_per_ofdmsymb) > 1:
            raise ValueError(f'[ERROR][LMMSE Estimator] The number of pilots in OFDM symbols is not constant. Distribution of pilots in OFDM symbols: {unique_count}. Moreover, the subcarrier positions should be the same.')
        elif tf.size(self._pilots_subcarrier_position) > self._nb_pilots_per_ofdmsymb:
            raise ValueError(f'[ERROR][LMMSE Estimator] The subcarrier positions of the pilots within the OFDM symbol is not the same. All subcarrier positions: {self._pilots_subcarrier_position}.')

        self._w_n0 = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.),
            name='w_n0',
            trainable=False,
            dtype=tf.float32
        )

        self._w_covariance_matrix = self.add_weight(
            shape=(resource_grid.num_effective_subcarriers, resource_grid.num_effective_subcarriers),
            initializer=ComplexIdentity(),
            name='w_covariance_matrix',
            trainable=False,
            dtype=tf.complex64
        )

        self._w_covariance_block = self.add_weight(
            shape=(resource_grid.num_effective_subcarriers, self._nb_pilots_per_ofdmsymb[0]),
            initializer=ComplexIdentity(),
            name='w_covariance_block',
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
    def w_covariance_block(self):
        return self._w_covariance_block
    
    @w_n0.setter
    def w_n0(self, new_weight):
        self._w_n0.assign(tf.constant(new_weight, shape=(1,)))
        self._update_covariance_block_

    @w_covariance_matrix.setter
    def w_covariance_matrix(self, new_weight):
        self._w_covariance_matrix.assign(new_weight)
        self._compute_covariance_block_
    
    @w_covariance_block.setter
    def w_covariance_block(self, new_weight):
        self._w_covariance_block.assign(new_weight)
    
    @property
    def _compute_covariance_block_(self):
        # out shape = [1, num_effective_subcarriers, nb_pilots_per_ofdmsymb]

        ## Basic formula using linalg.inv and full covariance matrix
        ## From "IET Signal Processing - 2017 - Savaux - LMMSE channel estimation in OFDM context a review"
        # tmp = self._w_covariance_matrix @ tf.linalg.inv(self._w_covariance_matrix + tf.eye(self._resource_grid.num_effective_subcarriers, dtype=tf.complex64) * tf.cast(self._w_n0, dtype=tf.complex64))
        # self._w_covariance_block.assign(tmp)

        # Block pilots
        if tf.size(self._pilots_subcarrier_position) == self._resource_grid.num_effective_subcarriers:
            ## Formula using the eigenvalue and unitary matrix decomposition of the full covariance matrix
            ## From "IET Signal Processing - 2017 - Savaux - LMMSE channel estimation in OFDM context a review"
            self._eigenvalues, self._unitary_matrix = tf.linalg.eigh(self._w_covariance_matrix)
            self._eigenvalues = tf.math.real(self._eigenvalues)
            tmp = self._unitary_matrix @ tf.cast(tf.linalg.diag(tf.math.divide_no_nan(self._eigenvalues, self._eigenvalues + self._w_n0)), dtype=tf.complex64) @ tf.linalg.adjoint(self._unitary_matrix)
            
            self._w_covariance_block.assign(tmp)

        # Comb pilots
        else:
            ## Formula for frequency interpolation with LMMSE (if needed), using part eigenvalue decomposition with part of the covariance matrix.
            ## From "IET Signal Processing - 2017 - Savaux - LMMSE channel estimation in OFDM context a review"
            tmp_cov_1 = tf.gather(self._w_covariance_matrix, indices=self._pilots_subcarrier_position, axis=1)
            tmp_cov_2 = tf.gather(tmp_cov_1, indices=self._pilots_subcarrier_position, axis=0)
            
            self._eigenvalues, self._unitary_matrix = tf.linalg.eigh(tmp_cov_2)
            tmp = tmp_cov_1 @ self._unitary_matrix @ tf.cast(tf.linalg.diag(tf.math.divide_no_nan(self._eigenvalues, self._eigenvalues + self._w_n0)), dtype=tf.complex64) @ tf.linalg.adjoint(self._unitary_matrix)
            
            self._w_covariance_block.assign(tmp)

    @property
    def _update_covariance_block_(self):
        # out shape = [1, num_effective_subcarriers, nb_pilots_per_ofdmsymb]
        
        # Block pilots
        if tf.size(self._pilots_subcarrier_position) == self._resource_grid.num_effective_subcarriers:
            tmp = self._unitary_matrix @ tf.cast(tf.linalg.diag(tf.math.divide_no_nan(self._eigenvalues, self._eigenvalues + self._w_n0)), dtype=tf.complex64) @ tf.linalg.adjoint(self._unitary_matrix)
            self._w_covariance_block.assign(tmp)

        # Comb pilots
        else:
            tmp = tf.gather(self._w_covariance_matrix, indices=self._pilots_subcarrier_position, axis=1) @ self._unitary_matrix @ tf.cast(tf.linalg.diag(tf.math.divide_no_nan(self._eigenvalues, self._eigenvalues + self._w_n0)), dtype=tf.complex64) @ tf.linalg.adjoint(self._unitary_matrix)
            self._w_covariance_block.assign(tmp)


    def call(self, inputs):
        # input shape = [batch_size, num_pilots]
        h_ls = inputs

        # reshape h_ls to place LS estimates of the same OFDM symbol into the same column
        h_ls = tf.reshape(h_ls, shape=[tf.shape(h_ls)[0], self._nb_unique_pilots_ofdmsymb, self._nb_pilots_per_ofdmsymb[0]]) # out shape = [batch_size, nb_unique_pilots_ofdmsymb, nb_pilots_per_ofdmsymb]

        # out shape = [batch_size, nb_unique_pilots_ofdmsymb, num_effective_subcarriers]
        h_lmmse = tf.transpose(tf.matmul(self._w_covariance_block, h_ls, transpose_b=True), perm=[0,2,1])

        # out shape = [batch_size, num_pilots = nb_unique_pilots_ofdmsymb * num_effective_subcarriers]
        h_lmmse = sn.utils.flatten_last_dims(h_lmmse, num_dims=2)

        outputs = h_lmmse # out shape = [batch_size, num_pilots]

        return outputs


def compute_covmatrix_expdecayPDP(nb_taps, rms_delay_spread, fft_size):
    """
    From "OFDM Channel Estimation by Singular Value Decomposition"
    The RMS delay spread (rms_delay_spread) should be normalized by the sample time as well as the number of taps (nb_taps).
    NB: nb_taps should be higher or equal to the real number of channel taps, e.g. equal to CP length.
    """
    basis_cov = tf.tile(tf.expand_dims(tf.range(start=0, limit=fft_size, delta=1, dtype=tf.float32), axis=1), multiples=[1, fft_size]) - tf.tile(tf.expand_dims(tf.range(start=0, limit=fft_size, delta=1, dtype=tf.float32), axis=0), multiples=[fft_size, 1])
    num = 1. - tf.complex(tf.math.exp(-nb_taps / rms_delay_spread), 0.) * tf.math.exp(tf.complex(0., -2 * np.pi * basis_cov * nb_taps / fft_size))
    den = tf.complex((1. - tf.math.exp(-nb_taps / rms_delay_spread)), 0.) * tf.complex(1., 2 * np.pi * basis_cov * rms_delay_spread / fft_size)

    cov_matrix = num / den # out shape = [fft_size, fft_size]

    return cov_matrix

def compute_covmatrix_BCOMexpdecayPDP(nb_taps, fft_size):
    """
    From "IET Signal Processing - 2017 - Savaux - LMMSE channel estimation in OFDM context a review"
    The maximum delay of the channel (nb_taps) can be equal to : CP_length.
    NB: the maximum delay should correspond to a number of taps higher or equal to the real number of channel taps, e.g. equal to CP length.
    """
    basis_cov = tf.tile(tf.expand_dims(tf.range(start=0, limit=fft_size, delta=1, dtype=tf.float32), axis=1), multiples=[1, fft_size]) - tf.tile(tf.expand_dims(tf.range(start=0, limit=fft_size, delta=1, dtype=tf.float32), axis=0), multiples=[fft_size, 1])
    num = 1. - tf.complex(tf.math.exp(-1.), 0.) * tf.math.exp(tf.complex(0., -2 * np.pi * basis_cov * nb_taps / fft_size))
    den = tf.complex(1. - tf.math.exp(-1.), 0.) * tf.complex(1., 2 * np.pi * basis_cov * nb_taps / fft_size)

    cov_matrix = num / den # out shape = [fft_size, fft_size]

    return cov_matrix

def compute_covmatrix_uniformPDP(nb_taps, fft_size):
    """
    From "OFDM Channel Estimation by Singular Value Decomposition"
    The maximum delay of the channel (nb_taps) can be equal to : CP_length.
    NB: the maximum delay should correspond to a number of taps higher or equal to the real number of channel taps, e.g. equal to CP length.
    """
    basis_cov = tf.tile(tf.expand_dims(tf.range(start=0, limit=fft_size, delta=1, dtype=tf.float32), axis=1), multiples=[1, fft_size]) - tf.tile(tf.expand_dims(tf.range(start=0, limit=fft_size, delta=1, dtype=tf.float32), axis=0), multiples=[fft_size, 1])
    tmp = tf.complex(0., 2 * np.pi * basis_cov * nb_taps / fft_size)
    num = 1. - tf.math.exp(-tmp)
    den = tmp

    cov_matrix = tf.math.divide_no_nan(num, den) + tf.eye(fft_size, dtype=tf.complex64) # out shape = [fft_size, fft_size] ## tf.eye added to correct the limit computation when m=n

    return cov_matrix

def compute_covmatrix_cir(taps_pwr, taps_delay, fft_size):
    """
    From "OFDM Channel Estimation by Singular Value Decomposition"
    The taps power (taps_pwr) should be in linear and normalized, so that the sum = 1.
    The taps delays (taps_delay) should be normalized by the sample time.
    """
    basis_cov = tf.tile(tf.expand_dims(tf.range(start=0, limit=fft_size, delta=1, dtype=tf.float32), axis=1), multiples=[1, fft_size]) - tf.tile(tf.expand_dims(tf.range(start=0, limit=fft_size, delta=1, dtype=tf.float32), axis=0), multiples=[fft_size, 1])
    basis_cov = tf.cast(tf.expand_dims(basis_cov, axis=2), dtype=tf.complex64)
    taps_delay = tf.cast(sn.utils.expand_to_rank(taps_delay, target_rank=3, axis=0), dtype=tf.complex64)
    taps_pwr = tf.cast(sn.utils.expand_to_rank(taps_pwr, target_rank=3, axis=0), dtype=tf.complex64)

    cov_matrix = tf.reduce_sum(taps_pwr * tf.math.exp(-1j * 2 * np.pi * taps_delay * basis_cov / fft_size), axis=2) # out shape = [fft_size, fft_size]

    return cov_matrix

def compute_time_covmatrix_cir(taps_pwr, taps_delay, fft_size, l_min, l_max):
    """
    From "OFDM Channel Estimation by Singular Value Decomposition"
    The taps power (taps_pwr) should be in linear and normalized, so that the sum = 1.
    The taps delays (taps_delay) should be normalized by the sample time.
    """

    # Compute the expectation matrix
    n = tf.expand_dims(tf.range(start=l_min, limit=l_max+1, delta=1, dtype=tf.float32), axis=0)
    n = n - tf.expand_dims(taps_delay, axis=1)
    n = tf.experimental.numpy.sinc(n)
    n = tf.expand_dims(n, axis=1)
    m = tf.transpose(n, perm=[0,2,1])
    mn = m * n
    mn = mn * tf.expand_dims(tf.expand_dims(taps_pwr, axis=1), axis=2)
    mn = tf.reduce_sum(mn, axis=0)

    k0 = tf.expand_dims(tf.range(start=0, limit=fft_size, delta=1, dtype=tf.float32), axis=1) * tf.expand_dims(tf.range(start=0, limit=l_max+1-l_min, delta=1, dtype=tf.float32), axis=0)
    k1 = tf.expand_dims(k0, axis=2)
    k1 = tf.expand_dims(tf.transpose(tf.tile(k1, multiples=[1,1,fft_size]), perm=[0,2,1]), axis=3)
    k0 = tf.expand_dims(k0, axis=1)
    kk = k1 - k0

    r = tf.cast(tf.expand_dims(tf.expand_dims(mn, axis=0), axis=0), dtype=tf.complex64) * tf.exp(tf.complex(0., -2 * np.pi * kk / fft_size))
    r = tf.reduce_sum(tf.reduce_sum(r, axis=-1), axis=-1)

    return r

def compute_covmatrix_channelgenerator(channel_generator, fft_size, batch_size=10_000):
    h_time = channel_generator(batch_size=batch_size)[:,0,:] # out shape = [batch_size, l_tot = l_max - l_min + 1]
    h_freq = np.fft.fft(h_time, n=fft_size, axis=-1, norm="backward")
    h_freq = np.expand_dims(h_freq, axis=-1)
    h_freqH = np.swapaxes(np.conj(h_freq), 1, 2)
    R_hh = np.mean((h_freq @ h_freqH), axis=0)

    R_hh = tf.cast(R_hh, dtype=tf.complex64)

    return R_hh