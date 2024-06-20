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

# Imports
import os
import json
import numpy as np
import tensorflow as tf


# tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices([gpus[0]],"GPU")
NB_GPUS = len(gpus)
EAGER = False


import sionna as sn
import pandas as pd
import matplotlib.pyplot as plt


from datasets import random_messages_dataset
from callbacks import BatchTerminationCallback
from metrics import BitErrorRate, BlockErrorRate, BinomialProportionConfidenceInterval, BlockErrorCount, SymbolErrorRate, BinomialProportionConfidenceInterval_SER, ShiftBlockErrorRate
from ccsk import CCSKModulator, CCSKDemapper, zadoff_chu_sequence, cross_correlation, IterativeMapper, IterativeDemapper, evaluate_analytic_ccsk
from equalizers import ZeroForcingOneTapEqualizer, BiasedLMMSEOneTapEqualizer, UnbiasedLMMSEOneTapEqualizer, MatchedFilterOneTapEqualizer, IdentityEqualizer, DFTLinkEqualizer, MaximumRationCombiningEqualizer
from channel_estimators import LeastSquareEstimator, LMMSEEstimator, TimeChannelEstimation, compute_covmatrix_BCOMexpdecayPDP, compute_covmatrix_cir, compute_covmatrix_expdecayPDP, compute_covmatrix_uniformPDP, compute_time_covmatrix_cir, compute_covmatrix_channelgenerator
from channel_interpolators import PieceWiseConstantInterpolator, LinearInterpolator, SplineCubicInterpolator, AverageInterpolator, SecondOrderInterpolator
from channels import CustomTDL, CustomGenerateTimeChannel, CustomApplyTimeChannel, analyze_channel_model, compute_sionna_cir
from pilots import RegularPilots
from demodulators import DFTLinkDemodulator


# Local Models
class OFDM_CCSK_TC_Model(tf.keras.Model):
    
    def __init__(self, name, resource_grid, root_sequence, num_bits_per_ccsk_sequence, num_ccsk_sequence_per_frame, ccsk_scheme, add_awgn, channel_model, 
                 channel_estimator, channel_interpolator, channel_equalizator, ccsk_demodulator='Standard', l_min=0, l_max=1, u=1, oversampling=4, **kwargs):

        super().__init__(name=name)
        
        assert (ccsk_scheme == 'A') or (ccsk_scheme == 'B') or (ccsk_scheme == 'C')

        self._resource_grid = resource_grid
        self._add_awgn = add_awgn
        self._root_sequence = root_sequence
        self._num_ccsk_sequence_per_frame = tf.cast(num_ccsk_sequence_per_frame, dtype=tf.int32)
        self.num_bits_per_ccsk_sequence = num_bits_per_ccsk_sequence
        self._N = tf.size(root_sequence)
        self._Nf = tf.cast(self._N, dtype=tf.float32)
        self._u = u
        self._oversampling = oversampling
        self._sampling_frequency = resource_grid.bandwidth * oversampling
        self._sqrt_fft_size_complex = tf.cast(tf.sqrt(tf.cast(self._resource_grid.fft_size, dtype=tf.float32)), dtype=tf.complex64)
        self._sqrt_oversample_fft_size_complex = tf.cast(tf.sqrt(tf.cast(self._resource_grid.fft_size * oversampling, dtype=tf.float32)), dtype=tf.complex64)
        self._ccsk_scheme = ccsk_scheme
        self._channel_estimator_mode = channel_estimator
        self._channel_equalizator_mode = channel_equalizator
        self._channel_interpolator_mode = channel_interpolator
        self._ccsk_demodulator_mode = ccsk_demodulator

        if channel_model is not None:
            self._add_channel = True
        else:
            self._add_channel = False
            l_min = 0
            l_max = 0

        self._removed_nulled_scs = sn.ofdm.RemoveNulledSubcarriers(resource_grid)

        if self._add_channel == True:
            cir, cir_pwr = compute_sionna_cir(channel_model=channel_model, taps_sampling_frequency=self._sampling_frequency, l_min=l_min, l_max=l_max)

        self._rest = resource_grid.num_time_samples
        self._l_tot = l_max - l_min + 1 # in sample time, so including oversampling
        self._l_min = l_min

        tmp = -2 * np.pi * tf.cast(l_min, tf.float32) * tf.range(resource_grid.fft_size * self._oversampling, dtype=tf.float32) / tf.cast(resource_grid.fft_size * self._oversampling, tf.float32)
        self._phase_compensation = tf.cast(tf.exp(tf.complex(0., tmp)), dtype=tf.complex64)
        self._covmatrix_phase_correction = tf.expand_dims(self._phase_compensation, axis=1) @ tf.linalg.adjoint(tf.expand_dims(self._phase_compensation, axis=1))

        self._ifft_zc0 = tf.expand_dims(sn.signal.ifft(root_sequence), axis=0)
        
        # For CCSK schemes B. In case of channel estimation, it only works with block type pilots.
        if (ccsk_scheme == 'B'):
            num_ccsk_sequence_per_ofdm_symbol = tf.cast(resource_grid.num_effective_subcarriers/self._N, tf.int32)
            upsampled_root_sequence = tf.math.conj(sn.signal.ifft(sn.signal.Upsampling(samples_per_symbol=num_ccsk_sequence_per_ofdm_symbol, axis=0)(root_sequence))) * tf.sqrt(tf.cast(num_ccsk_sequence_per_ofdm_symbol, tf.complex64)) # normalized
            pilots_ofdmsymb_mask = tf.expand_dims(tf.clip_by_value(tf.reduce_sum(resource_grid.pilot_pattern.mask[0,0,:,:], axis=1), clip_value_min=0, clip_value_max=1), axis=1) 
            self._ifft_conj_upsampled_zc0 = tf.where(pilots_ofdmsymb_mask == 1, tf.cast(1., dtype=tf.complex64), upsampled_root_sequence)


        # TX
        self.ccsk_modulator = CCSKModulator(root_sequence=root_sequence, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence, shift_output=True)
        self.iterative_mapper = IterativeMapper(ccsk_sequence_length=self._N, resource_grid=resource_grid)
        self.resource_mapper = sn.ofdm.ResourceGridMapper(self._resource_grid, dtype=tf.complex64)


        # Channel
        if self._add_channel == True:
            self.channel_generator = CustomGenerateTimeChannel(channel_model=channel_model, channel_sampling_frequency=1/resource_grid.ofdm_symbol_duration, data_sampling_frequency=self._sampling_frequency, 
                                                               num_time_samples=resource_grid.num_ofdm_symbols, l_min=l_min, l_max=l_max, cir_pwr=cir_pwr)
            self.channel_applier = CustomApplyTimeChannel(num_time_samples=(resource_grid.fft_size + resource_grid.cyclic_prefix_length)*oversampling, l_tot=self._l_tot, add_awgn=False, dtype=tf.complex64)

        if self._add_awgn == True:
            self.awgn_applier = sn.channel.AWGN(dtype=tf.complex64)


        # RX
        self.downsampler = sn.signal.Downsampling(samples_per_symbol=oversampling, offset=0, num_symbols=None, axis=-1)
        self.iterative_demapper = IterativeDemapper(ccsk_sequence_length=self._N, resource_grid=resource_grid)
        self.ccsk_demapper = CCSKDemapper(root_sequence=root_sequence, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence)

        if self._add_channel == True:
            self._channel_estimator_submode = kwargs.get('channel_estimator_submode', None)

            # Common compute of covariance matrix for LMMSE schemes
            if self._channel_estimator_submode in ['LMMSE', 'LMMSE_TrueP']:
                lmmse_covmatrix_cir = compute_time_covmatrix_cir(taps_pwr=channel_model.mean_powers, taps_delay=channel_model.delays * self._sampling_frequency, fft_size=resource_grid.fft_size * self._oversampling, l_min=l_min, l_max=l_max)
                lmmse_covmatrix_cir = lmmse_covmatrix_cir * self._covmatrix_phase_correction
                ## We compute the covariance matrix for the full oversampled channel and then select the relevant part for the equalization around DC.
                lmmse_covmatrix_cir = tf.concat([lmmse_covmatrix_cir[:,:resource_grid.fft_size//2],lmmse_covmatrix_cir[:,-resource_grid.fft_size//2:]], axis=-1)
                lmmse_covmatrix_cir = tf.concat([lmmse_covmatrix_cir[:resource_grid.fft_size//2,:],lmmse_covmatrix_cir[-resource_grid.fft_size//2:,:]], axis=0)
                lmmse_covmatrix_cir = tf.gather(lmmse_covmatrix_cir, indices=resource_grid.effective_subcarrier_ind, axis=0)
                lmmse_covmatrix_cir = tf.gather(lmmse_covmatrix_cir, indices=resource_grid.effective_subcarrier_ind, axis=1)
            elif self._channel_estimator_submode in ['expPDP', 'expPDP_TrueP']:
                expPDP_covmatrix_cir = compute_covmatrix_expdecayPDP(nb_taps=resource_grid.cyclic_prefix_length, rms_delay_spread=(1/self._sampling_frequency), fft_size=resource_grid.fft_size) # rms_delay_spread = channel_model.delay_spread
                # In the current simulation, the IDFT and DFT operation are done in baseband withoud shifting the DC position.
                # So, in the received signal after the DFT, the DC is at subcarrier index 0. The last subcarrier, index N, is actually at index -1, in the negative frequency.
                # It means that the first and last subcarriers are actually close in the spectrum, and so is the correlation between the channels at those subcarriers.
                # To reflect this design choice in the precomputed covariance matrices, we have to do this (same size blocks):
                # A | B
                # C | D
                ## into:
                # D | C
                # B | A
                expPDP_covmatrix_cir = tf.roll(expPDP_covmatrix_cir, shift=[resource_grid.fft_size//2,resource_grid.fft_size//2], axis=[0,1])
                expPDP_covmatrix_cir = tf.gather(expPDP_covmatrix_cir, indices=resource_grid.effective_subcarrier_ind, axis=0)
                expPDP_covmatrix_cir = tf.gather(expPDP_covmatrix_cir, indices=resource_grid.effective_subcarrier_ind, axis=1)
            elif self._channel_estimator_submode in ['expBCOMPDP', 'expBCOMPDP_TrueP']:
                expBCOMPDP_covmatrix_cir = compute_covmatrix_BCOMexpdecayPDP(nb_taps=resource_grid.cyclic_prefix_length, fft_size=resource_grid.fft_size)
                # Idem
                expBCOMPDP_covmatrix_cir = tf.roll(expBCOMPDP_covmatrix_cir, shift=[resource_grid.fft_size//2,resource_grid.fft_size//2], axis=[0,1])
                expBCOMPDP_covmatrix_cir = tf.gather(expBCOMPDP_covmatrix_cir, indices=resource_grid.effective_subcarrier_ind, axis=0)
                expBCOMPDP_covmatrix_cir = tf.gather(expBCOMPDP_covmatrix_cir, indices=resource_grid.effective_subcarrier_ind, axis=1)
            elif self._channel_estimator_submode in ['uniformPDP', 'uniformPDP_TrueP']:
                uniformPDP_covmatrix_cir = compute_covmatrix_uniformPDP(nb_taps=resource_grid.cyclic_prefix_length, fft_size=resource_grid.fft_size)
                # Idem
                uniformPDP_covmatrix_cir = tf.roll(uniformPDP_covmatrix_cir, shift=[resource_grid.fft_size//2,resource_grid.fft_size//2], axis=[0,1])
                uniformPDP_covmatrix_cir = tf.gather(uniformPDP_covmatrix_cir, indices=resource_grid.effective_subcarrier_ind, axis=0)
                uniformPDP_covmatrix_cir = tf.gather(uniformPDP_covmatrix_cir, indices=resource_grid.effective_subcarrier_ind, axis=1)
            
        
        # Channel estimator
        if channel_estimator == 'Perfect':
            pass
        elif channel_estimator == 'LS':
            self.channel_LS_estimator = LeastSquareEstimator(resource_grid=resource_grid)
        elif channel_estimator == 'LMMSE':
            self._channel_estimator_submode = kwargs.get('channel_estimator_submode', None)
            self.channel_LS_estimator = LeastSquareEstimator(resource_grid=resource_grid)
            self.channel_LMMSE_estimator = LMMSEEstimator(resource_grid=resource_grid)
            if self._channel_estimator_submode == 'LMMSE':
                covariance_matrix = lmmse_covmatrix_cir
            elif self._channel_estimator_submode == 'expPDP':
                covariance_matrix = expPDP_covmatrix_cir
            elif self._channel_estimator_submode == 'expBCOMPDP':
                covariance_matrix = expBCOMPDP_covmatrix_cir
            elif self._channel_estimator_submode == 'uniformPDP':
                covariance_matrix = uniformPDP_covmatrix_cir
            else:
                covariance_matrix = None
            self.channel_LMMSE_estimator.w_covariance_matrix = covariance_matrix
        elif channel_estimator == 'DFT_Link':
            self._channel_estimator_submode = kwargs.get('channel_estimator_submode', 'LS')
            self._time_est_output = kwargs.get('time_est_output', False)
            self._dft_link_references_shift_value = kwargs.get('references_shift_value', 0)
            self.channel_time_estimator = TimeChannelEstimation(resource_grid=resource_grid, root_sequence=root_sequence, time_est_output=self._time_est_output, submode=self._channel_estimator_submode, 
                                                                references_shift_value=self._dft_link_references_shift_value, u=u, oversampling=oversampling, l_tot=self._l_tot, l_min=l_min)
            
            if self._channel_estimator_submode == 'LMMSE_TrueP':
                self.channel_time_estimator.w_covariance_matrix = lmmse_covmatrix_cir
            elif self._channel_estimator_submode == 'expPDP_TrueP':
                self.channel_time_estimator.w_covariance_matrix = expPDP_covmatrix_cir
            elif self._channel_estimator_submode == 'expBCOMPDP_TrueP':
                self.channel_time_estimator.w_covariance_matrix = expBCOMPDP_covmatrix_cir
            elif self._channel_estimator_submode == 'uniformPDP_TrueP':
                self.channel_time_estimator.w_covariance_matrix = uniformPDP_covmatrix_cir

        if channel_estimator == 'LMMSE':
            interpolation_mask = self.channel_LMMSE_estimator.updated_mask
        else:
            interpolation_mask = self._resource_grid.pilot_pattern.mask[0,0,:,:]


        # Channel interpolator
        if channel_interpolator == 'NN1D':
            self.channel_interpolator = PieceWiseConstantInterpolator(resource_grid=resource_grid, interpolation_type='1D', mask=interpolation_mask)
        elif channel_interpolator == 'linear':
            self.channel_interpolator = LinearInterpolator(resource_grid=resource_grid, mask=interpolation_mask)
        elif channel_interpolator == 'second':
            self.channel_interpolator = SecondOrderInterpolator(resource_grid=resource_grid, mask=interpolation_mask)
        elif channel_interpolator == 'cubic':
            self.channel_interpolator = SplineCubicInterpolator(resource_grid=resource_grid, mask=interpolation_mask)
        elif channel_interpolator == 'average': # Not fully functional, average over whole OFDM symbol, no time interpolation
            self.channel_interpolator = AverageInterpolator(resource_grid=resource_grid, mask=interpolation_mask)
        

        # Channel equalizator
        if channel_equalizator == 'ZF':
            self.channel_equalizer = ZeroForcingOneTapEqualizer(resource_grid=resource_grid)
        elif channel_equalizator == 'BMMSE':
            self.channel_equalizer = BiasedLMMSEOneTapEqualizer(resource_grid=resource_grid) 
        elif channel_equalizator == 'UMMSE':    
            self.channel_equalizer = UnbiasedLMMSEOneTapEqualizer(resource_grid=resource_grid) 
        elif channel_equalizator == 'MF':    
            self.channel_equalizer = MatchedFilterOneTapEqualizer(resource_grid=resource_grid)
        elif channel_equalizator == 'Id':
            self.channel_equalizer = IdentityEqualizer(resource_grid=resource_grid)
        elif channel_equalizator == 'MRC':
            self.channel_equalizer = MaximumRationCombiningEqualizer(resource_grid=resource_grid)
        elif channel_equalizator == 'DFT_Link':
            self.dft_link_equalizer = DFTLinkEqualizer(root_sequence=root_sequence, resource_grid=resource_grid, normalization_mode='as_proba')
        

        # CCSK demodulator
        if ccsk_demodulator == 'Standard':
            pass
        elif ccsk_demodulator == 'DFT_Link':
            self._dft_link_reference_index_list = kwargs.get('dft_link_reference_index_list', [0])
            self._dft_link_window_size = kwargs.get('dft_link_window_size', -1)
            self._dft_link_demodulator_submode = kwargs.get('demodulator_submode', 'Standard')
            self._dft_link_demodulator_iterations = kwargs.get('demodulator_iterations', 10)
            self._dft_link_demodulator_symmetry = kwargs.get('demodulator_symmetry', True)
            self.dft_link_demodulator = DFTLinkDemodulator(root_sequence=root_sequence, 
                                                           resource_grid=resource_grid, 
                                                           reference_index_list=self._dft_link_reference_index_list, 
                                                           window_size=self._dft_link_window_size, 
                                                           submode=self._dft_link_demodulator_submode,
                                                           iterations=self._dft_link_demodulator_iterations,
                                                           symmetry=self._dft_link_demodulator_symmetry)
        
        # TensorFlow Variable
        self._w_n0 = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.),
            name='w_n0',
            trainable=False,
            dtype=tf.float32
        )

    @property
    def w_n0(self):
        return self._w_n0
    
    @w_n0.setter
    def w_n0(self, new_weight):
        self._w_n0.assign(tf.constant(new_weight, shape=(1,)))

        if self._channel_estimator_mode == 'LMMSE':
           self.channel_LMMSE_estimator.w_n0 = self._w_n0
        elif self._channel_estimator_mode == 'DFT_Link':
           if self.channel_time_estimator.LMMSE_TrueP_mode:
                self.channel_time_estimator.w_n0 = self._w_n0

    def call(self, inputs):
        x = inputs

        # TX
        x, shift_val = self.ccsk_modulator(x) # out shape = [batch_size, num_ccsk_per_frame * N] | [batch_size, num_ccsk_per_frame]
        ## CCSK Scheme 'C'
        if self._ccsk_scheme == 'C':
            x = tf.reshape(tf.one_hot(indices=shift_val, depth=self._resource_grid.fft_size, dtype=tf.complex64, on_value=tf.complex(tf.sqrt(self._Nf), 0.)), shape=tf.shape(x)) # out shape = [batch_size, num_ofdm_symbols, num_ccsk_per_symbol * N]
        x = self.iterative_mapper(x) # out shape = [batch_size, num_ccsk_per_frame * N]
        x = sn.utils.insert_dims(x, num_dims=2, axis=1) # out shape = [batch_size, num_tx, num_streams_per_tx, num_ccsk_per_frame * N]
        x = self.resource_mapper(x) # out shape = [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]

        ## Custom OFDM Modulation
        # Shift DC subcarrier to first position
        # x = tf.signal.ifftshift(x, axes=-1) # out shape = [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]

        # Compute IFFT along the last dimension
        ## To oversample the signal, zeros are added on both sides of the spectrum or equivalently at the center of the right-side spectrum. 
        ## The resulting time signal after the IFFT is equivalent to applying a gate filter in frequency or a cyclic-sinc filter in time with oversampling.
        x = tf.concat([x[...,0:self._resource_grid.fft_size//2], tf.zeros(shape=tf.concat([tf.shape(x)[0:-1],[self._resource_grid.fft_size * (self._oversampling-1)]], axis=-1), dtype=tf.complex64), x[...,self._resource_grid.fft_size//2:]], axis=-1)
        x = sn.signal.ifft(x, axis=-1) * tf.sqrt(tf.cast(self._oversampling, dtype=tf.complex64)) # out shape = [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size * oversampling]

        # Obtain cyclic prefix
        ## The CP length is provided in symbol time, we need to add the oversampling.
        cp = x[..., (self._resource_grid.fft_size - self._resource_grid.cyclic_prefix_length) * self._oversampling:]
        
        # Prepend cyclic prefix
        x = tf.concat([cp, x], -1) # out shape = [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, (fft_size + cyclic_prefix_length) * oversampling]
        
        # Serialize last two dimensions
        x = x[:,0,0,:,:] # out shape = [batch_size, num_ofdm_symbols, (fft_size + cyclic_prefix_length) * oversampling]


        # Time Channel
        if self._add_channel == True:
            h_time = self.channel_generator(batch_size=tf.shape(inputs)[0]) # out shape = [batch_size, num_ofdm_symbols, l_tot = l_max - l_min + 1]
            x = self.channel_applier([x, h_time]) # out shape = [batch_size, (num_ofdm_symbols * (fft_size + cyclic_prefix_length) * oversampling) + l_tot - 1]
        else:
            h_time = tf.ones(shape=[tf.shape(inputs)[0], self._resource_grid.num_ofdm_symbols, 1], dtype=tf.complex64) # out shape = [batch_size, num_ofdm_symbols, l_tot = 1]
            x = tf.reshape(x, shape=[tf.shape(inputs)[0], -1]) # out shape = [batch_size, num_ofdm_symbols * (fft_size + cyclic_prefix_length) * oversampling]

        # Add AWGN
        if self._add_awgn:
            x = self.awgn_applier([x, self._w_n0]) # out shape = [batch_size, (num_ofdm_symbols * (fft_size + cyclic_prefix_length) * oversampling) + l_tot - 1]


        # RX
        ## Detection + time synchronization
        # Downsampling
        x = self.downsampler(x[:,-self._l_min:]) # out shape = [batch_size, (num_ofdm_symbols * (fft_size + cyclic_prefix_length)) + (l_tot-1)//oversampling residue]

        ## Custom OFDM Demodulation at symbol time
        # Cut last samples that do not fit into an OFDM symbol
        x = x[...,:self._rest] # out shape = [batch_size, num_time_samples = num_ofdm_symbols * (fft_size + cyclic_prefix_length)]

        # Reshape input to separate OFDM symbols
        new_shape = tf.concat([tf.shape(x)[:-1], [self._resource_grid.num_ofdm_symbols], [self._resource_grid.fft_size + self._resource_grid.cyclic_prefix_length]], axis=0)
        x = tf.reshape(x, new_shape) # out shape = [batch_size, num_ofdm_symbols, fft_size + cyclic_prefix_length]

        # Remove cyclic prefix
        x = x[..., self._resource_grid.cyclic_prefix_length:] # out shape = [batch_size, num_ofdm_symbols, fft_size]

        ## CCSK Scheme 'B'
        if self._ccsk_scheme == 'B':
            x = x * sn.utils.expand_to_rank(self._ifft_conj_upsampled_zc0, target_rank=tf.rank(x), axis=0) # out shape = [batch_size, num_ofdm_symbols, fft_size]

        ## Channel estimation in time for the DFT Link algorithm
        if self._channel_estimator_mode == 'DFT_Link':
            if self._channel_estimator_submode == 'Perfect':
                h_est = self.channel_time_estimator([h_time, shift_val])
            else:
                h_est = self.channel_time_estimator([x, shift_val])

        # Compute FFT
        x = sn.signal.fft(x, axis=-1) # out shape = [batch_size, num_ofdm_symbols, fft_size]

        # Shift DC subcarrier to the middle
        # x = tf.signal.fftshift(x, axes=-1)

        # Apply phase shift compensation to all subcarriers
        # x = x * sn.utils.expand_to_rank(self._phase_compensation, tf.rank(x), 0) # out shape = [batch_size, num_ofdm_symbols, fft_size]


        ## Channel Estimation
        if self._channel_estimator_mode == 'Perfect':
            # To get the correct channel, we compute the total FFT over the whole represented spectrum and then select the central sub-carriers of interest.
            h_est = tf.signal.fft(tf.roll(tf.pad(h_time, [[0,0],[0,0],[0,self._resource_grid.fft_size * self._oversampling - self._l_tot]]), shift=self._l_min, axis=-1)) # out shape = [batch_size, num_ofdm_symbols, fft_size * oversampling]
            h_est = tf.concat([h_est[:,:,:self._resource_grid.fft_size//2], h_est[:,:,-self._resource_grid.fft_size//2:]], axis=-1) # out shape = [batch_size, num_ofdm_symbols, fft_size]
        elif self._channel_estimator_mode == 'LS':
            h_est, _ = self.channel_LS_estimator([x, self._w_n0]) # out shape = [batch_size, num_pilot_symb] ; [1, num_pilot_symb]
        elif self._channel_estimator_mode == 'LMMSE':
            h_est, _ = self.channel_LS_estimator([x, self._w_n0]) # out shape = [batch_size, num_pilot_symb] ; [1, num_pilot_symb]
            h_est = self.channel_LMMSE_estimator(h_est) # out shape = [batch_size, num_pilot_symb]


        ## Channel Interpolation
        if self._channel_estimator_mode in ['Perfect','DFT_Link']:
            h_int = self._removed_nulled_scs(h_est) # out shape = [batch_size, num_ofdm_symbols, num_effective_subcarriers]
        elif self._channel_estimator_mode is not None:
            h_int = self.channel_interpolator(h_est) # out shape = [batch_size, num_ofdm_symbols, num_effective_subcarriers]


        ## Channel Equalization
        if self._channel_equalizator_mode == 'Id':
            x = self.channel_equalizer(x) # out shape = [batch_size, num_ccsk_per_frame * N]
        elif self._channel_equalizator_mode == 'DFT_Link':
            # Currently, output last dim has a dim of fft_size = num_effective_subcarriers = ccsk_length
            # input shape = [batch_size, num_ofdm_symb, fft_size] || [batch_size, num_ofdm_symb, fft_size]
            relative_shift_probability_matrix = self.dft_link_equalizer([x, h_int]) # output shape = [batch_size, num_ofdm_symb, num_ofdm_symb, fft_size]
        else:
            # input shape = [batch_size, num_ofdm_symb, fft_size] ; [batch_size, num_ofdm_symb, num_effective_subcarriers] ; broadcastable to [batch_size, num_ccsk_per_frame * N] as tf.float32
            x, _ = self.channel_equalizer([x, h_int, self._w_n0]) # out shape = [batch_size, num_ccsk_per_frame * N] ; [batch_size, num_ccsk_per_frame * N]


        ## CCSK Demodulation
        if self._ccsk_demodulator_mode == 'Standard':
            x = self.iterative_demapper(x) # out shape = [batch_size, num_ccsk_per_frame * N]
            x = tf.reshape(x, shape=[tf.shape(x)[0], self._num_ccsk_sequence_per_frame, self._N]) # out shape = [batch_size, num_ccsk_per_frame, N]
            if self._ccsk_scheme == 'A':
                if self._channel_equalizator_mode == 'Id':
                    x = cross_correlation(x, self._root_sequence, axis=-1, normalized=False, out_operator='abs') # out shape = [batch_size, num_ccsk_per_frame, N]
                else:
                    x = cross_correlation(x, self._root_sequence, axis=-1, normalized=False, out_operator='real') # out shape = [batch_size, num_ccsk_per_frame, N]
            else:
                x = tf.abs(x) # out shape = [batch_size, num_ccsk_per_frame, N]
            shift_probability = x
            shift_est = tf.math.argmax(shift_probability, axis=-1) # out shape = [batch_size, num_ccsk_per_frame]
            x = self.ccsk_demapper(shift_est) # out shape = [batch_size, num_ccsk_per_frame * num_bits_per_ccsk_sequence]
        elif self._ccsk_demodulator_mode == 'DFT_Link':
            shift_probability = self.dft_link_demodulator(relative_shift_probability_matrix) # out shape = [batch_size, num_ccsk_per_frame, fft_size]
            shift_est = tf.math.argmax(shift_probability, axis=-1, output_type=tf.int32) # out shape = [batch_size, num_ccsk_per_frame]
            x = self.ccsk_demapper(shift_est) # out shape = [batch_size, num_ccsk_per_frame * num_bits_per_ccsk_sequence]
        
        outputs = [x, shift_probability] # out shape = [batch_size, num_ccsk_per_frame * num_bits_per_ccsk_sequence] || [batch_size, num_ccsk_per_frame]

        return outputs

    def test_step(self, data):
        input_bits = data
        output_bits, outputs_shift_probabilities = self(data)
        self.compiled_metrics.update_state(input_bits, output_bits)

        return {m.name: m.result() for m in self.metrics}


# Local Functions
def ber_ci_condition(_, logs):
    if 'ber_confidence_interval' in logs:
        epsilon = 1e-7
        (ci_span, ci_low, ber, ci_high) = logs['ber_confidence_interval']
        return (ci_span)/(ber+epsilon) < 0.01
    else:
        return False

def ser_ci_condition(_, logs):
    if 'ser_confidence_interval' in logs:
        epsilon = 1e-7
        (ci_span, ci_low, ser, ci_high) = logs['ser_confidence_interval']
        return (ci_span)/(ser+epsilon) < 0.01 #or ser < 1e-3
    else:
        return False

def sbler_ci_condition(_, logs):
    if 'sbler_confidence_interval' in logs:
        epsilon = 1e-7
        (ci_span, ci_low, sbler, ci_high) = logs['sbler_confidence_interval']
        return (ci_span)/(sbler+epsilon) < 0.05
    else:
        return False

def evaluate_model(model, eval_eb_n0_db, eval_length, eval_batch_size, num_bits_per_ccsk_sequence, ccsk_sequence_length, coderate, resource_grid, oversampling):
    indices = [[model.name],['ber', 'ser', 'bler', 'sbler'], ['mean', '95%CI']]
    mi = pd.MultiIndex.from_product(indices)
    df = pd.DataFrame(index=eval_eb_n0_db.numpy(), columns=mi)
    df.index.name='Eb/N0'

    # In ebnodb2no for the resource_grid computation
    ## It means that the transmitted time symbols are assume to have energy <= 1.
    ## The objective is to have symbols with energy of 1 at the receiver after the DFT, and adapting the noise power to take that into account
    ## Hence the effective noise bandwidth and data rate does not take into account the null subcarriers
    ## The CP is shortened by the factor num_effective_subcarriers / fft_size to reflect the fact that we do not take into accound the null subcarriers
    eval_n0 = sn.utils.ebnodb2no(eval_eb_n0_db, num_bits_per_symbol=num_bits_per_ccsk_sequence/ccsk_sequence_length, coderate=coderate, resource_grid=resource_grid) #/ tf.cast(oversampling, tf.float32)
    eval_snr_db = 10 * np.log10(1 / eval_n0)
    
    df.insert(loc=0, column='SNR', value=eval_snr_db, allow_duplicates=False)

    # To work with multiple GPUs, the independant first build of the network to obtain the summary should manualy divide the batch_size by number of GPUs
    # Seem to increase the memory occupation of the #0 GPU
    # model(next(random_messages_dataset(length=eval_length, batch=eval_batch_size//NB_GPUS).as_numpy_iterator()))
    # model.summary()
    
    for (eb_n0_db, snr_db, eval_n0) in zip(eval_eb_n0_db, eval_snr_db, eval_n0):
        print(f'\nEvaluating {model.name} at Eb/N0: {eb_n0_db}dB & SNR: {snr_db}dB & N0: {10*sn.utils.log10(eval_n0)}dB\n')

        termination_callback = BatchTerminationCallback(ser_ci_condition)

        input_dataset = random_messages_dataset(length=eval_length, batch=eval_batch_size)
 
        model.w_n0 = eval_n0

        summary = model.evaluate(
            input_dataset,
            steps=1_000_000,
            return_dict=True,
            callbacks=[termination_callback]
        )

        (ber_ci_span, ci_min, ber_ci, ci_max) = summary['ber_confidence_interval']
        (ser_ci_span, ci_min, ser_ci, ci_max) = summary['ser_confidence_interval']
        (bler_ci_span, ci_min, bler_ci, ci_max) = summary['bler_confidence_interval']
        (sbler_ci_span, ci_min, sbler_ci, ci_max) = summary['sbler_confidence_interval']
        ber = summary['ber']
        ser = summary['ser']
        bler = summary['bler']
        sbler = summary['sbler']
        print(f'ber: {ber}/{ber_ci} 95% CI: {ber_ci_span} || ser: {ser}/{ser_ci} 95% CI: {ser_ci_span} || bler: {bler}/{bler_ci} 95% CI: {bler_ci_span} || sbler: {sbler}/{sbler_ci} 95% CI: {sbler_ci_span}')
        df.loc[eb_n0_db.numpy(), (model.name)] = [ber, ber_ci_span, ser, ser_ci_span, bler, bler_ci_span, sbler, sbler_ci_span]
        print(f"Eb/N0: {eb_n0_db}dB (SNR: {snr_db}dB) BER: {ber} SER: {ser} BLER: {bler} SBLER: {sbler}")
    
    # store results and model
    return df

def reset_metrics(metrics_list):
    for metric in metrics_list:
        metric.reset_state()



def main():
    # Global Simulation Parameters
    ## Results Storage
    for doppler in [0]:
        for pilot_config in ['A']:
            for num_ofdm_symbols_val in [20]:
                for num_bits_per_ccsk_sequence_val in [6]: 
                    results_directory = os.path.join('./', 'study_test/') #+str(doppler)+'/'+pilot_config)
                    os.makedirs(results_directory, exist_ok=True)
                    

                    ## OFDM Setup
                    subcarrier_spacing = 15e3 # Hz
                    num_ofdm_symbols = num_ofdm_symbols_val # in the frame
                    num_guard_carriers = [0,0]
                    dc_null = False
                    cp_length = 10
                    oversampling = 4


                    ## Channels
                    file_name = 'EPA.json' # 'EPA.json' || 'EVA.json' || 'ETU.json'
                    model_path = os.path.join('./channels/models', file_name)
                    rms_delay_spread = 400e-9 # sec  ## EPA 43ns || EVA 357ns || ETU 991ns
                    carrier_frequency = 900e6 # Hz
                    doppler_max = doppler # Hz
                    min_speed = (doppler_max * 3e8) / carrier_frequency # m/sec
                    print(f'Minimum device speed used: {min_speed} m/s || {min_speed*3.6} km/h')
                    
                    
                    ## CCSK Modulation
                    num_bits_per_ccsk_sequence = num_bits_per_ccsk_sequence_val
                    u = 1
                    zc_length = int(2**num_bits_per_ccsk_sequence)
                    zc0 = zadoff_chu_sequence(zc_length,u)
                    max_num_ccsk_per_ofdm_symb = 1


                    ## Pilot Pattern
                    with_pilots = True
                    if pilot_config == 'A':
                        ofdm_symb_list = [0] # tf.range(start=0, limit=1, delta=2, dtype=tf.int32).numpy().tolist()
                        # offset_list = (tf.math.mod(tf.range(tf.size(ofdm_symb_list), dtype=tf.int32), 2) * 2).numpy().tolist() # equivalent to [0, 2, 0, 2, 0, ...]
                        offset_list = tf.zeros(len(ofdm_symb_list), dtype=tf.int32).numpy().tolist()
                        subcarrier_step = 1 # Step of the Iterative Mapping will look like num_ccsk_per_ofdm_symb + 1
                        pilots_symbols = zc0

                    elif pilot_config == 'B':
                        ofdm_symb_list = [0,19] # tf.range(start=0, limit=1, delta=2, dtype=tf.int32).numpy().tolist()
                        # offset_list = (tf.math.mod(tf.range(tf.size(ofdm_symb_list), dtype=tf.int32), 2) * 2).numpy().tolist() # equivalent to [0, 2, 0, 2, 0, ...]
                        offset_list = tf.zeros(len(ofdm_symb_list), dtype=tf.int32).numpy().tolist()
                        subcarrier_step = 1 # Step of the Iterative Mapping will look like num_ccsk_per_ofdm_symb + 1
                        pilots_symbols = tf.concat([zc0, zc0], axis=-1)         

                    elif pilot_config == 'C':
                        ofdm_symb_list = [19] # tf.range(start=0, limit=1, delta=2, dtype=tf.int32).numpy().tolist()
                        # offset_list = (tf.math.mod(tf.range(tf.size(ofdm_symb_list), dtype=tf.int32), 2) * 2).numpy().tolist() # equivalent to [0, 2, 0, 2, 0, ...]
                        offset_list = tf.zeros(len(ofdm_symb_list), dtype=tf.int32).numpy().tolist()
                        subcarrier_step = 1 # Step of the Iterative Mapping will look like num_ccsk_per_ofdm_symb + 1
                        pilots_symbols = zc0

                    elif pilot_config == 'D':
                        ofdm_symb_list = [0,4,8,12,16] # tf.range(start=0, limit=1, delta=2, dtype=tf.int32).numpy().tolist()
                        # offset_list = (tf.math.mod(tf.range(tf.size(ofdm_symb_list), dtype=tf.int32), 2) * 2).numpy().tolist() # equivalent to [0, 2, 0, 2, 0, ...]
                        offset_list = tf.zeros(len(ofdm_symb_list), dtype=tf.int32).numpy().tolist()
                        subcarrier_step = 1 # Step of the Iterative Mapping will look like num_ccsk_per_ofdm_symb + 1
                        pilots_symbols = tf.concat([zc0, zc0, zc0, zc0, zc0], axis=-1)
                    

                    ## Resource Grid Config
                    num_effective_subcarriers = max_num_ccsk_per_ofdm_symb * zc_length # Data + Pilots
                    fft_size = (tf.reduce_sum(tf.convert_to_tensor(num_guard_carriers, dtype=tf.int32)) + num_effective_subcarriers + tf.cast(dc_null, dtype=tf.int32)).numpy().tolist()




                    # Resource Grid Setup
                    pilot_pattern = RegularPilots(num_ofdm_symb=num_ofdm_symbols, num_effective_subcarriers=num_effective_subcarriers, offset_list=offset_list, ofdm_symb_list=ofdm_symb_list, subcarriers_step=subcarrier_step, pilots=pilots_symbols)
                    
                    resource_grid_pilots = sn.ofdm.ResourceGrid(num_ofdm_symbols=num_ofdm_symbols, fft_size=fft_size, subcarrier_spacing=subcarrier_spacing, num_tx=1, num_streams_per_tx=1, cyclic_prefix_length=cp_length,
                                                                num_guard_carriers=num_guard_carriers, dc_null=dc_null, pilot_pattern=pilot_pattern, pilot_ofdm_symbol_indices=None, dtype=tf.complex64)
                    
                    resource_grid_no_pilots = sn.ofdm.ResourceGrid(num_ofdm_symbols=num_ofdm_symbols, fft_size=fft_size, subcarrier_spacing=subcarrier_spacing, num_tx=1, num_streams_per_tx=1, cyclic_prefix_length=cp_length,
                                                                num_guard_carriers=num_guard_carriers, dc_null=dc_null, pilot_pattern=None, pilot_ofdm_symbol_indices=None, dtype=tf.complex64)
                    
                    if with_pilots:
                        resource_grid = resource_grid_pilots
                    else:
                        resource_grid = resource_grid_no_pilots

                    if cp_length != 0:
                        print(f'Nyquist theorem for pilot position in frequency (< 1/4) : {tf.math.floor(0.5 * (fft_size / cp_length))}')
                    print(f'Nyquist theorem for pilot position in time (< 1/4) : {tf.math.floor(0.25 * (1. / ((doppler_max + tf.keras.backend.epsilon()) * resource_grid.ofdm_symbol_duration)))}')

                    fig = resource_grid.show()
                    fig.savefig(f'{results_directory}/resource_grid.png', format='png', dpi=300, facecolor='w', transparent=False)
                    plt.close()

                    assert tf.math.mod(resource_grid.num_data_symbols, zc_length) == 0
                    num_ccsk_per_frame = resource_grid.num_data_symbols / zc_length

                    ## Model Evaluation
                    eval_eb_n0_db = tf.range(start=-10, limit=15, delta=4, dtype=tf.float32)
                    # eval_eb_n0_db = tf.range(start=0, limit=35, delta=2, dtype=tf.float32)
                    # eval_eb_n0_db = tf.constant([25.]) # tf.range(start=8, limit=35, delta=2, dtype=tf.float32)
                    eval_batch_size = 128
                    eval_length = int(num_bits_per_ccsk_sequence * num_ccsk_per_frame)



                    ## Simulation Settings Display
                    tf.print(f'OFDM symbol bandwidth (fft_size) [Hz]: {resource_grid.bandwidth}')


                    # Channel and Objects Setup
                    channel_model = CustomTDL(model_path=model_path, delay_spread=rms_delay_spread, carrier_frequency=carrier_frequency, min_speed=min_speed, max_speed=None, normalize_taps=True, dtype=tf.complex64)
                    analyze_channel_model(channel_model, save_path=f'{results_directory}/channel.png', symbol_time=1/resource_grid.bandwidth, block_fading_time=resource_grid.ofdm_symbol_duration, l_list=[], oversampling=oversampling)

                    
                    # Model Setup
                    # channel_estimator
                        # perfect 
                        # LS
                        # LMMSE
                        # DFT_Link
                    # channel_interpolator
                        # NN1D
                        # linear
                        # second
                        # cubic
                        # average
                    # channel_equalizator
                        # ZF
                        # BMMSE
                        # UMMSE
                        # MF
                        # DFT_Link
                    # LRA_LMMSE_mode 
                        # expPDP
                        # expBCOMPDP
                        # uniformPDP
                    
                    ## Model Names
                    TC_perfectCSI_ZF_model_name = f'perfectCSI-ZF'
                    TC_perfectCSI_MRC_model_name = f'perfectCSI-MRC-{num_ofdm_symbols}'
                    CIR_TC_C_perfectCSI_ZF_model_name = f'CCSKC_perfectCSI_ZF_N{zc_length}_q{max_num_ccsk_per_ofdm_symb}'
                    
                    TC_A_Id_model_name = f'No-Equalization-Scheme-A'
                    CIR_TC_B_Id_model_name = f'CCSKB'
                    TC_C_Id_model_name = f'No-Equalization-Scheme-C'
                    
                    TC_LS_ZF_model_name = f'LS-ZF'
                    TC_LS_MRC_model_name = f'LS-MRC-{num_ofdm_symbols}'
                    CIR_TC_C_LS_ZF_model_name = f'CCSKC_LS_ZF_N{zc_length}_q{max_num_ccsk_per_ofdm_symb}'
                    
                    TC_LMMSE_ZF_model_name = f'LMMSE-ZF'
                    TC_LMMSE_MRC_model_name = f'LMMSE-MRC'
                    
                    TC_LRALMMSEExpBCOM_ZF_model_name = f'LRA-LMMSE-ExpTmax-ZF'
                    TC_LRALMMSEUni_ZF_model_name = f'LRA-LMMSE-Uniform-ZF'
                    TC_LRALMMSEExpBCOM_MRC_model_name = f'LRA-LMMSE-ExpTmax-MRC'
                    TC_LRALMMSEUni_MRC_model_name = f'LRA-LMMSE-Uniform-MRC-{num_ofdm_symbols}'

                    TC_DFTLink_Std_LS_model_name = f'DFTLink-Std-LS-MRC'
                    TC_DFTLink_Std_LRAUniTrueP_model_name = f'DFTLink-Std-LRA-UniformTrueP-MRC'
                    TC_DFTLink_Std_LMMSETrueP_model_name = f'DFTLink-Std-LMMSETrueP-MRC'
                    TC_DFTLink_Std_Perfect_model_name = f'DFTLink-Std-PerfectCSI-MRC'

                    TC_DFTLink_GLAD_LS_model_name = f'DFTLink-GLAD-LS-MRC-{num_ofdm_symbols}'
                    TC_DFTLink_GLAD_LRAUniTrueP_model_name = f'DFTLink-GLAD-LRA-UniformTrueP-MRC'
                    TC_DFTLink_GLAD_LMMSETrueP_model_name = f'DFTLink-GLAD-LMMSETrueP-MRC'
                    TC_DFTLink_GLAD_Perfect_model_name = f'DFTLink-GLAD-PerfectCSI-MRC-Test'


                    
                    l_min = -2
                    l_max = 7

                    channel_interpolator='linear'
                    window_size = -1
                    ite = 5


                    # TC_perfectCSI_ZF_model = OFDM_CCSK_TC_Model(name=TC_perfectCSI_ZF_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                             num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='Perfect', 
                    #                                             channel_interpolator=channel_interpolator, channel_equalizator='ZF', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling)
                    
                    TC_perfectCSI_MRC_model = OFDM_CCSK_TC_Model(name=TC_perfectCSI_MRC_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                                                                num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='Perfect', 
                                                                channel_interpolator=channel_interpolator, channel_equalizator='MRC', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling)

                    # TC_A_Id_model = OFDM_CCSK_TC_Model(name=TC_A_Id_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                    num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator=None, 
                    #                                    channel_interpolator=None, channel_equalizator='Id', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling)

                    # TC_C_Id_model = OFDM_CCSK_TC_Model(name=TC_C_Id_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                    num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='C', add_awgn=True, channel_model=channel_model, channel_estimator=None, 
                    #                                    channel_interpolator=None, channel_equalizator='Id', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling)

                    # TC_LS_ZF_model = OFDM_CCSK_TC_Model(name=TC_LS_ZF_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                     num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='LS', 
                    #                                     channel_interpolator=channel_interpolator, channel_equalizator='ZF', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling)
                    
                    TC_LS_MRC_model = OFDM_CCSK_TC_Model(name=TC_LS_MRC_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                                                        num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='LS', 
                                                        channel_interpolator=channel_interpolator, channel_equalizator='MRC', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling)

                    # TC_LMMSE_ZF_model = OFDM_CCSK_TC_Model(name=TC_LMMSE_ZF_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                        num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='LMMSE', 
                    #                                        channel_interpolator=channel_interpolator, channel_equalizator='ZF', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling, channel_estimator_submode='LMMSE')
                    
                    # TC_LMMSE_MRC_model = OFDM_CCSK_TC_Model(name=TC_LMMSE_MRC_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                         num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='LMMSE', 
                    #                                         channel_interpolator=channel_interpolator, channel_equalizator='MRC', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling, channel_estimator_submode='LMMSE')

                    # TC_LRALMMSEUni_ZF_model = OFDM_CCSK_TC_Model(name=TC_LRALMMSEUni_ZF_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                              num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='LMMSE', 
                    #                                              channel_interpolator=channel_interpolator, channel_equalizator='ZF', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling, channel_estimator_submode='uniformPDP')
                    
                    TC_LRALMMSEUni_MRC_model = OFDM_CCSK_TC_Model(name=TC_LRALMMSEUni_MRC_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                                                                num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='LMMSE', 
                                                                channel_interpolator=channel_interpolator, channel_equalizator='MRC', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling, channel_estimator_submode='uniformPDP')

                    # TC_LRALMMSEExp_ZF_model = OFDM_CCSK_TC_Model(name=TC_LRALMMSEExpBCOM_ZF_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                              num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='LMMSE', 
                    #                                              channel_interpolator=channel_interpolator, channel_equalizator='ZF', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling, channel_estimator_submode='expBCOMPDP')
                    
                    # TC_LRALMMSEExp_MRC_model = OFDM_CCSK_TC_Model(name=TC_LRALMMSEExpBCOM_MRC_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                             num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='LMMSE', 
                    #                                             channel_interpolator=channel_interpolator, channel_equalizator='MRC', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling, channel_estimator_submode='expBCOMPDP')


                    # Standard sub-mode
                    # TC_DFTLink_Std_LS_model = OFDM_CCSK_TC_Model(name=TC_DFTLink_Std_LS_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                              num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='DFT_Link', 
                    #                                              channel_interpolator=None, channel_equalizator='DFT_Link', ccsk_demodulator='DFT_Link', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling, 
                    #                                              channel_estimator_submode='LS', time_est_output=False, dft_link_reference_index_list=ofdm_symb_list, dft_link_window_size=window_size, demodulator_submode='Standard')

                    # TC_DFTLink_Std_LRAUniTrueP_model = OFDM_CCSK_TC_Model(name=TC_DFTLink_Std_LRAUniTrueP_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                                       num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='DFT_Link', 
                    #                                                       channel_interpolator=None, channel_equalizator='DFT_Link', ccsk_demodulator='DFT_Link', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling,
                    #                                                       channel_estimator_submode='uniformPDP_TrueP', time_est_output=False, dft_link_reference_index_list=ofdm_symb_list, dft_link_window_size=window_size, demodulator_submode='Standard',
                    #                                                       references_shift_value=0)
                    
                    # TC_DFTLink_Std_LMMSETrueP_model = OFDM_CCSK_TC_Model(name=TC_DFTLink_Std_LMMSETrueP_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                                      num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='DFT_Link', 
                    #                                                      channel_interpolator=None, channel_equalizator='DFT_Link', ccsk_demodulator='DFT_Link', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling,
                    #                                                      channel_estimator_submode='LMMSE_TrueP', time_est_output=False, dft_link_reference_index_list=ofdm_symb_list, dft_link_window_size=window_size, demodulator_submode='Standard',
                    #                                                      references_shift_value=0)

                    # TC_DFTLink_Std_Perfect_model = OFDM_CCSK_TC_Model(name=TC_DFTLink_Std_Perfect_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                                   num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='DFT_Link', 
                    #                                                   channel_interpolator=None, channel_equalizator='DFT_Link', ccsk_demodulator='DFT_Link', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling,
                    #                                                   channel_estimator_submode='Perfect', time_est_output=False, dft_link_reference_index_list=ofdm_symb_list, dft_link_window_size=window_size, demodulator_submode='Standard',
                    #                                                   references_shift_value=0)
                    
                    # BP RShift sub-mode
                    TC_DFTLink_GLAD_LS_model = OFDM_CCSK_TC_Model(name=TC_DFTLink_GLAD_LS_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                                                                num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='DFT_Link', 
                                                                channel_interpolator=None, channel_equalizator='DFT_Link', ccsk_demodulator='DFT_Link', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling, 
                                                                channel_estimator_submode='LS', time_est_output=False, dft_link_reference_index_list=ofdm_symb_list, dft_link_window_size=window_size, demodulator_submode='GLAD', 
                                                                demodulator_iterations=ite)
                    
                    
                    # TC_DFTLink_GLAD_LRAUniTrueP_model = OFDM_CCSK_TC_Model(name=TC_DFTLink_GLAD_LRAUniTrueP_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                                      num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='DFT_Link', 
                    #                                                      channel_interpolator=None, channel_equalizator='DFT_Link', ccsk_demodulator='DFT_Link', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling,
                    #                                                      channel_estimator_submode='uniformPDP_TrueP', time_est_output=False, dft_link_reference_index_list=ofdm_symb_list, dft_link_window_size=window_size, demodulator_submode='GLAD', 
                    #                                                      demodulator_iterations=ite, references_shift_value=0)
                    
                    # TC_DFTLink_GLAD_LMMSETrueP_model = OFDM_CCSK_TC_Model(name=TC_DFTLink_GLAD_LMMSETrueP_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                                     num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='DFT_Link', 
                    #                                                     channel_interpolator=None, channel_equalizator='DFT_Link', ccsk_demodulator='DFT_Link', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling,
                    #                                                     channel_estimator_submode='LMMSE_TrueP', time_est_output=False, dft_link_reference_index_list=ofdm_symb_list, dft_link_window_size=window_size, demodulator_submode='GLAD', 
                    #                                                     demodulator_iterations=ite, references_shift_value=0)

                    # TC_DFTLink_GLAD_Perfect_model = OFDM_CCSK_TC_Model(name=TC_DFTLink_GLAD_Perfect_model_name, resource_grid=resource_grid, root_sequence=zc0, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence,
                    #                                          num_ccsk_sequence_per_frame=num_ccsk_per_frame, ccsk_scheme='A', add_awgn=True, channel_model=channel_model, channel_estimator='DFT_Link', 
                    #                                          channel_interpolator=None, channel_equalizator='DFT_Link', ccsk_demodulator='DFT_Link', l_min=l_min, l_max=l_max, u=u, oversampling=oversampling,
                    #                                          channel_estimator_submode='Perfect', time_est_output=False, dft_link_reference_index_list=ofdm_symb_list, dft_link_window_size=window_size, demodulator_submode='GLAD', 
                    #                                          demodulator_iterations=ite, references_shift_value=0)


                    model_list = [TC_DFTLink_GLAD_LS_model, TC_perfectCSI_MRC_model, TC_LS_MRC_model, TC_LRALMMSEUni_MRC_model]


                    config_dict = {'subcarrier_spacing': subcarrier_spacing, 
                                'num_ofdm_symbols': num_ofdm_symbols,
                                'num_guard_carriers': num_guard_carriers,
                                'dc_null': dc_null,
                                'cp_length': cp_length,
                                'oversampling': oversampling,
                                'num_bits_per_ccsk_sequence': num_bits_per_ccsk_sequence,
                                'u': u,
                                'N': zc_length,
                                'max_num_ccsk_per_ofdm_symb': max_num_ccsk_per_ofdm_symb,
                                'with_pilots': with_pilots,
                                'ofdm_symb_list': ofdm_symb_list,
                                'offset_list': offset_list,
                                'subcarrier_step': subcarrier_step,
                                'num_effective_subcarriers': num_effective_subcarriers,
                                'fft_size': fft_size,
                                'channel_file_name': file_name,
                                'rms_delay_spread': rms_delay_spread,
                                'carrier_frequency': carrier_frequency,
                                'doppler_max': doppler_max,
                                'min_speed': min_speed,
                                'l_min': l_min,
                                'l_max': l_max,
                                'window_size': window_size
                                }

                    with open(f'{results_directory}/simulations_config.json', 'w') as f:
                        json.dump(config_dict, f, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=False, cls=None, indent=4, separators=(', ', ': '), default=None, sort_keys=False)
                    

                    # Evaluation Setup
                    metrics=[
                        BitErrorRate(from_logits=False, name="ber"),
                        SymbolErrorRate(num_bits_per_symbol=num_bits_per_ccsk_sequence, name='ser', from_logits=False),
                        BlockErrorRate(name='bler', from_logits=False),
                        ShiftBlockErrorRate(num_bits_per_symbol=num_bits_per_ccsk_sequence, name='sbler', from_logits=False),
                        BinomialProportionConfidenceInterval(
                            monitor_class=BitErrorRate, monitor_params={'from_logits':False},
                            fraction=0.95, dimensions=None, name='ber_confidence_interval'
                        ),
                        BinomialProportionConfidenceInterval_SER(
                            monitor_class=SymbolErrorRate, num_bits_per_symbol=num_bits_per_ccsk_sequence,
                            monitor_params={'num_bits_per_symbol':num_bits_per_ccsk_sequence, 'from_logits':False},
                            fraction=0.95, name='ser_confidence_interval'
                        ),
                        BinomialProportionConfidenceInterval(
                            monitor_class=BlockErrorRate, monitor_params={'from_logits':False},
                            fraction=0.95, dimensions=[0], name='bler_confidence_interval'
                        ),
                        BinomialProportionConfidenceInterval(
                            monitor_class=ShiftBlockErrorRate, monitor_params={'num_bits_per_symbol':num_bits_per_ccsk_sequence, 'from_logits':False},
                            fraction=0.95, dimensions=[0], name='sbler_confidence_interval'
                        ),
                    ]

                    for model in model_list:
                        model.compile(metrics=metrics, run_eagerly=EAGER)
                
                        # Evaluation
                        evaluation_results = evaluate_model(model=model, eval_eb_n0_db=eval_eb_n0_db, eval_length=eval_length, eval_batch_size=eval_batch_size, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence, 
                                                            ccsk_sequence_length=zc_length, coderate=1., resource_grid=resource_grid, oversampling=oversampling)

                        # Saving Results
                        results_file = os.path.join(results_directory, f'results_{model._name}.h5')
                        if os.path.isfile(results_file):
                            print(f'\nFile already exists! Removing previous results file {results_file}\n')
                            os.remove(results_file)

                        evaluation_results.to_hdf(results_file, key='results')

                        reset_metrics(metrics)


                    # Analytical Evaluation
                    analytical_model_name = 'CCSK-ZC64-Analytic-AWGN'
                    compute_analytics = False

                    if compute_analytics == True:
                        analytical_results = evaluate_analytic_ccsk(model_name=analytical_model_name, eval_eb_n0_db=eval_eb_n0_db, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence, 
                                                                    ccsk_sequence_length=zc_length, coderate=1., resource_grid=resource_grid)

                        ## Saving Results
                        results_file = os.path.join(results_directory, f'results_{analytical_model_name}.h5')
                        if os.path.isfile(results_file):
                            print(f'\nFile already exists! Removing previous results file {results_file}\n')
                            os.remove(results_file)

                        analytical_results.to_hdf(results_file, key='results')


if __name__ == "__main__":
    main()
