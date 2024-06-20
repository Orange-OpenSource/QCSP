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

import matplotlib.pyplot as plt
import tensorflow as tf
import sionna as sn
import numpy as np

from scipy.special import j0


def analyze_channel_model(channel_model, save_path='channel_cir.png', symbol_time=None, block_fading_time=None, l_list=[], oversampling=1):
    max_doppler = channel_model._max_doppler / (2 * np.pi)
    tf.print(f'Channel Model Max Doppler [Hz]: {max_doppler}')
    tf.print(f'Channel Model Number of Paths: {channel_model.num_clusters}')
    tf.print(f'Channel Model Paths Delays [s]: {channel_model.delays}')
    tf.print(f'Channel Model RMS Delay Spread [s]: {channel_model.delay_spread}')
    tf.print(f'Channel Model Paths Powers [W]: {channel_model.mean_powers} and [dB]: {10 * np.log10(channel_model.mean_powers)}')
    tf.print(f'Channel Model Paths Sum of Powers [W]: {tf.reduce_sum(channel_model.mean_powers)}')
    
    max_excess_delay, mean_excess_delay, rms_delay_spread = compute_channel_delays_metric(path_pwr_db=10*np.log10(channel_model.mean_powers), path_delays_s=channel_model.delays)
    tf.print(f'Computed from Paths Power and Delays:\n Max Excess Delay [s]: {max_excess_delay}\n Mean Excess Delay [s]: {mean_excess_delay}\n RMS Delay Spread [s]: {rms_delay_spread}') 

    coherence_bw_50 = 1 / (5 * rms_delay_spread)
    coherence_bw_90 = 1 / (50 * rms_delay_spread)
    # coherence_time_50 = 9 / (16 * np.pi * max_doppler)
    coherence_time_50 = 1.51 / (2 * np.pi * max_doppler) ## definition from the Rayleigh autocorrelation Clarke's model
    tf.print(f'Approximated 50% Channel Coherence Bandwidth [kHz]: {coherence_bw_50 / 1e3}')
    tf.print(f'Approximated 90% Channel Coherence Bandwidth [kHz]: {coherence_bw_90 / 1e3}')
    tf.print(f'Approximated 50% Channel Coherence Time [ms]: {coherence_time_50 * 1e3}')
    
    if block_fading_time is not None:
        tf.print(f'Estimated Channel Correlation between 2 Block Fading Generation [%]: {j0(channel_model._max_doppler * block_fading_time) * 100}')

    if symbol_time is not None:
        path_pwr_lin = channel_model.mean_powers / tf.reduce_sum(channel_model.mean_powers)
        path_delays_normalized = channel_model.delays / symbol_time

        if len(l_list) == 0:
            l_min, l_max = sn.channel.utils.time_lag_discrete_time_channel(bandwidth=oversampling/symbol_time, maximum_delay_spread=max_excess_delay)
            tf.print(f'Computed l_min and l_max values in sample time are: l_min = {l_min} and l_max = {l_max}')
        else:
            l_min, l_max = l_list
        
        l_tot = l_max - l_min + 1
        tf.print(f'Total length of the (oversampled) CIR (oversampled CP should be longer than this value): {l_tot}')

        channel_generator = sn.channel.GenerateTimeChannel(channel_model=channel_model, bandwidth=oversampling/symbol_time, num_time_samples=1, l_min=l_min, l_max=l_max, normalize_channel=False)

        h_t = channel_generator(batch_size=10_000)
        h_pwr = tf.reduce_mean(tf.square(tf.abs(h_t[:,0,0,0,0,0,:])), axis=0)

        threshold = 90
        opt_l_min, opt_l_max = compute_fir_size(h_pwr, l_min, threshold)
        ratio = (tf.reduce_sum(h_pwr[-l_min + opt_l_min : -l_min + opt_l_max + 1]) / tf.reduce_sum(h_pwr)) * 100
        tf.print(f'l_min and l_max values in sample time containing over {ratio}% of the power are: [{opt_l_min},{opt_l_max}]')
        threshold = 99
        opt_l_min, opt_l_max = compute_fir_size(h_pwr, l_min, threshold)
        ratio = (tf.reduce_sum(h_pwr[-l_min + opt_l_min : -l_min + opt_l_max + 1]) / tf.reduce_sum(h_pwr)) * 100
        tf.print(f'l_min and l_max values containing over {ratio}% of the power are: [{opt_l_min},{opt_l_max}]')
        threshold = 99.9
        opt_l_min, opt_l_max = compute_fir_size(h_pwr, l_min, threshold)
        ratio = (tf.reduce_sum(h_pwr[-l_min + opt_l_min : -l_min + opt_l_max + 1]) / tf.reduce_sum(h_pwr)) * 100
        tf.print(f'l_min and l_max values containing over {ratio}% of the power are: [{opt_l_min},{opt_l_max}]')

        time_index = tf.range(l_min, l_max + 1, dtype=tf.int32) / oversampling

        plt.figure(figsize=(10,7))
        plt.vlines(path_delays_normalized, 0., path_pwr_lin, colors='r', linewidth=2, label='Power Delay Profile')
        plt.plot(time_index, h_pwr, 'bo-', label='Evaluated Mean Path Power')
        plt.xlabel("Time [symbol time]")
        plt.ylabel("Normalized CIR power [lin]")
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path, format='png', dpi=300, facecolor='w', transparent=False)

        mean_channel_pwr = tf.reduce_mean(tf.reduce_sum(tf.square(tf.abs(h_t[:,0,0,0,0,0,:])), axis=1))
        print(f'Mean Channel Power [W]: {mean_channel_pwr}')

def compute_channel_delays_metric(path_pwr_db, path_delays_s):
    """Compute Max Excess Delay, Mean Excess Delay and RMS Delay Spread from a channel delay-power profile.

    Parameters
    ----------
    path_pwr_db : 1D tensor of tf.float32
        Path powers in dB
    path_delays_s : 1D tensor of tf.float32
        Path delays in seconds (not normalized by RMS Delay Spread)

    Returns
    -------
    Max Excess Delay, Mean Excess Delay and RMS Delay Spread as tf.float32
    """
    
    max_excess_delay = path_delays_s[-1] - path_delays_s[0]

    path_pwr_lin = tf.pow(10.0, path_pwr_db/10.0)
    norm_factor = tf.reduce_sum(path_pwr_lin)
    mean_excess_delay = tf.reduce_sum(path_delays_s * path_pwr_lin) / norm_factor

    mean_square_delay = tf.reduce_sum((path_delays_s**2) * path_pwr_lin) / norm_factor
    rms_delay_spread = tf.math.sqrt(mean_square_delay - (mean_excess_delay**2))

    return max_excess_delay, mean_excess_delay, rms_delay_spread

def compute_fir_size(channel_fir_pwr, l_min, threshold=99):    
    tot_pwr = tf.reduce_sum(channel_fir_pwr)
    threshold = tot_pwr * threshold / 100
    num_taps = tf.size(channel_fir_pwr)

    cum_pwr = channel_fir_pwr[-l_min]
    if cum_pwr >= threshold:
        return 0, 0

    opt_l_min = 0
    opt_l_max = 0
    back_pwr_tensor = tf.concat([tf.reverse(channel_fir_pwr[0:-l_min], [-1]), tf.zeros(shape=[1], dtype=tf.float32)], axis=-1)
    forward_pwr_tensor = tf.concat([channel_fir_pwr[-l_min + 1:], tf.zeros(shape=[1], dtype=tf.float32)], axis=-1)
    for _ in tf.range(num_taps):
        back_pwr = back_pwr_tensor[opt_l_min]
        forward_pwr = forward_pwr_tensor[opt_l_max]
        
        if back_pwr >= forward_pwr:
            cum_pwr += back_pwr
            if opt_l_min != tf.size(back_pwr_tensor):
                opt_l_min += 1
        else:
            cum_pwr += forward_pwr
            if opt_l_max != tf.size(forward_pwr_tensor):
                opt_l_max += 1
        
        if cum_pwr >= threshold:
            return -opt_l_min, opt_l_max

    return -opt_l_min, opt_l_max

def compute_sionna_cir(channel_model, taps_sampling_frequency, l_min, l_max):   
    h = tf.cast(channel_model.mean_powers, dtype=tf.float64)
    tau = tf.cast(channel_model.delays * taps_sampling_frequency, dtype=tf.float64)
    l = tf.range(l_min, l_max+1, dtype=tf.float64)
    
    tau = tf.expand_dims(tau, axis=-1)
    l = sn.utils.expand_to_rank(l, tau.shape.rank, axis=0)
    
    g = tf.experimental.numpy.sinc(l-tau)**2
    h = tf.expand_dims(h, axis=-1)

    cir = tf.cast(tf.reduce_sum(h*g, axis=0), dtype=tf.float32)
    cir_pwr = tf.reduce_sum(cir)
    
    return cir, cir_pwr
