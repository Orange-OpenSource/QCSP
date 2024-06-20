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
import numpy as np
import sionna as sn
import pandas as pd

from scipy.stats import norm



def zadoff_chu_sequence(N, u):
    assert N > u, 'N should be higher than u'
    assert tf.experimental.numpy.gcd(N,u) == 1, 'The GCD of N and u should be 1'

    n = tf.range(start=0, limit=N, delta=1, dtype=tf.float32)
    N = tf.cast(N, tf.float32)
    u = tf.cast(u, tf.float32)
    sequence = tf.cast(tf.math.exp(-1j * tf.cast((np.pi * u * n * (n + tf.math.mod(N,2))) / N, tf.complex128)), tf.complex64)
    sequence_r = tf.math.real(sequence)
    sequence_r = tf.where(tf.abs(sequence_r) < 1e-4, tf.constant(0., tf.float32), sequence_r)
    sequence_i = tf.math.imag(sequence)
    sequence_i = tf.where(tf.abs(sequence_i) < 1e-4, tf.constant(0., tf.float32), sequence_i)
    sequence = tf.complex(sequence_r, sequence_i)

    return sequence


def cross_correlation(seq1, seq2, axis=-1, normalized=False, out_operator='real'):
    N = tf.cast(tf.shape(seq1)[axis], tf.complex64)

    out = sn.signal.fft(sn.signal.ifft(seq1, axis=axis) * tf.math.conj(sn.signal.ifft(seq2, axis=axis)))

    if normalized:
        out /= tf.sqrt(N)
    
    if out_operator == 'real':
        out = tf.math.real(out)
    elif out_operator == 'imag':
        out = tf.math.imag(out)
    elif out_operator == 'abs':
        out = tf.math.abs(out)
    elif out_operator is None:
        out = out

    return out

def cross_correlation_c(seq1, seq2, out_operator=None):
    out = tf.signal.ifft(tf.signal.fft(seq1) * tf.math.conj(tf.signal.fft(seq2)))

    if out_operator == 'real':
        out = tf.math.real(out)
    elif out_operator == 'abs':
        out = tf.math.abs(out)
    elif out_operator is None:
        out = out

    return out

def cross_correlation_r(seq1, seq2, out_operator=None):
    out = tf.signal.irfft(tf.signal.rfft(seq1) * tf.math.conj(tf.signal.rfft(seq2)))

    if out_operator == 'abs':
        out = tf.math.abs(out)
    elif out_operator is None:
        out = out

    return out


def compute_modular_inverse(u, N):
    span = tf.range(1, N, dtype=tf.float32)
    u_inv = span[tf.cast(tf.where(tf.math.mod(u * span, N) == 1.)[0][0], dtype=tf.int32)]

    return u_inv


def analytic_ccsk_perf_awgn(es_n0_lin, ccsk_sequence_length):
    ser_ccsk_list = []

    def q_function(x): # 0.5 * erfc(x/sqrt(2))
        return norm.sf(x)

    for current_es_n0_lin in es_n0_lin:
        noise = np.random.normal(loc=0.0, scale=np.sqrt(ccsk_sequence_length/current_es_n0_lin/2), size=1000)
        val = np.sqrt(ccsk_sequence_length) + noise
        min_val = np.round(np.min(val),5) - 5
        max_val = np.round(np.max(val),5) + 5
        step = 0.0001
        val_range = np.arange(min_val, max_val+1, step=step)
        ser_ccsk_th_val = step*(1/np.sqrt(2*np.pi))*np.sum((1-(1-q_function(val_range))**(ccsk_sequence_length-1))*np.exp(-((val_range/np.sqrt(2))-np.sqrt(current_es_n0_lin))**2))
        ser_ccsk_list.append(ser_ccsk_th_val)

    ser_ccsk_th = tf.constant(ser_ccsk_list, dtype=tf.float32)
    ser_ccsk_th = tf.where(ser_ccsk_th < 1e-7, tf.constant(0., tf.float32), ser_ccsk_th)

    # print(f"Eb/N0: {eb_n0_db}dB SER: {ser_ccsk_th}")

    return ser_ccsk_th


def evaluate_analytic_ccsk(model_name, eval_eb_n0_db, num_bits_per_ccsk_sequence, ccsk_sequence_length, coderate, resource_grid):
    eb_n0_lin = 10 ** (eval_eb_n0_db/10)
    cp_overhead = resource_grid.cyclic_prefix_length / resource_grid.fft_size
    num_not_null_symbs = resource_grid.num_ofdm_symbols * (1 + cp_overhead) * resource_grid.num_effective_subcarriers
    # eb_n0_lin * num_bits_per_ccsk_sequence = quantity of energy spent on a CCSK sequence if only CCSK sequences are sent
    # Otherwise, this number of bits represents more than the CCSK sequence, because the Eb is overvalued
    # To compute the overvalue ratio, we compute resource_grid.num_data_symbols / num_not_null_symbs
    es_n0_lin = eb_n0_lin * tf.cast(resource_grid.num_data_symbols / num_not_null_symbs, tf.float32) * num_bits_per_ccsk_sequence

    ser_ccsk_th = analytic_ccsk_perf_awgn(es_n0_lin=es_n0_lin, num_bits_per_ccsk_sequence=num_bits_per_ccsk_sequence, ccsk_sequence_length=ccsk_sequence_length)
    
    indices = [[model_name],['ber', 'ser'], ['mean', '95%CI']]
    mi = pd.MultiIndex.from_product(indices)
    df = pd.DataFrame(index=eval_eb_n0_db.numpy(), columns=mi)
    df.index.name='Eb/N0'

    eval_n0 = sn.utils.ebnodb2no(eval_eb_n0_db, num_bits_per_symbol=num_bits_per_ccsk_sequence/ccsk_sequence_length, coderate=coderate, resource_grid=resource_grid)
    eval_snr_db = 10 * np.log10(1 / (eval_n0))
    df.insert(loc=0, column='SNR', value=eval_snr_db, allow_duplicates=False)
    
    for index in range(tf.size(eval_eb_n0_db)):
        df.loc[eval_eb_n0_db[index].numpy(), (model_name)] = [(ser_ccsk_th[index]/2).numpy(), 0., ser_ccsk_th[index].numpy(), 0.]  # [ber, ber_ci_span, bler, bler_ci_span]

    return df
