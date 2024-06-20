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


class IterativeMapper(tf.keras.layers.Layer):

    def __init__(self, ccsk_sequence_length, num_ccsk_sequence_per_ofdm_symbol=4, resource_grid=None, **kwargs):
        super().__init__(**kwargs)

        if resource_grid is None: # Backward compatibility
            temp = tf.range(start=0, limit=num_ccsk_sequence_per_ofdm_symbol*ccsk_sequence_length, delta=1, dtype=tf.int32)
            self._mapping_index = tf.reshape(tf.transpose(tf.reshape(temp, shape=[num_ccsk_sequence_per_ofdm_symbol, ccsk_sequence_length])), shape=[-1])
        
        else: # New design
            num_pilots_per_ofdmsymb = tf.reduce_sum(resource_grid.pilot_pattern.mask[0,0,:,:], axis=1)
            num_data_per_ofdmsymb = resource_grid.num_effective_subcarriers - num_pilots_per_ofdmsymb
            actual_num_ccsk_sequence_per_ofdm_symbol = tf.cast(num_data_per_ofdmsymb / ccsk_sequence_length, dtype=tf.int32)
            
            mapping_list = []
            offset = 0
            for num_ccsk_sequence in actual_num_ccsk_sequence_per_ofdm_symbol:
                if num_ccsk_sequence == 0:
                    continue
                temp = tf.range(start=0, limit=num_ccsk_sequence*ccsk_sequence_length, delta=1, dtype=tf.int32) + offset
                mapping_index = tf.reshape(tf.transpose(tf.reshape(temp, shape=[num_ccsk_sequence, ccsk_sequence_length])), shape=[-1])
                mapping_list.append(mapping_index)
                offset += num_ccsk_sequence*ccsk_sequence_length
            
            self._mapping_index = tf.concat(mapping_list, axis=0)

    def call(self, inputs):
        """
        Takes as inputs the CCSK symbols to map on an OFDM symbol and apply iterative mapping.
        
        Backward compatibility Input/Output
        input shape = [..., num_ccsk_per_ofmd_symbol * N], tf.complex64
        output shape = [..., num_ccsk_per_ofmd_symbol * N], tf.complex64

        New design Input/Output
        input shape = [..., num_ccsk_per_frame * N], tf.complex64
        output shape = [..., num_ccsk_per_frame * N], tf.complex64
        """
        # Apply iterative mapping per OFDM symbol
        x = tf.gather(inputs, indices=self._mapping_index, axis=-1)

        return x


class IterativeDemapper(tf.keras.layers.Layer):

    def __init__(self, ccsk_sequence_length, num_ccsk_sequence_per_ofdm_symbol=4, resource_grid=None, **kwargs):
        super().__init__(**kwargs)

        if resource_grid is None: # Backward compatibility
            temp = tf.range(start=0, limit=num_ccsk_sequence_per_ofdm_symbol*ccsk_sequence_length, delta=1, dtype=tf.int32)
            self._mapping_index = tf.reshape(tf.transpose(tf.reshape(temp, shape=[ccsk_sequence_length, num_ccsk_sequence_per_ofdm_symbol])), shape=[-1])
        
        else: # New design
            num_pilots_per_ofdmsymb = tf.reduce_sum(resource_grid.pilot_pattern.mask[0,0,:,:], axis=1)
            num_data_per_ofdmsymb = resource_grid.num_effective_subcarriers - num_pilots_per_ofdmsymb
            actual_num_ccsk_sequence_per_ofdm_symbol = tf.cast(num_data_per_ofdmsymb / ccsk_sequence_length, dtype=tf.int32)
            
            mapping_list = []
            offset = 0
            for num_ccsk_sequence in actual_num_ccsk_sequence_per_ofdm_symbol:
                if num_ccsk_sequence == 0:
                    continue
                temp = tf.range(start=0, limit=num_ccsk_sequence*ccsk_sequence_length, delta=1, dtype=tf.int32) + offset
                mapping_index = tf.reshape(tf.transpose(tf.reshape(temp, shape=[ccsk_sequence_length, num_ccsk_sequence])), shape=[-1])
                mapping_list.append(mapping_index)
                offset += num_ccsk_sequence*ccsk_sequence_length
            
            self._mapping_index = tf.concat(mapping_list, axis=0)

    def call(self, inputs):
        """
        Takes as inputs iterative mapping symbols and apply demapping.
        
        Backward compatibility Input/Output
        input shape = [..., num_ccsk_per_ofmd_symbol * N], tf.complex64
        output shape = [..., num_ccsk_per_ofmd_symbol * N], tf.complex64

        New design Input/Output
        input shape = [..., num_ccsk_per_frame * N], tf.complex64
        output shape = [..., num_ccsk_per_frame * N], tf.complex64
        """
        # Apply iterative demapping per OFDM symbol
        x = tf.gather(inputs, indices=self._mapping_index, axis=-1)

        return x