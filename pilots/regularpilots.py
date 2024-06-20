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


class RegularPilots(sn.ofdm.PilotPattern):

    def __init__(self, num_ofdm_symb, num_effective_subcarriers, offset_list, ofdm_symb_list, subcarriers_step, pilots=None, seed=1234):
        self._num_ofdm_symb = num_ofdm_symb
        self._num_effective_subcarriers = num_effective_subcarriers # do not take into account DC and guard bands
        self._offset_list = offset_list
        self._ofdm_symb_list = ofdm_symb_list
        self._subcarriers_step = subcarriers_step

        mask = self._generate_regular_pilots_mask_()
        mask = sn.utils.expand_to_rank(mask, 4, axis=0) # output shape = [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers]

        if pilots is None:
            pilots = sn.utils.QAMSource(2, seed=seed, dtype=tf.complex64)([1, 1, tf.reduce_sum(mask)]) # output shape = [num_tx, num_txt_ant, num_pilots]
        else:
            pilots = sn.utils.expand_to_rank(pilots, 3, axis=0) # output shape = [num_tx, num_txt_ant, num_pilots]
        
        super().__init__(mask, pilots, trainable=False, normalize=False, dtype=tf.complex64)

    @property
    def ofdm_symb_list(self):
        """List of the index of the pilot sequences"""
        return self._ofdm_symb_list
    
    def _generate_regular_pilots_mask_(self):
        mask = tf.zeros(shape=[self._num_ofdm_symb, self._num_effective_subcarriers], dtype=tf.int32)
        
        for (offset, symbol) in zip(self._offset_list, self._ofdm_symb_list):
            index_array = tf.range(start=offset, limit=self._num_effective_subcarriers, delta=self._subcarriers_step, dtype=tf.int32)
            update_indices = tf.concat([tf.expand_dims(tf.repeat(symbol, tf.size(index_array), axis=0), axis=1), tf.expand_dims(index_array, axis=1)], axis=1)
            mask = tf.tensor_scatter_nd_update(mask, indices=update_indices, updates=tf.ones(tf.size(index_array), tf.int32))

        return mask
    