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


class CCSKModulator(tf.keras.layers.Layer):

    def __init__(self, root_sequence, num_bits_per_ccsk_sequence, shift_output=False, **kwargs):
        super().__init__(**kwargs)
        self._root_sequence = root_sequence
        self._N = tf.size(root_sequence, out_type=tf.int32) 
        self._num_bits_per_ccsk_sequence = num_bits_per_ccsk_sequence
        self._binary_shift = tf.range(start=self._num_bits_per_ccsk_sequence-1, limit=-1, delta=-1, dtype=tf.int32)
        self.shift_output = shift_output

        tensor_list = []
        for shift in tf.range(self._N, dtype=tf.int32):
            tensor_list.append(tf.expand_dims(tf.roll(self._root_sequence, shift=shift, axis=0), axis=0))
        self._mapping_array = tf.concat(tensor_list, axis=0)

    def call(self, inputs):
        """
        Takes as inputs the bits to convert to CCSK symbols
        input shape = [batch_size, num_ccsk_per_frame * num_bits_per_ccsk_sequence], tf.float32
        output shape = [batch_size, num_ccsk_per_frame * N], tf.complex64
        """
        # Reshape inputs to the desired format
        new_shape = [inputs.shape[0], int(inputs.shape[-1] / self._num_bits_per_ccsk_sequence), self._num_bits_per_ccsk_sequence]
        inputs_reshaped = tf.cast(tf.reshape(inputs, new_shape), tf.int32) # output shape = [batch_size, num_ccsk_per_frame, num_bits_per_ccsk_sequence]

        # Convert the last dimension to an integer
        int_rep = tf.reduce_sum(tf.bitwise.left_shift(inputs_reshaped, self._binary_shift), axis=-1)

        # Map integers to constellation symbols
        x = tf.gather(self._mapping_array, indices=int_rep, axis=0)

        # Flatten CCSK symbols per OFDM frame
        x = sn.utils.flatten_last_dims(x, num_dims=2)

        if self.shift_output:
            return x, int_rep

        return x



class CCSKDemapper(tf.keras.layers.Layer):

    def __init__(self, root_sequence, num_bits_per_ccsk_sequence, **kwargs):
        super().__init__(**kwargs)
        self._root_sequence = root_sequence
        self._N = tf.size(root_sequence, out_type=tf.int32) 
        self._num_bits_per_ccsk_sequence = num_bits_per_ccsk_sequence
        self._binary_shift = tf.range(start=self._num_bits_per_ccsk_sequence-1, limit=-1, delta=-1, dtype=tf.int32)

        int_base = tf.repeat(tf.expand_dims(tf.range(2**self._num_bits_per_ccsk_sequence, dtype=tf.int32), axis=1), self._num_bits_per_ccsk_sequence, axis=1)
        self._demapping_array = tf.bitwise.right_shift(tf.bitwise.bitwise_and(int_base, 2**self._binary_shift), self._binary_shift)

    def call(self, inputs):
        """
        Takes as inputs the shift value from the argmax of the cross-correlation
        input shape = [batch_size, num_ccsk_per_frame], tf.int32
        output shape = [batch_size, num_ccsk_per_frame * num_bits_per_ccsk_sequence], tf.float32
        """
        # Get the binary representation corresponding to the shift values
        x = tf.gather(self._demapping_array, indices=inputs, axis=0) # output shape = [batch_size, num_ccsk_per_frame, num_bits_per_ccsk_sequence]

        # Reshape to have num_ccsk_per_frame * num_bits_per_ccsk_sequence
        x = tf.cast(sn.utils.flatten_last_dims(x, num_dims=2), tf.float32)
        
        return x