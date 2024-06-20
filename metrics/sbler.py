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


class ShiftBlockErrorRate(tf.keras.metrics.Metric):

  def __init__(self, num_bits_per_symbol, name='SBLER', from_logits=False, **kwargs):
    super(ShiftBlockErrorRate, self).__init__(name=name, **kwargs)
    self.num_bits_per_symbol = num_bits_per_symbol
    self.nb_shift = 2**num_bits_per_symbol
    self.bit_to_int_vec = tf.expand_dims(tf.expand_dims(2**tf.range(start=num_bits_per_symbol-1, limit=-1, delta=-1, dtype=tf.float32), axis=0), axis=0)
    self.from_logits = from_logits
    self.errors = self.add_weight(name='errors', initializer='zeros', dtype=tf.float32)
    self.total =  self.add_weight(name='total', initializer='zeros', dtype=tf.float32)
        
  def update_state(self, y_true, y_pred, sample_weight=None):
    if self.from_logits == True:
      y_pred = tf.math.sign(y_pred)
      y_pred += 1
      y_pred /= 2
    y_pred = tf.clip_by_value(y_pred, 0, 1)
    y_pred = tf.round(y_pred)

    y_true_int = tf.reduce_sum(tf.reshape(y_true, shape=[y_true.shape[0], -1, self.num_bits_per_symbol]) * self.bit_to_int_vec, axis=-1)
    y_pred_int = tf.reduce_sum(tf.reshape(y_pred, shape=[y_pred.shape[0], -1, self.num_bits_per_symbol]) * self.bit_to_int_vec, axis=-1)
    shift_values = tf.math.mod(y_pred_int - y_true_int, self.nb_shift)
    shift_errors = tf.reduce_max(tf.abs(shift_values - shift_values[:,:1]), axis=-1)
    errors_count = tf.math.count_nonzero(shift_errors, dtype=tf.float32)
    self.errors.assign_add(errors_count)

    blocks_count = tf.cast(tf.size(shift_errors), dtype=tf.float32)
    self.total.assign_add(blocks_count)

    
  def reset_state(self):
    self.errors.assign(0.)
    self.total.assign(0.)

  def result(self):
    return self.errors/self.total

  def get_config(self):
    return {
      "from_logits": self.from_logits
    }

  @classmethod
  def from_config(cls, config):
      return cls(**config)
