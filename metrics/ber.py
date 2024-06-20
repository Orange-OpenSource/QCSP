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


class BitErrorRate(tf.keras.metrics.Metric):

  def __init__(self, name='BER', from_logits=False, **kwargs):
    """Bit Error Rate metric

    Args:
        name (str, optional): metric's name. Defaults to 'BER'.
        from_logits (bool, optional): evaluate the BER from logits? Defaults to False.
    """
    super(BitErrorRate, self).__init__(name=name, **kwargs)
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
    
    errors = tf.abs(y_true-y_pred)
    errors_count = tf.reduce_sum(errors)
    self.errors.assign_add(errors_count)

    bits_count = tf.cast(tf.size(y_pred), dtype=tf.float32)
    self.total.assign_add(bits_count)
    
  def reset_state(self):
    self.errors.assign(0.)
    self.total.assign(0.)
    # tf.print(f'{self.name} is reset')
      

  def result(self):
    return self.errors/self.total

  def get_config(self):
    return {
      "from_logits": self.from_logits
    }

  @classmethod
  def from_config(cls, config):
      return cls(**config)
