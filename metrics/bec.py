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


class BitErrorCount(tf.keras.metrics.Metric):

  def __init__(self, name='BEC', from_logits=False, mode=None, **kwargs):
    """Bit Error Rate metric

    Args:
        name (str, optional): metric's name. Defaults to 'BEC'.
        from_logits (bool, optional): evaluate the BER from logits? Defaults to False.
        mode (str, optional): Defaults to None. 'average': average accross batches, 'sum': sum over batches.
    """
    super(BitErrorCount, self).__init__(name=name, **kwargs)
    self.bec = self.add_weight(name='BEC', initializer='zeros', dtype=tf.float32)
    self.n = self.add_weight(name='n', initializer='zeros', dtype=tf.float32)
    self.from_logits = from_logits
    self.mode = mode
    
    
  def update_state(self, y_true, y_pred, sample_weight=None):
    if self.from_logits == True:
      y_pred = tf.math.sign(y_pred)
      y_pred += 1
      y_pred /= 2
    y_pred = tf.clip_by_value(y_pred, 0, 1)
    y_pred = tf.round(y_pred)
    differences = tf.abs(y_true-y_pred)
    bec = tf.reduce_sum(differences)
    if self.mode == 'average':
      count = tf.cast(tf.size(y_pred), dtype=tf.float32)
      bec = self.bec * self.n + count * bec
      self.n.assign_add(count)
      self.bec.assign(bec/self.n)
    elif self.mode == 'sum':
      self.bec.assign_add(bec)
    else:
      self.bec.assign(bec)
      

  def result(self):
    return self.bec

  def reset_state(self):
    self.bec.assign(0.)
    self.n.assign(0.)
    tf.print(f'{self.name} is reset')

  def get_config(self):
    return {
      "from_logits": self.from_logits
    }

  @classmethod
  def from_config(cls, config):
      return cls(**config)

