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


class BlockErrorCount(tf.keras.metrics.Metric):

    def __init__(self, name='BLEC', from_logits=False, mode=None, **kwargs):
        """Block Error Rate metric

        Args:
            name (str, optional): metric's name. Defaults to 'BLEC'.
            from_logits (bool, optional): evaluate the metric from logits? Defaults to False.
            mode (str, optional): Defaults to None. 'average': average accross batches, 'sum': sum over batches.
        """
        super(BlockErrorCount, self).__init__(name=name, **kwargs)
        self.blec = self.add_weight(name='BLEC', initializer='zeros', dtype=tf.float32)
        self.from_logits=from_logits
        self.mode = mode
        self.n = 0

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits == True:
            y_pred = tf.math.sign(y_pred)
            y_pred += 1
            y_pred /= 2
        y_pred = tf.clip_by_value(y_pred, 0, 1)
        y_pred = tf.round(y_pred)
        differences = tf.abs(y_true-y_pred)
        differences = tf.reduce_max(differences, axis=1)
        blec = tf.reduce_sum(differences)
        
        if self.mode == 'average':
            count = tf.cast(tf.size(differences), dtype=tf.float32)
            blec = self.blec * self.n + count * blec
            self.n += count
            self.blec.assign(blec/self.n)
        elif self.mode == 'sum':
            blec = self.blec + blec
            self.blec.assign(blec)
        else:
            self.blec.assign(blec)

    def result(self):
        return self.blec

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

