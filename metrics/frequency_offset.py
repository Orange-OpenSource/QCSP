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


class MeanFractionalFrequencyOffset(tf.keras.metrics.Metric):

	def __init__(self, name='MeanFractionalFrequencyOffset', **kwargs):
		super(MeanFractionalFrequencyOffset, self).__init__(name=name, **kwargs)
		self.errors = self.add_weight(name='errors', initializer='zeros', dtype=tf.float32)
		self.total =  self.add_weight(name='total', initializer='zeros', dtype=tf.float32)
	
	def update_state(self, y_true, y_pred, sample_weight=None):
		errors = tf.abs(y_pred)
		errors_count = tf.reduce_sum(errors)
		self.errors.assign_add(errors_count)

		batch_count = tf.cast(tf.size(y_pred), dtype=tf.float32)
		self.total.assign_add(batch_count)

	def reset_state(self):
		self.errors.assign(0.)
		self.total.assign(0.)      

	def result(self):
		return self.errors/self.total

	def get_config(self):
		return {}

	@classmethod
	def from_config(cls, config):
		return cls(**config)


class HistogramFractionalFrequencyOffset(tf.keras.metrics.Metric):

	def __init__(self, name='HistogramFractionalFrequencyOffset', nb_bins=101, **kwargs):
		super(HistogramFractionalFrequencyOffset, self).__init__(name=name, dtype=tf.float32, **kwargs)
		self.max_neg_offset = -0.5
		self.max_pos_offset = +0.5
		self.nb_bins = nb_bins
		self.delta = 1/nb_bins
		self.histogram = self.add_weight(shape=[self.nb_bins], name='histogram', initializer='zeros', dtype=tf.int32)

	def update_state(self, y_true, y_pred, sample_weight=None):
		relative_errors = tf.reshape(y_pred, shape=[-1])
		current_histogram = tf.histogram_fixed_width(relative_errors, value_range=[self.max_neg_offset, self.max_pos_offset], nbins=self.nb_bins, dtype=tf.int32)
		self.histogram.assign_add(current_histogram)

	def reset_state(self):
		self.histogram.assign(tf.zeros(shape=[self.nb_bins], dtype=tf.int32))
	
	def result(self):
		return [tf.cast(self.histogram, dtype=tf.float32), tf.range(start=self.max_neg_offset, limit=self.max_pos_offset, delta=self.delta, dtype=tf.float32)]

	def get_config(self):
		return {
			'max_neg_offset': self.max_neg_offset,
			'max_pos_offset': self.max_pos_offset,
			'nb_bins': self.nb_bins
		}

	@classmethod
	def from_config(cls, config):
		return cls(**config)

