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


class MeanSampleOffset(tf.keras.metrics.Metric):

	def __init__(self, name='MeanSampleOffset', **kwargs):
		super(MeanSampleOffset, self).__init__(name=name, **kwargs)
		self.errors = self.add_weight(name='errors', initializer='zeros', dtype=tf.int32)
		self.total =  self.add_weight(name='total', initializer='zeros', dtype=tf.int32)
	
	def update_state(self, y_true, y_pred, sample_weight=None):
		errors = tf.abs(y_true - y_pred)
		errors_count = tf.reduce_sum(errors)
		self.errors.assign_add(errors_count)

		batch_count = tf.cast(tf.size(y_pred), dtype=tf.int32)
		self.total.assign_add(batch_count)

	def reset_state(self):
		self.errors.assign(0)
		self.total.assign(0)      

	def result(self):
		return self.errors/self.total

	def get_config(self):
		return {}

	@classmethod
	def from_config(cls, config):
		return cls(**config)


class HistogramSampleOffset(tf.keras.metrics.Metric):

	def __init__(self, name='HistogramSampleOffset', max_neg_offset=-100, max_pos_offset=+100, **kwargs):
		super(HistogramSampleOffset, self).__init__(name=name, **kwargs)
		self.max_neg_offset = max_neg_offset
		self.max_pos_offset = max_pos_offset
		self.nb_bins = max_pos_offset - max_neg_offset + 1
		self.histogram = self.add_weight(shape=[self.nb_bins], name='histogram', initializer='zeros', dtype=tf.int32)

	def update_state(self, y_true, y_pred, sample_weight=None):
		relative_errors = tf.reshape(tf.cast(y_pred - y_true, dtype=tf.int32), shape=[-1])
		relative_errors = tf.gather(relative_errors, indices=tf.where(tf.logical_and(tf.math.greater(relative_errors, self.max_neg_offset), tf.math.less(relative_errors, self.max_pos_offset))), axis=-1)
		current_histogram = tf.histogram_fixed_width(relative_errors, value_range=[self.max_neg_offset, self.max_pos_offset], nbins=self.nb_bins, dtype=tf.int32)
		self.histogram.assign_add(current_histogram)

	def reset_state(self):
		self.histogram.assign(tf.zeros(shape=[self.nb_bins], dtype=tf.int32))
	
	def result(self):
		return [self.histogram, tf.range(start=self.max_neg_offset, limit=self.max_pos_offset+1, delta=1, dtype=tf.int32)]

	def get_config(self):
		return {
			'max_neg_offset': self.max_neg_offset,
			'max_pos_offset': self.max_pos_offset,
			'nb_bins': self.nb_bins
		}

	@classmethod
	def from_config(cls, config):
		return cls(**config)

