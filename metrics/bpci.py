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

class BinomialProportionConfidenceInterval(tf.keras.metrics.Metric):

    def __init__(self, monitor_class, monitor_params=None, fraction=0.95, dimensions=None, name='BPCI', **kwargs):
        """Metric that monitors a binomial distributed loss or metric and evaluates the confidence interval.

            Args:
                monitored (tf.keras.metrics.Metric): metric to be monitored
                fraction (float, optional): fraction of the values in the interval.
                dimension (list|integer, optional): dimensions to consider for counting. Defaults to 'None'.
                name (str, optional): name of the metric. Defaults to 'BPCI'.
        """
        super(BinomialProportionConfidenceInterval, self).__init__(name=name, **kwargs)
        self.monitored_metric = monitor_class(**monitor_params)
        self.fraction = fraction
        self.dimensions = dimensions
        self.alpha = 1 - fraction
        self.z = 1.0/(1-self.alpha/2)

        self.n = self.add_weight(name='n', initializer='zeros', dtype=tf.float32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.monitored_metric.update_state(y_true, y_pred, sample_weight)
        if self.dimensions is None:
            batch_count = tf.cast(tf.size(y_pred), dtype=tf.float32)
        else:
            shape = tf.cast(tf.shape(y_pred), dtype=tf.float32)
            batch_count = tf.reduce_prod([shape[d] for d in self.dimensions])
        self.n.assign_add(batch_count)

        
    def reset_state(self):
        self.monitored_metric.reset_state()
        self.n.assign(0.)


    def result(self):
        value = self.monitored_metric.result()
        n_tilde = self.n + self.z**2
        k = value * self.n

        if k == 0.:
            half_span = 3.0/(2*self.n)
            confidence_interval = (0., 3.0/self.n)
        else:
            p_tilde = (1.0/n_tilde)*(k+(self.z**2)/2)
            half_span = self.z*tf.sqrt(p_tilde*(1-p_tilde)/n_tilde)
            confidence_interval = (
                p_tilde-half_span,
                p_tilde+half_span
            )
        return (2*half_span, confidence_interval[0], value, confidence_interval[1])

    def get_config(self):
        return {
            "monitored_metric": self.monitored_metric,
            "n": self.n
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BinomialProportionConfidenceInterval_SER(tf.keras.metrics.Metric):

    def __init__(self, monitor_class, num_bits_per_symbol, monitor_params=None, fraction=0.95, name='BPCI', **kwargs):
        """BPCI metric dedicated to SER.

            Args:
                monitored (tf.keras.metrics.Metric): metric to be monitored
                fraction (float, optional): fraction of the values in the interval.
                num_bits_per_symbol (integer): number of bits per symbol.
                name (str, optional): name of the metric. Defaults to 'BPCI'.
        """
        super(BinomialProportionConfidenceInterval_SER, self).__init__(name=name, **kwargs)
        self.monitored_metric = monitor_class(**monitor_params)
        self.fraction = fraction
        self.num_bits_per_symbol = num_bits_per_symbol
        self.alpha = 1 - fraction
        self.z = 1.0/(1-self.alpha/2)

        self.n = self.add_weight(name='n', initializer='zeros', dtype=tf.float32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.monitored_metric.update_state(y_true, y_pred, sample_weight)
        batch_count = tf.cast(tf.size(y_pred), dtype=tf.float32) / tf.cast(self.num_bits_per_symbol, dtype=tf.float32)
        self.n.assign_add(batch_count)

        
    def reset_state(self):
        self.monitored_metric.reset_state()
        self.n.assign(0.)


    def result(self):
        value = self.monitored_metric.result()
        n_tilde = self.n + self.z**2
        k = value * self.n

        if k == 0.:
            half_span = 3.0/(2*self.n)
            confidence_interval = (0., 3.0/self.n)
        else:
            p_tilde = (1.0/n_tilde)*(k+(self.z**2)/2)
            half_span = self.z*tf.sqrt(p_tilde*(1-p_tilde)/n_tilde)
            confidence_interval = (
                p_tilde-half_span,
                p_tilde+half_span
            )
        return (2*half_span, confidence_interval[0], value, confidence_interval[1])

    def get_config(self):
        return {
            "monitored_metric": self.monitored_metric,
            "n": self.n
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)