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

import pytest
import tensorflow as tf

from . import BitErrorRate
from . import BinomialProportionConfidenceInterval


def test_bpci_ber_logits():
    # errors in positions 1, 3 and high uncertainty on position 6
    bits_truth = tf.constant([[0, 0, 1, 1, 1, 1, 0, 1]], dtype=tf.float32)
    logits = tf.constant([[-10, 3, 5, -2, 1, 23, 0, 2]], dtype=tf.float32)
    
    expected_ber = 0.25

    ber_metric = BitErrorRate(from_logits=True)
    bpci_metric = BinomialProportionConfidenceInterval(ber_metric, fraction=0.95, dimension=-1, name='ber_confidence_interval')

    confidence_interval = bpci_metric(y_true=bits_truth, y_pred=logits)
    assert all(confidence_interval == [0.12614864, 0.43195713])

def test_bpci_ber_nologits():
    # errors in positions 5, 6, 7 
    bits_truth = tf.constant([[0, 0, 1, 1, 1, 1, 0, 1]], dtype=tf.float32)
    bits_pred = tf.constant([[0, 0, 1, 1, 1, 0, 1, 0]], dtype=tf.float32)
    ber_metric = BitErrorRate(from_logits=False)
    bpci_metric = BinomialProportionConfidenceInterval(ber_metric, fraction=0.95, dimension=-1, name='ber_confidence_interval')

    confidence_interval = bpci_metric(y_true=bits_truth, y_pred=bits_pred)
    assert all(confidence_interval == [0.22328992, 0.555763])

if __name__ == "__main__":
    pytest.main()