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



def test_ber_logits():
    # errors in positions 1, 3 and high uncertainty on position 6
    bits_truth = tf.constant([[0, 0, 1, 1, 1, 1, 0, 1]], dtype=tf.float32)
    logits = tf.constant([[-10, 3, 5, -2, 1, 23, 0, 2]], dtype=tf.float32)
    
    expected_ber = 0.25

    ber_metric = BitErrorRate(from_logits=True)
    ber = ber_metric(y_true=bits_truth, y_pred=logits)
    assert(ber == expected_ber)

def test_ber_nologits():
    # errors in positions 5, 6, 7 
    bits_truth = tf.constant([[0, 0, 1, 1, 1, 1, 0, 1]], dtype=tf.float32)
    bits_pred = tf.constant([[0, 0, 1, 1, 1, 0, 1, 0]], dtype=tf.float32)

    expected_ber = 3./8

    ber_metric = BitErrorRate(from_logits=False)
    ber = ber_metric(y_true=bits_truth, y_pred=bits_pred)
    assert(ber == expected_ber)

def test_ber_average_across_batches():
    # errors in positions 5, 6, 7 
    bits_truth = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
    bits_pred = tf.constant([[0, 0, 1, 1, 1, 0, 1, 0]], dtype=tf.float32)

    expected_ber = 4./16. 
    ber_metric = BitErrorRate(from_logits=False)
    ber = ber_metric(y_true=bits_truth, y_pred=bits_pred)
    ber = ber_metric(y_true=bits_truth, y_pred=bits_truth)
    assert(ber == expected_ber)

if __name__ == "__main__":
    pytest.main()