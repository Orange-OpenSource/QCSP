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

from . import BlockErrorRate



def test_bler_logits():
    # 1 bit error in block 3
    block_truth = tf.constant([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=tf.float32)
    logits = tf.constant([
        [-10, -10, -10, -10, -10, -10, -10, -10],
        [-10, -10, -10, -10, -10, -10, -10, -10],
        [-10, -10, -10, -10, -10, -10, -10, +10],
        [-10, -10, -10, -10, -10, -10, -10, -10],
    ], dtype=tf.float32)
    
    expected_bler = 0.25

    bler_metric = BlockErrorRate(from_logits=True)
    bler = bler_metric(y_true=block_truth, y_pred=logits)
    assert(bler == expected_bler)

def test_bler_nologits():
    # errors in positions 5, 6, 7 
    block_truth = tf.constant([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=tf.float32)
    block_pred =  tf.constant([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=tf.float32)

    expected_bler = 0.25

    bler_metric = BlockErrorRate(from_logits=False)
    bler = bler_metric(y_true=block_truth, y_pred=block_pred)
    assert(bler == expected_bler)

if __name__ == "__main__":
    pytest.main()