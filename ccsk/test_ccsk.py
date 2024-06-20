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
from . import cross_correlation_r
    

def test_cross_correlation_r():
    x = tf.constant([0.4,0.1,0.1,0.2,0.2,0.0], dtype=tf.float32)   
    y = tf.constant([0.1,0.2,0.2,0.0,0.4,0.1], dtype=tf.float32)   

    a = tf.constant([
        [0.4,0.1,0.1,0.2,0.2,0.0],
        [0.4,0.1,0.1,0.2,0.2,0.0],
        [0.4,0.1,0.1,0.2,0.2,0.0],
        [0.4,0.1,0.1,0.2,0.2,0.0],
        [0.4,0.1,0.1,0.2,0.2,0.0],
        [0.4,0.1,0.1,0.2,0.2,0.0],
    ])

    b = tf.constant([
        [0.1,0.2,0.2,0.0,0.4,0.1],
        [0.1,0.1,0.2,0.2,0.0,0.4],
        [0.4,0.1,0.1,0.2,0.2,0.0],
        [0.0,0.4,0.1,0.1,0.2,0.2],
        [0.2,0.0,0.4,0.1,0.1,0.2],
        [0.2,0.2,0.0,0.4,0.1,0.1]
    ])

    out = tf.reduce_sum(tf.multiply(a,b),axis=-1)

    abs_diff = tf.abs(out - cross_correlation_r(x,y))

    atol = 1e-3
    assert tf.reduce_all(tf.less_equal(abs_diff, atol))