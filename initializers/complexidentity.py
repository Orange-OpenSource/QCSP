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


class ComplexIdentity(tf.keras.initializers.Initializer):

    def __init__(self, r_index=0, c_index=1):
        super().__init__()
        self.r_index = r_index
        self.c_index = c_index
         
    def __call__(self, shape, dtype=None, **kwargs):
        return tf.eye(num_rows=shape[self.r_index], num_columns=shape[self.c_index], dtype=tf.complex64)
