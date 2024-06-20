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


class BatchTerminationCallback(tf.keras.callbacks.Callback):

    def __init__(self, condition):
        super(BatchTerminationCallback, self).__init__()
        self.condition = condition
        if tf.__version__ < "2.16.1":
            self.current_step = 0

    def on_test_batch_end(self, batch, logs=None):
        condition = self.condition(batch, logs)
        
        if tf.__version__ < "2.16.1":
            self.current_step += 1
            if condition and self.current_step > 1:
                print('\nStopping Evaluation')
                raise StopIteration() ## Generate an automatic warning from Tensorflow "WARNING:tensorflow:Your input ran out of data; interrupting training..." and correctly interrupt evaluation method

        else:
            if condition:
                print('\nStopping Evaluation')
                self.model.stop_evaluating = True ## Not yet implemented in 2.15.1
