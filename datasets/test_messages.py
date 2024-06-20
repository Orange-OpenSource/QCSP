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

from . import random_messages, random_messages_dataset, random_messages_base_dataset

def test_messages_shape():
    length = 8
    count = 1_000
    messages = random_messages(length, count)
    assert all(tf.shape(messages) == (count, length))

def test_values_range():
    length = 8
    count = 1_000
    messages = random_messages(length, count)
    assert tf.math.reduce_min(messages) == 0
    assert tf.math.reduce_max(messages) == 1
    assert all(m in {0,1} for m in tf.reshape(messages, shape=(-1,)).numpy())

def test_seed_repeatability():
    seed=1
    length = 8
    count = 1_000
    messages1 = random_messages(length, count, seed=seed)
    messages2 = random_messages(length, count, seed=seed)
    tf.debugging.assert_equal(messages1, messages2)

def test_random_messages_dataset_type():
    dataset = random_messages_dataset(8, seed=None)
    message = list(dataset.take(1))
    tf.debugging.assert_type(message, tf.float32)

def test_random_messages_dataset_batch_size():
    dataset = random_messages_dataset(8, seed=None)
    dataset.prefetch(10)
    dataset.batch(2)
    it = dataset.as_numpy_iterator()
    message = [next(it) for _ in range(3)]
    tf.debugging.assert_type(message, tf.float32)

def test_random_messages_base_dataset():
    dataset = random_messages_base_dataset(8)
    it = dataset.as_numpy_iterator()
    messages = [next(it) for _ in range(100)]
    ones_count = tf.reduce_sum(messages, axis=-1)
    assert all(tf.reduce_max(ones_count, axis=-1)== 1)

if __name__ == "__main__":
    pytest.main()