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


def random_messages(length, count, seed=None):
    """generate `count` random messages of length `length`.

    Args:
        length (int): length of the messages in bits
        count (int): number of messages to generate
        seed (integer, optional): RNG seed. Defaults to None.

    Returns:
        tf.Tensor(dtype=tf.float3): random messages
    """
    if seed is None:
        rng = tf.random.get_global_generator()
    else:
        rng = tf.random.Generator.from_seed(seed)
    dataset = tf.cast(rng.uniform((count, length), maxval=2, dtype=tf.int32), dtype=tf.float32)
    return dataset


def random_messages_dataset(length, batch=256, prefetch=tf.data.AUTOTUNE, seed=None):
    """generate random sequences of bits of length `length` as dataset

    Args:
        length (tf.int32): length of bits sequences
        batch (tf.int32, optional): batch size. Defaults to 256.
        prefetch (tf.int32, optional): count of pre-generated sequences. Defaults to AUTOTUNE.
        seed (tf.int64|none, optional): generator's seed. Defaults to None. If None, retrieve global generator.

    Returns:
        tf.data.Dataset: dataset
    """
    if seed is None:
        rng = tf.random.get_global_generator()
    else:
        rng = tf.random.Generator.from_seed(seed)

    dataset = tf.data.Dataset.from_tensors(
        tf.zeros(shape=(batch, length,), dtype=tf.float32)
    ).cache(
        ""
    ).repeat(
        count=None
    ).map(
        lambda x: x + tf.cast(rng.uniform(shape=(batch, length,), minval=0, maxval=2, dtype=tf.int32), dtype=tf.float32),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    ).prefetch(
        prefetch
    )
    return dataset
    

def random_messages_base_dataset(length, batch=256, prefetch=tf.data.AUTOTUNE, seed=None):
    """generate random bases of the sequences of bits of length `length`, e.g. [0,...,0,1,0,...,0] as dataset

    Args:
        length (tf.int32): length of bits sequences
        batch (tf.int32, optional): batch size. Defaults to 256.
        prefetch (tf.int32, optional): count of pre-generated sequences. Defaults to AUTOTUNE.
        seed (tf.int64|none, optional): generator's seed. Defaults to None. If None, retrieve global generator.

    Returns:
        tf.data.Dataset: dataset
    """
    if seed is None:
        rng = tf.random.get_global_generator()
    else:
        rng = tf.random.Generator.from_seed(seed)
    base = tf.eye(length, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensors(
        tf.zeros(shape=(batch, length,), dtype=tf.float32)
    ).cache(
        ""
    ).repeat(
        count=None
    ).map(
        lambda x: x + base[rng.uniform((1,), minval=0, maxval=length, dtype=tf.int32)[0]],
        num_parallel_calls=tf.data.AUTOTUNE, 
        deterministic=False
    ).prefetch(
        prefetch
    )
    return dataset


def zeros_messages_dataset(length, batch=256, prefetch=tf.data.AUTOTUNE):
    """generate constant 0. sequences of bits of length `length`, e.g. [0,...,0] as dataset

    Args:
        length (tf.int32): length of bits sequences
        batch (tf.int32, optional): batch size. Defaults to 256.
        prefetch (tf.int32, optional): count of pre-generated sequences. Defaults to AUTOTUNE.

    Returns:
        tf.data.Dataset: dataset
    """

    dataset = tf.data.Dataset.from_tensors(
        tf.zeros(shape=(batch, length,), dtype=tf.float32)
    ).cache(
        ""
    ).repeat(
        count=None
    ).prefetch(
        prefetch
    )
    return dataset