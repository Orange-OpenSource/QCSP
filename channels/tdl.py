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

"""
Derived from Sionna class sionna.channel.tr38901.TDL() in version 0.10 to load a custom .json TDL channel model + toggle normalize taps powers

SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""


import json
import numpy as np

import tensorflow as tf

from sionna import PI, SPEED_OF_LIGHT
from sionna.utils import insert_dims
from sionna.channel import ChannelModel



class CustomTDL(ChannelModel):
    """
    From the TDL model of Sionna

    Parameters
    -----------
    model_path : str
        Path to the JSON file TDL model to use.

    delay_spread : float
        RMS delay spread [s]

    carrier_frequency : float
        Carrier frequency [Hz]

    num_sinusoids : int
        Number of sinusoids for the sum-of-sinusoids model. Defaults to 20.

    los_angle_of_arrival : float
        Angle-of-arrival for LoS path [radian]. Only used with LoS models.
        Defaults to :math:`\pi/4`.

    min_speed : float
        Minimum speed [m/s]. Defaults to 0.

    max_speed : None or float
        Maximum speed [m/s]. If set to `None`,
        then ``max_speed`` takes the same value as ``min_speed``.
        Defaults to `None`.

    dtype : Complex tf.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.

    Input
    -----

    batch_size : int
        Batch size

    num_time_steps : int
        Number of time steps

    sampling_frequency : float
        Sampling frequency [Hz]

    Output
    -------
    a : [batch size, num_rx = 1, num_rx_ant = 1, num_tx = 1, num_tx_ant = 1, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx = 1, num_tx = 1, num_paths], tf.float
        Path delays [s]

    """

    def __init__(   self,
                    model_path,
                    delay_spread,
                    carrier_frequency,
                    num_sinusoids=20,
                    los_angle_of_arrival=PI/4.,
                    min_speed=0.,
                    max_speed=None,
                    normalize_taps=False,
                    dtype=tf.complex64):

        assert dtype.is_complex, "dtype must be a complex datatype"
        self._dtype = dtype
        real_dtype = dtype.real_dtype
        self._real_dtype = real_dtype

        self._normalize_taps = normalize_taps

        # Load model parameters
        self._load_parameters(model_path)

        self._carrier_frequency = tf.constant(carrier_frequency, real_dtype)
        self._num_sinusoids = tf.constant(num_sinusoids, tf.int32)
        self._los_angle_of_arrival = tf.constant(los_angle_of_arrival, real_dtype)
        self._delay_spread = tf.constant(delay_spread, real_dtype)
        self._min_speed = tf.constant(min_speed, real_dtype)
        if max_speed is None:
            self._max_speed = self._min_speed
        else:
            assert max_speed >= min_speed, \
                "min_speed cannot be larger than max_speed"
            self._max_speed = tf.constant(max_speed, real_dtype)

        # Pre-compute maximum and minimum Doppler shifts
        self._min_doppler = self._compute_doppler(self._min_speed)
        self._max_doppler = self._compute_doppler(self._max_speed)

        # Precompute average angles of arrivals for each sinusoid
        alpha_const = 2.*PI/num_sinusoids * \
                      tf.range(1., self._num_sinusoids+1, 1., dtype=real_dtype)
        self._alpha_const = tf.reshape( alpha_const,
                                        [   1, # batch size
                                            1, # num rx
                                            1, # num rx ant
                                            1, # num tx
                                            1, # num tx ant
                                            1, # num clusters
                                            1, # num time steps
                                            num_sinusoids])
    @property
    def carrier_frequency(self):
        return self._carrier_frequency
    
    @property
    def min_speed(self):
        return self._min_speed

    @property
    def max_speed(self):
        return self._max_speed

    @property
    def num_clusters(self):
        r"""Number of paths (:math:`M`)"""
        return self._num_clusters

    @property
    def los(self):
        r"""`True` if this is a LoS model. `False` otherwise."""
        return self._los

    @property
    def k_factor(self):
        r"""K-factor in linear scale. Only available with LoS models."""
        assert self._los, "This property is only available for LoS models"
        return tf.math.real(self._k_factor)

    @property
    def delays(self):
        r"""Path delays [s]"""
        return self._delays*self._delay_spread

    @property
    def mean_powers(self):
        r"""Path powers in linear scale"""
        return tf.math.real(self._mean_powers)

    @property
    def delay_spread(self):
        r"""RMS delay spread [s]"""
        return self._delay_spread

    @delay_spread.setter
    def delay_spread(self, value):
        self._delay_spread = value

    def __call__(self, batch_size, num_time_steps, sampling_frequency):

        # Time steps
        sample_times = tf.range(num_time_steps, dtype=self._real_dtype) / sampling_frequency
        sample_times = tf.expand_dims(insert_dims(sample_times, 6, 0), -1)

        # Generate random maximum Doppler shifts for each sample
        # The Doppler shift is different for each TX-RX link, but shared by
        # all RX ant and TX ant couple for a given link.
        doppler = tf.random.uniform([   batch_size,
                                        1, # num rx
                                        1, # num rx ant
                                        1, # num tx
                                        1, # num tx ant
                                        1, # num clusters
                                        1, # num time steps
                                        1], # num sinusoids
                                        self._min_doppler,
                                        self._max_doppler,
                                        self._real_dtype)

        # Eq. (7) in the paper [TDL] (see class docstring in Sionna)
        # The angle of arrival is different for each TX-RX link.
        theta = tf.random.uniform([ batch_size,
                                    1, # num rx
                                    1, # 1 RX antenna
                                    1, # num tx
                                    1, # 1 TX antenna
                                    self._num_clusters,
                                    1, # num time steps
                                    self._num_sinusoids],
                                    -PI/tf.cast(self._num_sinusoids,
                                                self._real_dtype),
                                    PI/tf.cast( self._num_sinusoids,
                                                self._real_dtype),
                                    self._real_dtype)
        alpha = self._alpha_const + theta

        # Eq. (6a)-(6c) in the paper [TDL] (see class docstring in Sionna)
        phi = tf.random.uniform([   batch_size,
                                    1, # 1 RX
                                    1, # 1 RX antenna
                                    1, # 1 TX
                                    1, # 1 TX antenna
                                    self._num_clusters,
                                    1, # Phase shift is shared by all time steps
                                    self._num_sinusoids],
                                    -PI,
                                    PI,
                                    self._real_dtype)

        argument = doppler * sample_times * tf.cos(alpha) + phi

        # Eq. (6a) in the paper [SoS]
        h = tf.complex(tf.cos(argument), tf.sin(argument))
        normalization_factor = 1./tf.sqrt(  tf.cast(self._num_sinusoids,
                                            self._real_dtype))
        h = tf.complex(normalization_factor, tf.constant(0., self._real_dtype))\
            *tf.reduce_sum(h, axis=-1)

        # Scaling by average power
        mean_powers = tf.expand_dims(insert_dims(self._mean_powers, 5, 0), -1)
        h = tf.sqrt(mean_powers)*h

        # Add specular component to first tap Eq. (11) in [SoS] if LoS
        if self._los:
            # The first tap has a total power of 0dB and follows a Rician
            # distribution

            # Specular component phase shift
            phi_0 = tf.random.uniform([ batch_size,
                                        1, # num rx
                                        1, # 1 RX antenna
                                        1, # num tx
                                        1, # 1 TX antenna
                                        1, # only the first tap is concerned
                                        1], # Shared by all time steps
                                        PI,
                                        -PI,
                                        self._real_dtype)
            # Remove the sinusoids dim
            doppler = tf.squeeze(doppler, axis=-1)
            sample_times = tf.squeeze(sample_times, axis=-1)
            arg_spec = doppler*sample_times*tf.cos(self._los_angle_of_arrival)\
                    + phi_0
            h_spec = tf.complex(tf.cos(arg_spec), tf.sin(arg_spec))

            # Update the first tap with the specular component
            h = tf.concat([ (h_spec*tf.sqrt(self._k_factor) + h[:,:,:,:,:,:1,:])
                            /tf.sqrt(tf.cast(1, self._dtype) + self._k_factor),
                            h[:,:,:,:,:,1:,:]],
                            axis=5) # Path dims

        # Delays
        delays = self._delays*self._delay_spread
        delays = insert_dims(delays, 3, 0)
        delays = tf.tile(delays, [batch_size, 1, 1, 1])

        # Stop gadients to avoid useless backpropagation
        h = tf.stop_gradient(h)
        delays = tf.stop_gradient(delays)

        return h, delays

    ###########################################
    # Internal utility functions
    ###########################################

    def _compute_doppler(self, speed):
        """Compute the maximum radian Doppler frequency [Hz] for a given
        speed [m/s].

        Input
        ------
        speed : float
            Speed [m/s]

        Output
        --------
        doppler_shift : float
            Doppler shift [Hz]
        """
        return 2.*PI*speed/SPEED_OF_LIGHT*self._carrier_frequency

    def _load_parameters(self, file_path):
        r"""Load parameters of a TDL model.

        The model parameters are stored as JSON files with the following keys:
        * los : boolean that indicates if the model is a LoS model
        * num_clusters : integer corresponding to the number of clusters (paths)
        * delays : List of path delays in ascending order normalized by the RMS
            delay spread
        * powers : List of path powers in dB scale, normalized by the most powerful path
            Highest power is 0 dB.

        For LoS models, the two first paths have zero delay, and are assumed
        to correspond to the specular and NLoS component, in this order.

        Input
        ------
        file_path : str
            Path of the file from which to load the parameters.

        Output
        ------
        None
        """
        # pylint: disable=unspecified-encoding
        with open(file_path) as parameter_file:
            params = json.load(parameter_file)

        # LoS scenario ?
        self._los = bool(params['los'])

        # Loading cluster delays and mean powers
        self._num_clusters = tf.constant(params['num_clusters'], tf.int32)

        # Retrieve power and delays
        delays = tf.constant(params['delays'], self._real_dtype)
        mean_powers = np.power(10.0, np.array(params['powers'])/10.0)
        mean_powers = tf.constant(mean_powers, self._dtype)

        if self._los:
            # The first tap has a mean power of 1 (0 dB) and follows a Rician
            # distribution.
            # K-factor of this Rician distribution
            k_factor = mean_powers[0]/mean_powers[1]
            self._k_factor = k_factor

            # We remove the delays and powers of the specular component of the
            # first tap as it is added separately
            mean_powers = mean_powers[1:]
            # Set the power of the non-specular component of the first tap to
            # one. The specular and non-specular components are scaled
            # accordingly using the K-factor when generating the channel
            # coefficients
            mean_powers = tf.tensor_scatter_nd_update(mean_powers,
                                                      [[0]],
                                                      tf.ones([1], self._dtype))
            delays = delays[1:]

        self._delays = delays

        if self._normalize_taps: 
            self._mean_powers = mean_powers / tf.reduce_sum(mean_powers)
        else:
            self._mean_powers = mean_powers
