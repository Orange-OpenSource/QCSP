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

from .ofdm import LeastSquareEstimator, LMMSEEstimator, TimeChannelEstimation, compute_covmatrix_cir, compute_covmatrix_expdecayPDP, compute_covmatrix_uniformPDP, compute_covmatrix_BCOMexpdecayPDP, compute_time_covmatrix_cir, compute_covmatrix_channelgenerator