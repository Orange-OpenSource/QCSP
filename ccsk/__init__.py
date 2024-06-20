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

from .utils import zadoff_chu_sequence, cross_correlation, compute_modular_inverse, analytic_ccsk_perf_awgn, evaluate_analytic_ccsk, cross_correlation_r, cross_correlation_c
from .ccsk import CCSKModulator, CCSKDemapper
from .mapping import IterativeMapper, IterativeDemapper