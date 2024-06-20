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

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import itertools



def plot_complex_constellation(symbols, save_path='constellation.png', fig_title='Constellation'):
    """Plot the complex symbol provided in input as a constellation with real and imaginary parts.
    The figure is saved as a .png

    Parameters
    ----------
    symbols : 1D Tensor of tf.complex64/128
        The symbols to plot

    save_path : str, optional
        The full save path with figure file name .png, by default 'constellation.png'
        
    fig_title : str, optional
        Title of the figure, by default 'Constellation'

    Returns
    -------
    matplotlib.pyplot Figure
        The created figure.
    """
    maxval = tf.reduce_max(tf.math.abs(symbols))*1.05
    fig = plt.figure(facecolor='w', figsize=(7,7))
    ax = fig.add_subplot(111)
    plt.xlim(-maxval, maxval)
    plt.ylim(-maxval, maxval)
    ax.set_aspect('equal', adjustable='box')
    plt.scatter(tf.math.real(symbols), tf.math.imag(symbols), marker='o', c='b')
    plt.vlines(0., ymin=-maxval, ymax=maxval, colors='k', linewidths=2.)
    plt.hlines(0., xmin=-maxval, xmax=maxval, colors='k', linewidths=2.)
    plt.grid(True, which="both", axis="both")
    plt.title(fig_title)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.savefig(save_path, format='png', dpi=300, facecolor='w', transparent=False)
    return fig


def plot_dataframe_ber_bler(dataframe, save_path, title=None, which='ber', index='ebn0', figsize=(15,10)):
    """Plot the which vs index from the dataframe.
    The dataframe is expected to have all the data to add to the plot.
    The expected structure of the dataframe is as follow:

                SNR     model_1                              
                        ber                 bler (or ser)          
                        mean      95%CI     mean      95%CI
        Eb/N0                                        
        5.0     -5.0    0.006032  0.000507  0.550781  0.03679
        6.0     -6.0    0.002349  0.000224  0.251953  0.02272
        7.0     -7.0    0.000763  0.000076  0.093299  0.00904
        8.0     -8.0    0.000192  0.000019  0.024414  0.00240

    The index Eb/N0 can vary in size.
    Several models can be present (stacked horizontally).

    Parameters
    ----------
    dataframe : pandas Dataframe
        The input dataframe

    save_path : str
        The full save path with figure file name .png
        
    title : str
        Title of the figure. If None or '' set to ylabel vs xlabel.
    
    which : str, optional
        'ber' or 'bler' or 'ser' to plot, by default 'ber'
    
    index : str, optinal
        'ebn0' or 'snr', index to use, by default 'ebn0'

    Returns
    -------
    matplotlib.pyplot Figure
        The created figure.
    """
    if index == 'snr':
        index_label = 'SNR'
        xlabel = 'SNR (dB)'
    else:
        index_label = 'Eb/N0'
        xlabel = 'Eb/N0 (dB)'

    result_df = dataframe.iloc[:, dataframe.columns.get_level_values(1)==which]
    result_mean = result_df.iloc[:, result_df.columns.get_level_values(2)=='mean']
    result_ci = result_df.iloc[:, result_df.columns.get_level_values(2)=='95%CI']
    result_no_mi =  result_mean.droplevel(level=1, axis=1).droplevel(level=1, axis=1)
            
    result_to_plot = result_no_mi

    marker_list = ["^", "s", "o", "*", "p", "x", "v", "2", "d"]
    color_list = "bgrcmyk"

    marker_iterator = itertools.cycle(marker_list)
    color_iterator = itertools.cycle(color_list)

    if which == 'ber':
        ylabel = 'BER'
    elif which == 'bler':
        ylabel = 'BLER'
    elif which == 'ser':
        ylabel = 'SER'

    if (title is None) or (title == ''):
        title = f'{ylabel} vs {index_label}'

    style_list = []
    for _ in np.arange(result_to_plot.shape[1]):
        style_list.append(next(color_iterator) + next(marker_iterator))

    plt.rcParams['text.usetex'] = False
    ax = result_to_plot.plot(figsize=figsize, kind='line', logy=True, xlabel=xlabel, ylabel=ylabel, title=title, ls='-', style=style_list, ms=7., linewidth=2.2)
    ax.grid(which='both')
    ax.grid(which='major', color='black')
    ax.grid(which='minor', color='grey')
    fig = ax.get_figure()
    fig.patch.set_facecolor('w')
    ax.figure.savefig(save_path, format='png', dpi=300, facecolor='w', transparent=False)
    return fig


def plot_list_dataframe_ber_bler(list_dataframe, save_path, title=None, which='ber', index='ebn0', figsize=(15,10)):
    """Plot which vs index from the dataframe.
    The dataframe is expected to have all the data to add to the plot.
    The expected structure of the dataframe is as follow:

                SNR     model_1                              
                        ber                 bler (ser, sbler)  sample_offset     
                        mean      95%CI     mean      95%CI    mean
        Eb/N0                                        
        5.0     -5.0    0.006032  0.000507  0.550781  0.03679  1.0
        6.0     -6.0    0.002349  0.000224  0.251953  0.02272  2.0
        7.0     -7.0    0.000763  0.000076  0.093299  0.00904  2.0
        8.0     -8.0    0.000192  0.000019  0.024414  0.00240  1.0

    The index Eb/N0 can vary in size.
    Several models can be present (stacked horizontally).

    Parameters
    ----------
    dataframe : pandas Dataframe
        The input dataframe

    save_path : str
        The full save path with figure file name .png
        
    title : str
        Title of the figure. If None or '' set to ylabel vs xlabel.
    
    which : str, optional
        'ber' or 'bler' or 'ser' or 'sbler' or 'sample_offset' or 'fractional_frequency_offset' to plot, by default 'ber'
    
    index : str, optinal
        'ebn0' or 'snr', index to use, by default 'ebn0'

    Returns
    -------
    matplotlib.pyplot Figure
        The created figure.
    """
    if index == 'snr':
        index_label = 'SNR'
        xlabel = 'SNR (dB)'
    else:
        index_label = 'Eb/N0'
        xlabel = 'Eb/N0 (dB)'

    for index in np.arange(len(list_dataframe)):
        df = list_dataframe[index]
        result_df = df.iloc[:, df.columns.get_level_values(1)==which]
        result_mean = result_df.iloc[:, result_df.columns.get_level_values(2)=='mean']
        result_ci = result_df.iloc[:, result_df.columns.get_level_values(2)=='95%CI']
        result_no_mi =  result_mean.droplevel(level=1, axis=1).droplevel(level=1, axis=1)
        list_dataframe[index] = result_no_mi

    marker_list = ["^", "s", "o", "*", "p", "x", "v", "2", "d"]
    color_list = "bgrcmyk"

    marker_iterator = itertools.cycle(marker_list)
    color_iterator = itertools.cycle(color_list)

    logy = True

    if which == 'ber':
        ylabel = 'BER'
    elif which == 'bler':
        ylabel = 'BLER'
    elif which == 'ser':
        ylabel = 'SER'
    elif which == 'sbler':
        ylabel = 'SBLER'
    elif which == 'sample_offset':
        ylabel = 'Sample Offset'
    elif which == 'fractional_frequency_offset':
        ylabel = 'Fractional Frequency Offset'


    if (title is None) or (title == ''):
        title = f'{ylabel} vs {index_label}'

    style_list = []
    for _ in np.arange(len(list_dataframe)):
        style_list.append(next(color_iterator) + next(marker_iterator))

    ax = plt.axes()
    plt.rcParams['text.usetex'] = False
    
    for index in np.arange(len(list_dataframe)):
        result_to_plot = list_dataframe[index]
        result_to_plot.plot(figsize=figsize, ax=ax, kind='line', logy=logy, xlabel=xlabel, ylabel=ylabel, title=title, ls='-', style=style_list[index], ms=7., linewidth=2.2)
    
    ax.grid(which='both')
    ax.grid(which='major', color='black')
    ax.grid(which='minor', color='grey')
    fig = ax.get_figure()
    fig.patch.set_facecolor('w')
    ax.figure.savefig(save_path, format='png', dpi=300, facecolor='w', transparent=False)
    return fig

def subplot_list_dataframe_ber_bler(list_dataframe, ax, save_path, title=None, which='ber', index='ebn0', figsize=(15,10)):
    """
    The dataframe is expected to have all the data to add to the plot.
    The expected structure of the dataframe is as follow:

                SNR     model_1                              
                        ber                 bler (ser, sbler)  sample_offset     
                        mean      95%CI     mean      95%CI    mean
        Eb/N0                                        
        5.0     -5.0    0.006032  0.000507  0.550781  0.03679  1.0
        6.0     -6.0    0.002349  0.000224  0.251953  0.02272  2.0
        7.0     -7.0    0.000763  0.000076  0.093299  0.00904  2.0
        8.0     -8.0    0.000192  0.000019  0.024414  0.00240  1.0

    The index Eb/N0 can vary in size.
    Several models can be present (stacked horizontally).

    Parameters
    ----------
    dataframe : pandas Dataframe
        The input dataframe

    save_path : str
        The full save path with figure file name .png
        
    title : str
        Title of the figure. If None or '' set to ylabel vs xlabel.
    
    which : str, optional
        'ber' or 'bler' or 'ser' or 'sbler' or 'sample_offset' or 'fractional_frequency_offset' to plot, by default 'ber'
    
    index : str, optinal
        'ebn0' or 'snr', index to use, by default 'ebn0'

    Returns
    -------
    matplotlib.pyplot Figure
        The created figure.
    """
    if index == 'snr':
        index_label = 'SNR'
        xlabel = 'SNR (dB)'
    else:
        index_label = 'Eb/N0'
        xlabel = 'Eb/N0 (dB)'

    for index in np.arange(len(list_dataframe)):
        df = list_dataframe[index]
        result_df = df.iloc[:, df.columns.get_level_values(1)==which]
        result_mean = result_df.iloc[:, result_df.columns.get_level_values(2)=='mean']
        result_ci = result_df.iloc[:, result_df.columns.get_level_values(2)=='95%CI']
        result_no_mi =  result_mean.droplevel(level=1, axis=1).droplevel(level=1, axis=1)
        list_dataframe[index] = result_no_mi

    marker_list = ["^", "s", "o", "*", "p", "x", "v", "2", "d"]
    color_list = "bgrcmyk"

    marker_iterator = itertools.cycle(marker_list)
    color_iterator = itertools.cycle(color_list)

    logy = True

    if which == 'ber':
        ylabel = 'BER'
    elif which == 'bler':
        ylabel = 'BLER'
    elif which == 'ser':
        ylabel = 'SER'
    elif which == 'sbler':
        ylabel = 'SBLER'
    elif which == 'sample_offset':
        ylabel = 'Sample Offset'
    elif which == 'fractional_frequency_offset':
        ylabel = 'Fractional Frequency Offset'


    if (title is None) or (title == ''):
        title = f'{ylabel} vs {index_label}'

    style_list = []
    for _ in np.arange(len(list_dataframe)):
        style_list.append(next(color_iterator) + next(marker_iterator))

    plt.rcParams['text.usetex'] = False
    
    for index in np.arange(len(list_dataframe)):
        result_to_plot = list_dataframe[index]
        result_to_plot.plot(figsize=figsize, ax=ax, kind='line', logy=logy, xlabel=xlabel, ylabel=ylabel, title=title, ls='-', style=style_list[index], ms=7., linewidth=2.2)
    
    ax.grid(which='both')
    ax.grid(which='major', color='black')
    ax.grid(which='minor', color='grey')
    ax.figure.savefig(save_path, format='png', dpi=300, facecolor='w', transparent=False)
