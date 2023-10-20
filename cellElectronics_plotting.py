from scipy import ndimage as nd
import numpy as np
import math
import random
try: import axroHFDFCpy.construct_connections as cc
except: import construct_connections as cc
from operator import itemgetter
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FixedLocator
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
plt.rcParams['savefig.facecolor']='white'
from datetime import date
import string
import copy
import pickle

def plot_cell_status_concave(cells, figsize=(10,8), plot_title=None, fontsize=18):
    """
    Takes a list of cell objects and plots visually the status of whether they
    have been measured, are missing, or are shorted when viewed from the CONCAVE
    SIDE OF THE MIRROR.
    This function works off the IF performance of the cells, not the voltage.
    """
    if plot_title is None: # format title with today's date if none is provided
        today_dtype = date.today()
        today = today_dtype.strftime('%y')+'-'+today_dtype.strftime('%m')+'-'+today_dtype.strftime('%d')
        plot_title = 'C1S04 Cell Status '+today
    # calculate how many cells are good, how many are shorted, and how manyh are missing
    good_cells, miss_cells, short_cells = cc.cell_status(cells)
    N_good = len(good_cells)
    N_miss = len(miss_cells)
    N_short = len(short_cells)
    # list that tells us which shorted cells have been plotted to share the same marker
    partner_short_cell_nos = []
    # colors shared by shorted cells
    short_colors = list(mcolors.TABLEAU_COLORS.keys())
    short_colors.pop(short_colors.index('tab:brown'))
    fig, ax = plt.subplots(figsize=figsize)
    ax.invert_yaxis()
    ax.set_title(plot_title, fontsize=fontsize)
    ax.set_ylabel('Row', fontsize=fontsize)
    ax.set_xlabel('Column', fontsize=fontsize)
    # flags for only putting one entry of each type in legend
    first_short, first_miss, first_good = True, True, True
    for i, cell in enumerate(cells):
        marker = "."
        markersize = 240
        if cell.short_cell_no != -1: # cell is shorted
            if cell.no in partner_short_cell_nos:
                # have cell share the same color
                color_idx = partner_short_cell_nos.index(cell.no)
                color = short_colors[color_idx]
            else:
                # use new short marker
                color = short_colors[len(partner_short_cell_nos)]
                partner_short_cell_nos.append(cell.short_cell_no)
            # use cell no as marker
            marker = '${}$'.format(cell.no)
            if first_short: label = 'Measured & shorted ({})'.format(N_short)
            else: label = ''
            if cell.no > 99: markersize *= 1.5
            first_short = False
        elif cell.rms == -1: # cell is missing
            color = 'darkred'
            if first_miss: label = 'Missing ({})'.format(N_miss)
            else: label = ''
            first_miss = False
        else: # cell is not shorted
            color = 'limegreen'
            if first_good: label = 'Measured w/out short ({})'.format(N_good)
            else: label = ''
            first_good = False
        ax.scatter(cell.grid_coord[1], cell.grid_coord[0], marker=marker,
                    s=markersize, color=color, label=label)
    ax.set_yticks([i for i in range(18)])
    ax.set_xticks([i for i in range(16)])
    ax.set_yticklabels([i+1 for i in range(18)], fontsize=fontsize-6)
    ax.set_xticklabels([i+1 for i in range(16)], fontsize=fontsize-6)
    # Put a legend to the right of axis
    fig.legend(bbox_to_anchor=(0.71, 0.5), loc="center left")
    fig.subplots_adjust(right=0.7)
    return fig

def plot_cellTests(cells, date, tvolts=[-10.0, -1.0, 0.0, 1.0, 10.0],
                    title='Cell Test Voltages Response', figsize=None, fontsize=12):
    """
    Plots the voltage history for a list of cell objects that were tested at a
    list of known voltages.
    """
    N_rows = math.ceil(len(tvolts)/2)
    if '_' in date: date = date.replace('_', '')
    if figsize is None: figsize = (12, N_rows*5)
    fig, axs = plt.subplots(N_rows, 2, figsize=figsize)
    if len(tvolts) % 2 != 0: fig.delaxes(axs[-1,-1])
    for i in range(len(tvolts)):
        ax = axs.ravel()[i]
        cell_nos = [cell.no for cell in cells]
        volts = np.array([cell.volt_hist[i] for cell in cells])
        mean_volt = np.mean(volts)
        ax.plot(cell_nos, volts)
        ax.set_ylabel('Voltage (V)', fontsize=fontsize)
        ax.set_xlabel('Cell #', fontsize=fontsize)
        ax.set_title('Applied Voltage: {} V'.format(tvolts[i]), fontsize=fontsize)
        ax.text(0.85, 0.9, 'Mean: {:.2f}V'.format(mean_volt), fontsize=fontsize,
                ha='center', va='center', transform=ax.transAxes)
        ax.grid(axis='y')
        ax.set_ylim([-15,15])
        ax.set_xlim([np.min(cell_nos), np.max(cell_nos)])
    fig.suptitle(title+'\nDate: '+date, fontsize=fontsize+2)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def plot_cellTests_crossVolts(cells, date, tvolts, tdelays=None,
                            title='Cell Test Voltages Response',
                            ylim=None, figsize=None, fontsize=12):
    cells_per_subplot = 72
    N_subplots = math.ceil(len(cells)/cells_per_subplot)
    if '_' in date: date = date.replace('_', '')
    if figsize is None: figsize = (20, 9)
    fig, axs = plt.subplots(N_subplots, 1, figsize=figsize)
    keyword, keyunit = 'voltage', 'V'
    if tdelays is not None: keyword, keyunit = 'delay', 's'
    for i in range(N_subplots):
        ax = axs.ravel()[i]
        subplot_cells = [cell for cell in cells[cells_per_subplot*i:cells_per_subplot*(i+1)]]
        cell_nos = [cell.no for cell in subplot_cells]
        ax.set_ylabel('Voltage (V)', fontsize=fontsize)
        for j in range(len(tvolts)):
            volts = [cell.volt_hist[j] for cell in subplot_cells]
            legend_val = tvolts[j]
            if tdelays is not None: legend_val = tdelays[j]
            ax.plot(cell_nos, volts, marker='.',
                    label='Applied {}: {:.2f} {}, Mean voltage: {:.2f} V'\
                    .format(keyword, legend_val, keyunit, np.mean(volts)))
        if i == 0:
            if len(tvolts) > 4:
                ncols = math.floor(len(tvolts)/2)
                vert_pos = 2.00
                vert_gap = 1
            else:
                ncols = len(tvolts)
                vert_pos = 1.35
                vert_gap = 0.99
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, vert_pos), ncols=ncols,
            fancybox=True, shadow=True, fontsize=fontsize)
        ax.set_xticks(cell_nos)
        ax.set_xticklabels(cell_nos, fontsize=fontsize, rotation=45)
        ax.grid(axis='x')
        all_volts = []
        for cell in cells:
            all_volts += cell.volt_hist
        if ylim is None: ylim = [np.min(all_volts)-0.5, np.max(all_volts)+0.5]
        ax.set_ylim(ylim)
        ax.set_xlim([np.min(cell_nos), np.max(cell_nos)])
    ax.set_xlabel('Cell #', fontsize=fontsize)
    fig.suptitle(title+'\nDate: '+date, fontsize=fontsize+2)
    fig.tight_layout(rect=[0, 0, 1, vert_gap])
    return fig

def percent_diff(x1, x2):
    return ((np.abs(x1-x2)) / (np.mean([x1,x2]))) * 100

def plot_voltageCompare_wDMM(cells, date, tvolts, figsize=None, fontsize=12,
                             title='Cell Test Voltages Compared With DMM\nDiconnected from Mirror'):
    """
    Plot voltages tested using the boards vs those measured with a DMM. Each cell object in cells
    should have cell.volt_hist = [boardVal_1, boardVal_2, ..., boardVal_n, DMMVal_1, DMMVal_2, ..., DMMVal_n]
    where boardVal_i and DMMVal_i correspond to the same voltage that was applied/tested, and matches tvolts[i].
    """
    N_subplots = int(len(cells[0].volt_hist) / 2)
    if '_' in date: date = date.replace('_', '')
    if figsize is None: figsize = (6, 8)
    cell_nos = [cell.no for cell in cells]
    fig, axs = plt.subplots(N_subplots, 1, figsize=figsize)
    for i in range(N_subplots):
        ax = axs.ravel()[i]
        ax.axhline(tvolts[i], color='mediumpurple', linestyle='dashed')
        boardVals, DMMVals = [], []
        xvals = [i for i in range(len(cell_nos))]
        for cell in cells:
            boardVals.append(cell.volt_hist[i])
            DMMVals.append(cell.volt_hist[N_subplots:][i])
        ax.plot(xvals, boardVals, color='dodgerblue', marker='.', linestyle='solid', label='Board measurement')
        ax.plot(xvals, DMMVals, color='firebrick', marker='^', linestyle='solid', label='DMM measurement')
        if i == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.65), ncol=2,
            fancybox=True, shadow=True)
        ax.text(1.0, 1.0, 'Diff of mean voltages: {:.2f} V'.format(np.mean(boardVals)-np.mean(DMMVals)),
               ha='right', va='bottom', transform=ax.transAxes, fontsize=fontsize-2)
        ax.set_ylabel('Voltage (V)')
        ax.set_title('{} V Test'.format(tvolts[i]), loc='left', fontsize=fontsize)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True)
        ax.set_xticks(xvals)
        ax.set_xticklabels(cell_nos, fontsize=fontsize)
        subplot_volts = boardVals+DMMVals+[tvolts[i]]
        ax.set_ylim([np.min(subplot_volts)-.01, np.max(subplot_volts)+.01])
#         ax.set_xlim([np.min(cell_nos), np.max(cell_nos)])
    ax.set_xlabel('Cell #', fontsize=fontsize)
    fig.suptitle(title+'\nDate: '+date, fontsize=fontsize+2)
    fig.tight_layout(rect=[0, 0, 1, 1.025])
    return fig

def plot_short_test(cells, date, tvolt, thresh=0.2, title='Cell Shorted Groups',
                    figsize=None, fontsize=12):
    short_groups = get_short_groups(cells, thresh=thresh)
    N_subplots = len(short_groups) # plot shorted groups
    if N_subplots == 0:
        print("Didn't find any short groups!")
        return None
    if '_' in date: date = date.replace('_', '')
    if figsize is None: figsize = (6, 14)
    fig, axs = plt.subplots(N_subplots, 1, figsize=figsize)
    for i in range(N_subplots):
        ax = axs.ravel()[i]
        short_group = short_groups[i][1]
        for j in range(short_group.shape[1]):
            ax.plot(j+1, short_group[1][j], color='blue', linestyle='dashed',
                    marker='.', markersize=8)
        ax.set_ylabel('Voltage (V)', fontsize=fontsize)
        ax.set_title('Tested Cell: {}'.format(short_groups[i][0]), fontsize=fontsize)
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        xvals = [i+1 for i in range(short_group.shape[1])]
        ax.set_xticks(xvals)
        xticklabels = [int(cell_no) for cell_no in short_group[0]]
        ax.set_xticklabels(xticklabels, fontsize=fontsize)
        subplot_volts = short_group[1]
        ax.set_ylim([np.min(subplot_volts)-.5, np.max(subplot_volts)+.5])
    ax.set_xlabel('Cell #', fontsize=fontsize)
    fig.suptitle(title+'\nApplied Voltage {:.2f} V'.format(tvolt)\
    +'\nVoltage Threshold to Detect Short: {:.2f} V'.format(thresh)\
    +'\nDate: '+date, fontsize=fontsize+2)
    fig.subplots_adjust(hspace=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

def get_short_groups(cells, thresh=0.2):
        """
        Use this after running ce.test_for_shorts() on the cells to get short_groups:
        short_groups is a list of lists:
        * short_groups[i] indexes the short group
        * short_groups[i][0] is the cell number that was tested when the short group was measured
        * short_group[i][1] is the short group. It is array of size (2, n), where n is the number of cells
        that are included in the short.
        The top row of the array is the cell nums, the 2nd row is their corresponding voltages
        * thresh: the voltage above ground that a grounded cell needs
        to be at to be registered as shorted
        """
        short_groups = []
        for cell in cells: # extract shorted groups from cell objects
            # print('====================CELL: {}===================='.format(cell.no))
            cell_volt_hist = np.array(cell.volt_hist)
            high_volts_args = np.argwhere(cell_volt_hist>thresh)
            # print('I found that these cell nums were above the thresh:', high_volts_args.flatten()+1)
            if high_volts_args.size > 1: # if there is a shorted group
                extract_volts_formatted = cell_volt_hist[high_volts_args].reshape(high_volts_args.size)
                # print('Here are their voltages:', extract_volts_formatted.flatten())
                high_cell_nos = high_volts_args.reshape(high_volts_args.size) + 1 # convert from cell index to cell no
                add_to_short_groups = True
                for sub_ls in short_groups:
                    if not sub_ls: continue
                    else: short_group = sub_ls[1]
                    if short_group[0].size == high_cell_nos.size: # shorted group is unique if it's a different size than all other groups
                        sorted_short_nos = np.sort(short_group[0]) # compare the cell numbers of the past group with
                        sorted_high_nos = np.sort(high_cell_nos) # with the cell numbers of the current group
                        if np.all(np.equal(sorted_short_nos, sorted_high_nos)):
                            add_to_short_groups = False
                            # print('\nI have found that the cell groups\n{} and\n{} are the same.'.format(sorted_short_nos, high_cell_nos))
                if add_to_short_groups:
                    short_group = np.stack((high_cell_nos, extract_volts_formatted), axis=0)
                    short_groups.append([cell.no, short_group])
                    # print('I added to short groups')
        return short_groups

def plot_cell_status_convex(cells=[], short_cells=[], date='', short_group_thresh=0.2,
                            badCell_thresh=9.5, badIF_cell_nos=np.array([]),
                            figsize=None, title_fontsize=16, ax_fontsize=14,
                            global_title='C1S04 Cell Voltage Status Viewed\nFrom Back (Non-reflective) Side',
                            cbar_title='Region / Cell Status', badCell_label=None,
                            shortCell_label=None, badIF_label=None, legend_coords=(1.28, -0.01)):
    """
    Takes a list of cell objects and plots visually the status of whether they
    have been measured, are missing, or are shorted when viewed from the CONVEX
    SIDE OF THE MIRROR.
    This function works off the voltage performance of the cells, not the IF.

    To produce this plot, you need to do the following:
    * run a test of the boards using the new method at tvolt=10. The list of cells
    you get back => "cells" in this function.
    * run test_for_shorts at tvolt=10. The list of cells you get back  => "short_cells"
    in this function.

    Other parameters:
    * short_group_thresh: the voltage above ground that a grounded cell needs
    to be at to be registered as shorted
    * badCell_thresh: high cell voltages < badCell_thresh are registered as being bad
    """
    if badIF_cell_nos is None: badIF_cell_nos = np.array([])
    if short_cells:
        short_groups = get_short_groups(short_cells, thresh=short_group_thresh) # get the short groups from the cells that were tested for shorts
        short_cell_nums = np.concatenate([short_groups[i][1][0] for i in \
                                        range(len(short_groups))], axis=0)
    else:
        short_groups, short_cell_nums = np.array([]), np.array([])
    if cells:
        nonShort_badCells = cc.get_nonShort_badCells(cells, short_groups,
                                                    thresh=badCell_thresh) # isolate the bad cells not part of a short group
    else:
        cells = cc.construct_cells()
        nonShort_badCells, short_cell_nums = np.array([]), np.array([])

    # set colorbar cell status labels
    if badCell_label is None: badCell_label = 'Dead Cell: < {} V'.format(badCell_thresh)
    if shortCell_label is None: shortCell_label = 'Shorted Cell: > {} V'.format(short_group_thresh)
    if badIF_label is None: badIF_label = 'Bad IF'
    cbar_labels, cmap, cbar_label_vals, cbar_tick_bounds, labelpad = \
    get_colorbar_properties(short_cell_nums, nonShort_badCells, short_group_thresh,
                            badCell_thresh, badCell_label, shortCell_label)
    cell_value_array = np.zeros(cc.cell_order_array.shape)
    row_nums = np.array([i for i in range(cell_value_array.shape[0])])
    col_nums = np.array([i for i in range(cell_value_array.shape[1])])
    print('Number of shorted cells:', len(short_cell_nums))
    short_colors = list(copy.deepcopy(mcolors.TABLEAU_COLORS).keys()) + \
                    list(copy.deepcopy(mcolors.BASE_COLORS).keys())
    short_color_count = 0
    if not figsize: figsize = (8, 8)
    if date is not None and '_' in date: date = date.replace('_', '')

    fig, ax = plt.subplots(figsize=figsize)
    hline_locs = (row_nums[1:]+row_nums[:-1]) / 2
    vline_locs = (col_nums[1:]+col_nums[:-1]) / 2
    for loc in hline_locs: ax.axhline(loc, color='black')
    for loc in vline_locs: ax.axvline(loc, color='black')
    for cell in cells:
        args = np.argwhere(cc.cell_order_array==cell.no)[0]
        row, col = args[0], args[1]
        # if (cell.no in badIF_cell_nos) and (cell.no not in nonShort_badCells)\
        # and (cell.no not in short_cell)
        if cell.no in nonShort_badCells: # if the cell is bad but not shorted
            cell_value_array[row][col] = cbar_labels.index(badCell_label)
            ax.text(col, row, int(cell.no), color='white', ha='center', va='center',
                    fontsize=ax_fontsize-6, zorder=5.)
        elif cell.no in short_cell_nums: # if the cell is in a shorted group
            cell_value_array[row][col] = cbar_labels.index(shortCell_label)
            for i in range(len(short_groups)):
                if cell.no in short_groups[i][1][0]:
                    ax.text(col, row, int(cell.no), color=short_colors[i],
                            ha='center', va='center', fontsize=ax_fontsize-6)
                    ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1,
                                edgecolor=short_colors[i], facecolor="None", lw=1.5,
                                zorder=2.0))
                    break
                else: continue
        else: # if the cell is good
            cell_value_array[row][col] = cbar_labels.index(cell.region)
            ax.text(col, row, int(cell.no), color='black', ha='center', va='center',
                    fontsize=ax_fontsize-6, zorder=5.)

    im = ax.imshow(cell_value_array, aspect='equal', cmap=cmap)
    # mark the cells that have bad IFs, but do not have bad voltages and are not shorted
    if len(badIF_cell_nos > 0): mark_badIF_cells(badIF_cell_nos, nonShort_badCells,
                                                short_cell_nums, badIF_label, ax, ax_fontsize,
                                                legend_coords)
    # label axes
    ax.set_xlabel('Column', fontsize=ax_fontsize)
    ax.set_ylabel('Row', fontsize=ax_fontsize)
    if date: global_title += '\nDate: {}'.format(date)
    ax.set_title(global_title, fontsize=title_fontsize)
    # attach static colorbar
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(im, cax=cax, boundaries=cbar_tick_bounds, ticks=cbar_label_vals)
    cbar.set_label(cbar_title, fontsize=ax_fontsize, labelpad=labelpad)
    cbar.ax.set_yticklabels(cbar_labels, fontsize=ax_fontsize)
    fig.subplots_adjust(top=0.85, hspace=0.5, wspace=0.8)
    # adjust tick labels
    ax.set_yticks(row_nums)
    ax.set_xticks(col_nums)
    ax.set_yticklabels(row_nums+1)
    ax.set_xticklabels(col_nums+1)
    return fig

def get_colorbar_properties(short_cell_nums, nonShort_badCells, short_group_thresh,
                            badCell_thresh, badCell_label, shortCell_label):
    cbar_labels = [shortCell_label, badCell_label,
                'RD', 'RC', 'RB', 'RA', 'LD', 'LC', 'LB', 'LA']
    colors = ['white', 'black', 'mediumturquoise', 'pink', 'chocolate',
                            'yellow', 'mediumorchid', 'limegreen', 'dodgerblue', 'red']
    # cmap = ListedColormap(['white', 'black', 'silver', 'pink', 'chocolate',
    #                         'yellow', 'mediumorchid', 'limegreen', 'dodgerblue', 'red'])
    if np.any(short_cell_nums) and np.any(nonShort_badCells):
        print('You got shorts and bad cells.')
        labelpad = -40
    elif np.any(short_cell_nums) and not np.any(nonShort_badCells):
        print('You got shorts but not bad cells.')
        cbar_labels.pop(1)
        colors.pop(1)
        labelpad = -10
    elif not np.any(short_cell_nums) and np.any(nonShort_badCells):
        print('You got no shorts but got bad cells.')
        cbar_labels.pop(0)
        colors.pop(0)
        labelpad = -40
    else:
        print('You got no shorts and no bad cells.')
        cbar_labels.pop(0)
        colors.pop(0)
        cbar_labels.pop(0)
        colors.pop(0)
        labelpad = 10
    cmap = ListedColormap(colors)
    cbar_label_vals = np.arange(len(cbar_labels))
    cbar_tick_bounds = np.arange(cbar_label_vals[0]-0.5, cbar_label_vals[-1]+1, 1)
    # print('ticks:', cbar_label_vals)
    # print('bounds:', cbar_tick_bounds)
    return cbar_labels, cmap, cbar_label_vals, cbar_tick_bounds, labelpad

def mark_badIF_cells(badIF_cell_nos, nonShort_badCells, short_cell_nums, badIF_label,
                    ax, ax_fontsize, legend_coords):
    y_coords, x_coords = [], []
    # reduced_badIF_cell_nos = np.where(badIF_cell_nos!=nonShort_badCells)
    # print('Bad IF cells:\n{}'.format(badIF_cell_nos))
    # print('Non-short bad cells:\n{}'.format(nonShort_badCells))
    # print('Short cell nos:\n{}'.format(short_cell_nums))
    # print('reduced bad IF cell nos:\n{}'.format(reduced_badIF_cell_nos))
    for cell_no in badIF_cell_nos:
        if (cell_no not in nonShort_badCells) and (cell_no not in short_cell_nums):
            args = np.argwhere(cc.cell_order_array==cell_no)[0]
            y_coords.append(args[0])
            x_coords.append(args[1])
    ax.plot(x_coords, y_coords, color='darkgrey', marker="X", label=badIF_label,
            linestyle="None", markersize=20, zorder=1.)
    ax.legend(loc='upper right', bbox_to_anchor=(legend_coords[0], legend_coords[1]),
                fancybox=True, fontsize=ax_fontsize)
    # ax.legend(fontsize=ax_fontsize)
