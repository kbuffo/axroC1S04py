from scipy import ndimage as nd
import numpy as np
import math
from operator import itemgetter
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
import matplotlib.colors as mcolors
plt.rcParams['savefig.facecolor']='white'

import utilities.figure_plotting as fp
import imaging.man as man
import imaging.analysis as alsis
import axroOptimization.evaluateMirrors as eva
import axroOptimization.solver as solver

from datetime import date
import string

def printer():
    print('Hello construct connections!')

############### Constructing cell, pin, and cable structure ####################
topBoard = 3 # YOU NEED TO CHANGE THESE IF YOU SWAP OUT THE BOARDS IN THE ENCLOSURE
bottomBoard = 1
N_cells = 288 # number of cells
N_rows = 18 # number of rows (axial dimension)
N_cols = 16 # number of columns (azimuthal dimension)
N_ppins = 104 # number of pins on a PV cable and a PA cable
N_p0xv_pins = 51 # number of pins on a P0XV cable (these connect to the mirror)
N_p0xa_pins = 78 # number of pins on a P0XA cable (these connect to boards 1 & 3)
N_region_pins = 40 # number of pins per region
# N_j_ports = 4 # number of J ports per board
N_j_pins = 78 # number of pins on a J port
N_aout_pins = 256 # number of AOUT pins that connect to J pins
N_channels = 32 # number of channels per DAC

cell_nums = np.arange(1, N_cells+1, 1) # array of cell numbers
rows = np.arange(0, N_rows, 1) # array of cell rows
cols = np.arange(0, N_cols, 1) # array of cell columns
# list of grid coordinates for each cell in the standard format
grid_coords = [[rows[(N_rows-1)-(i%N_rows)],cols[i//N_rows]] for i in range(N_cells)]
# for i in grid_coords:
#     i[0] *= -1
# grid_coords = sorted(grid_coords, key=lambda k: [k[0], k[1]])
# for i in grid_coords:
#     i[0] *= -1
# grid_coords = sorted(grid_coords, key=itemgetter(0), reverse=True)
# print(grid_coords)
# list of row-column index lables for each cell in the standard format
rc_labels = ['R'+str(coord[0]+1)+'C'+str(coord[1]+1) for coord in grid_coords]
# construct the list of cells in each region in the pinout format
concs = ['GND', 'AOUT-XX', 'AGND', 'NC']
la_cells = [concs[0], concs[1]] + cell_nums[0:32].tolist() + [concs[1]]*7 + [concs[2]] + [concs[3]]*9
lb_cells = [concs[2]] + cell_nums[32:72].tolist() + [concs[2]] + [concs[3]]*9
lc_cells = [concs[2]] + cell_nums[72:111].tolist() + [concs[1], concs[2]] + [concs[3]]*9
ld_cells = [concs[2]] + cell_nums[111:144].tolist() + [concs[1]]*7 + [concs[2]] + [concs[3]]*9
ra_cells = [concs[2]] + [concs[1]]*7 + np.flip(cell_nums[144:176]).tolist() + [concs[1], concs[0]] + [concs[3]]*9
rb_cells = [concs[1]] + [int(cell_nums[215])] + np.flip(cell_nums[176:215]).tolist() + [concs[2]] + [concs[3]]*9
rc_cells = [concs[2], concs[1]] + np.flip(cell_nums[216:255]).tolist() + [concs[2]] + [concs[3]]*9
rd_cells = [concs[2]] + [concs[1]]*7 + np.flip(cell_nums[255:288]).tolist() + [concs[2]] + [concs[3]]*9
# dictionary of regions that connects to each region's cells and the corresponding
# cables connected to that region
regions = {'LA':[la_cells, 'P1V', 'P01V', 'P1A'], 'LB':[lb_cells, 'P1V', 'P02V', 'P1A'],
           'LC':[lc_cells, 'P2V', 'P03V', 'P2A'], 'LD':[ld_cells, 'P2V', 'P04V', 'P2A'],
           'RA':[ra_cells, 'P3V', 'P05V', 'P3A'], 'RB':[rb_cells, 'P3V', 'P06V', 'P3A'],
           'RC':[rc_cells, 'P4V', 'P07V', 'P4A'], 'RD':[rd_cells, 'P4V', 'P08V', 'P4A']}

# create array that orders cells based on pin diagram
l_array = np.zeros((N_rows, N_cols))
l_array[0][:6] = np.flip(cell_nums[:6]) # LA region
l_array[0][6:8] = cell_nums[6:8]
l_array[1][:8] = np.flip(cell_nums[8:16])
l_array[2][:8] = np.flip(cell_nums[16:24])
l_array[3][:8] = np.flip(cell_nums[24:32])

l_array[4][1:3] = np.flip(cell_nums[32:34]) # LB region
l_array[4][0] = np.flip(cell_nums[34])
l_array[4][3:8] = cell_nums[35:40]
l_array[5][:8] = cell_nums[40:48]
l_array[6][:8] = cell_nums[48:56]
l_array[7][:4] = np.flip(cell_nums[56:60])
l_array[7][4:8] = cell_nums[60:64]
l_array[8][:6] = np.flip(cell_nums[64:70])
l_array[8][6:8] = cell_nums[70:72]

l_array[9][:8] = np.flip(cell_nums[72:80]) # LC region
l_array[10][:8] = np.flip(cell_nums[80:88])
l_array[11][:8] = np.flip(cell_nums[88:96])
l_array[12][:8] = np.flip(cell_nums[96:104])
l_array[13][:8] = np.flip(cell_nums[104:112])

l_array[14][:8] = cell_nums[112:120] # LD region
l_array[15][:8] = cell_nums[120:128]
l_array[16][:8] = cell_nums[128:136]
l_array[17][6:8] = np.flip(cell_nums[136:138])
l_array[17][:6] = cell_nums[138:144]

flip_l_array = np.fliplr(l_array) # generate R region
r_array = np.where(flip_l_array>0, flip_l_array+144, flip_l_array)

cell_pin_array = l_array + r_array

# create array that orders cells using standard format
cell_order_array = np.zeros((N_rows, N_cols))
# print(cell_order_array)
for i in range(N_cells):
    y, x = grid_coords[i][0], grid_coords[i][1]
    cell_order_array[y][x] = cell_nums[i]


############################## VAC side cables ##############################

# construct the list of pin numbers in the pinout format
p0xv_pins = np.arange(1, N_p0xv_pins+1, 1)
pv_odd_pins = ([i+1 for i in range(9)] + [86] + [i+1 for i in range(21, 29)] +
           [i+1 for i in range(43, 51)] + [i+1 for i in range(63, 71)] + [i+1 for i in range(86, 93)] +
           [43])
pv_even_pins = ([i+1 for i in range(10, 19)] + [96] + [i+1 for i in range(31, 39)] +
               [i+1 for i in range(53, 61)] + [i+1 for i in range(73, 81)] +
               [i+1 for i in range(96, 103)] +[53])
pv_odd_pins.extend([concs[3]]*(len(p0xv_pins)-len(pv_odd_pins)))
pv_even_pins.extend([concs[3]]*(len(p0xv_pins)-len(pv_even_pins)))

vac_cables = {'P1V' : {'P01V':[p0xv_pins, pv_odd_pins], 'P02V':[p0xv_pins, pv_even_pins]},
          'P2V' : {'P03V':[p0xv_pins, pv_odd_pins], 'P04V':[p0xv_pins, pv_even_pins]},
          'P3V' : {'P05V':[p0xv_pins, pv_odd_pins], 'P06V':[p0xv_pins, pv_even_pins]},
          'P4V' : {'P07V':[p0xv_pins, pv_odd_pins], 'P08V':[p0xv_pins, pv_even_pins]}}

############################## AIR side cables ##############################
p0xa_pins = np.arange(1, N_p0xa_pins+1, 1)
pa_odd_pins = ([i+1 for i in range(10)] + [i+1 for i in range(21, 31)] +
                [i+1 for i in range(42, 52)] + [i+1 for i in range(63, 73)] +
                [i+1 for i in range(84, 94)])
pa_even_pins = ([i+1 for i in range(10, 20)] + [i+1 for i in range(31, 41)] +
                [i+1 for i in range(52, 62)] + [i+1 for i in range(73, 83)] +
                [i+1 for i in range(94, 104)])
pa_odd_pins.extend([concs[3]]*(len(p0xa_pins)-len(pa_odd_pins)))
pa_even_pins.extend([concs[3]]*(len(p0xa_pins)-len(pa_even_pins)))

air_cables = {'P1A' : {
                        'P01A':[p0xa_pins, pa_odd_pins, 'J2'],
                        'P02A':[p0xa_pins, pa_even_pins, 'J5'],
                        'Board':topBoard
                        },
             'P2A' : {
                        'P03A':[p0xa_pins, pa_odd_pins, 'J7'],
                        'P04A':[p0xa_pins, pa_even_pins, 'J9'],
                        'Board':topBoard
                        },
             'P3A' : {
                        'P05A':[p0xa_pins, pa_odd_pins, 'J2'],
                        'P06A':[p0xa_pins, pa_even_pins, 'J5'],
                        'Board':bottomBoard
                        },
             'P4A' : {
                        'P07A':[p0xa_pins, pa_odd_pins, 'J7'],
                        'P08A':[p0xa_pins, pa_even_pins, 'J9'],
                        'Board':bottomBoard
                        }}

############################## Board Layout ##############################
j_pins = [i+1 for i in range(N_j_pins)]
j_gnd_pins = [1, 10, 19, 20, 21, 30, 39, 40, 41, 50, 59, 60, 69, 78] # J pins that are grounded
for pin in j_gnd_pins: # remove the grounded J pins from the list of J pins
    j_pins.remove(pin)
j2_aout_pins = (
                [i for i in range(8)] + [i for i in range(32, 40)] + [i for i in range(16, 24)] + # J2
                [i for i in range(48, 56)] + [i for i in range(8, 16)] + [i for i in range(40, 48)] +
                [i for i in range(24, 32)] + [i for i in range(56, 64)]
                )
j5_aout_pins = (
                [i for i in range(64, 72)] + [i for i in range(96, 104)] + [i for i in range(80, 88)] + # J5
                [i for i in range(112, 120)] + [i for i in range(72, 80)] + [i for i in range(104, 112)] +
                [i for i in range(88, 96)] + [i for i in range(120, 128)]
                )
j7_aout_pins = (
                [i for i in range(128, 136)] + [i for i in range(160, 168)] + [i for i in range(144, 152)] + # J7
                [i for i in range(176, 184)] + [i for i in range(136, 144)] + [i for i in range(168, 176)] +
                [i for i in range(152, 160)] + [i for i in range(184, 192)]
                )
j9_aout_pins = (
                [i for i in range(192, 200)] + [i for i in range(224, 232)] + [i for i in range(208, 216)] + # J9
                [i for i in range(240, 248)] + [i for i in range(200, 208)] + [i for i in range(232, 240)] +
                [i for i in range(216, 224)] + [i for i in range(248, N_aout_pins)]
                )
channels = [i for i in range(N_channels)]
board_dict = {'J2' : {
                        0:[i for i in range(32)], # AOUT pins for each DAC
                        1:[i for i in range(32, 64)],
                        'j_aout_pins':j2_aout_pins
                        },
             'J5' : {
                        2:[i for i in range(64, 96)],
                        3:[i for i in range(96, 128)],
                        'j_aout_pins':j5_aout_pins
                        },
             'J7' : {
                        4:[i for i in range(128, 160)],
                        5:[i for i in range(160, 192)],
                        'j_aout_pins':j7_aout_pins
                        },
             'J9' : {
                        6:[i for i in range(192, 224)],
                        7:[i for i in range(224, N_aout_pins)],
                        'j_aout_pins':j9_aout_pins
                        }}

##################### Cell constructor functions ################################

def construct_cells(N_cells=N_cells, grid_coords=grid_coords,
                    cell_pin_array=cell_pin_array, regions=regions,
                    rc_labels=rc_labels):
    """
    Generates a list of cell objects ordered in the standard format.
    Each cell object has the attributes of the Cell class.
    """
    cells = []
    for i in range(N_cells): # for each cell...
        y, x = grid_coords[i][0], grid_coords[i][1] # find the cell's coordinates in the standard format
        matching_pin_cell = int(cell_pin_array[y][x]) # find the corresponding pin cell no. at those coordinates

        for region, region_ls in regions.items(): # iterate through each region of the mirror
            region_cells = region_ls[0] # get the list of pin cell nos in that region
            if matching_pin_cell in region_cells: # if the corresponding pin cell no is located in that region...
                idx_v = region_cells.index(matching_pin_cell)
                pv_cable_key, p0xv_cable_key, pa_cable_key = region_ls[1], region_ls[2], region_ls[3] # get the p cable and p0x cable it's connected to
                pv_pin = vac_cables[pv_cable_key][p0xv_cable_key][1][idx_v] # find the pins of each cable it's connected to
                p0xv_pin = vac_cables[pv_cable_key][p0xv_cable_key][0][idx_v]
                pa_pin = pv_pin

                for p0xa_cable_key, pin_ls in air_cables[pa_cable_key].items():
                    if p0xa_cable_key == 'Board': continue
                    pa_pins = pin_ls[1]
                    if pa_pin in pa_pins:
                        idx_a = pa_pins.index(pa_pin)
                        p0xa_pin = pin_ls[0][idx_a]
                        j_pin = p0xa_pin
                        j_port = pin_ls[2]
                        p0xa_cable = p0xa_cable_key
                        board_num = air_cables[pa_cable_key]['Board']
                        j_idx = j_pins.index(j_pin)
                        aout_pin = board_dict[j_port]['j_aout_pins'][j_idx]

                        for dac, pins in board_dict[j_port].items():
                            if dac == 'j_aout_pins': continue
                            if aout_pin in pins:
                                ch_idx = pins.index(aout_pin)
                                channel = channels[ch_idx]
                                dacy = dac
                                # create a cell object with all the attributes
                                cell_ls = [i, i+1, grid_coords[i], rc_labels[i], region,
                                            matching_pin_cell,
                                            p0xv_cable_key, p0xv_pin,
                                            pv_cable_key, pv_pin,
                                            pa_cable_key, pa_pin,
                                            p0xa_cable_key, p0xa_pin,
                                            j_pin, j_port, aout_pin,
                                            channel, dac, board_num]
                                cells.append(Cell(cell_ls))

                            else: continue

                    else: continue

            else: continue
            # print('cell:', i+1)
            # print('grid_coords:', grid_coords[i])
            # print('rc_label:', rc_labels[i])
            # print('region:', region)
            # print('pin cell:', matching_pin_cell)
            # print('p0xv cable:', p0xv_cable_key, 'p0xv_pin:', p0xv_pin)
            # print('pv cable:', pv_cable_key, 'pv_pin:', pv_pin)
            # print('pa cable:', pa_cable_key, 'pa_pin:', pa_pin)
            # print('p0xa cable:', p0xa_cable, 'p0xa_pin:', p0xa_pin)
            # print('J pin:', j_pin)
            # print('J port:', j_port)
            # print('aout pin:', aout_pin)
            # print('DAC:', dacy)
            # print('channel:', channel)
            # print('board:', board_num)
            # print('==============================================================')

    return cells

class Cell:

    def __init__(self, cell_ls):
        self.idx = cell_ls[0]
        self.no = cell_ls[1]
        self.grid_coord = cell_ls[2]
        self.rc_label = cell_ls[3]
        self.region = cell_ls[4]
        self.pin_cell_no = cell_ls[5]
        self.p0xv_cable = cell_ls[6]
        self.p0xv_pin = cell_ls[7]
        self.pv_cable = cell_ls[8]
        self.pv_pin = cell_ls[9]
        self.pa_cable = cell_ls[10]
        self.pa_pin = cell_ls[11]
        self.p0xa_cable = cell_ls[12]
        self.p0xa_pin = cell_ls[13]
        self.j_pin = cell_ls[14]
        self.j_port = cell_ls[15]
        self.aout_pin = cell_ls[16]
        self.channel = cell_ls[17]
        self.dac_num = cell_ls[18]
        self.board_num = cell_ls[19]
        self.ifunc = np.array(None)
        self.rms = -1
        self.pv = -1
        self.maxInd = None
        self.short_cell_no = -1
        self.voltage = None
        self.volt_hist = []
    def add_if(self, ifunc):
        self.ifunc = ifunc
        self.rms = alsis.rms(ifunc)
        self.pv = alsis.ptov(ifunc)

    def add_maxInd(self, maxInd, short_cell_no):
        self.maxInd = maxInd
        if short_cell_no is not None: self.short_cell_no = short_cell_no


##################### Cell utility functions #################################

def print_cells_info(cells):
    """
    Print all the attributes of a list of cell objects.
    """
    for cell in cells:
        print('===============CELL #: {}================='.format(cell.no))
        print('Index: {}'.format(cell.idx))
        print('---------------Location---------------')
        print('Grid Coords: {}, RC Label: {}'.format(cell.grid_coord, cell.rc_label))
        print('Region:      {},     Pin Cell #: {}'.format(cell.region, cell.pin_cell_no))
        print('maxInd:\n {}\nShorted Cell #: {}'.format(cell.maxInd, int(cell.short_cell_no)))
        print('----------------Cables----------------')
        print('P0XV Cable, Pin: {}'.format([cell.p0xv_cable, cell.p0xv_pin]))
        print('PV   Cable, Pin: {}'.format([cell.pv_cable, cell.pv_pin]))
        print('PA   Cable, Pin: {}'.format([cell.pa_cable, cell.pa_pin]))
        print('P0XA Cable, Pin: {}'.format([cell.p0xa_cable, cell.p0xa_pin]))
        print('J    Port,  Pin: {}'.format([cell.j_port, cell.j_pin]))
        print('AOUT Pin: {}'.format(cell.aout_pin))
        print('----------------Control-----------------')
        print('BRD: {}, DAC: {}, CH: {}'.format(cell.board_num, cell.dac_num, cell.channel))
        print('------------------IF--------------------')
        print('IF Shape: {}, RMS: {:.2f} um, PV: {:.2f} um'.format(cell.ifunc.shape, cell.rms, cell.pv))
        print('=========================================\n')

def cells_to_array(cells):
    """
    Takes a list of cell objects and returns an 3D array of their IFs in
    the same order as the cell object list.
    """
    return np.stack([cell.ifunc for cell in cells], axis=0)

def add_IFs_to_cells_wCellnos(cell_nos, ifs, cells_ls):
    """
    Takes an array of cell numbers that match a corresponding 3D array of IFs and
    adds those IFs to an existing list of cell objects.
    """
    # generate a list of all the cell numbers from the cell object list
    cell_full_nos = np.array([cell.no for cell in cells_ls])
    for i, cell_no in enumerate(cell_nos):
        # find the current cell no in the list of cell object
        cell_no_idx = np.argwhere(cell_no==cell_full_nos)[0][0]
        cells_ls[cell_no_idx].add_if(ifs[i]) # add the IF to the cell object

def add_maxInds_to_cells_wCellnos(cell_nos, maxInds, cells_ls):
    """
    Takes an array of cell numbers that match a corresponding 3D array of maxInds and
    adds those maxInds to an existing list of cell objects. Finds which cells are shorted
    and adds that to the cells' attributes.
    """
    # generate a list of all the cell numbers from the cell object list
    cell_full_nos = np.array([cell.no for cell in cells_ls])
    for i, cell_no in enumerate(cell_nos):
        # print('Cell:', cell_no)
        # find the current cell no in the list of cell objects
        cell_no_idx = np.argwhere(cell_no==cell_full_nos)[0][0]
        short_cell_no = -1
        if -1 not in maxInds[i][1]: # if there is a short
            # print('Cell {} has a short'.format(cell_no))
            # get primary maxInds as rows of an array
            primary_maxInds = maxInds[:, 0, :]
            for j in range(primary_maxInds.shape[0]):
                # check for matching primary maxInd given shorted maxInd
                if np.all(maxInds[i][1]==primary_maxInds[j]):
                    short_cell_no = cell_nos[j]
                    # print('Cell {}s shorted partner is {}'.format(cell_no, short_cell_no))
        # print('-----------------------')
        cells_ls[cell_no_idx].add_maxInd(maxInds[i], short_cell_no)

def cell_status(cells):
    """
    Returns a list of measured cell objects without shorts, a list of missing
    cell objects, and a list of shorted cell objects. You must add IFs to list
    of cell objects before using this function.
    """
    meas_cells = [cell for cell in cells if cell.rms != -1 and cell.short_cell_no == -1]
    miss_cells = [cell for cell in cells if cell.rms == -1]
    short_cells = [cell for cell in cells if cell.short_cell_no != -1]
    return meas_cells, miss_cells, short_cells

def plot_cell_status(cells, figsize=(10,8), plot_title=None, fontsize=18):
    """
    Takes a list of cell objects and plots visually the status of whether they
    have been measured, are missing, or are shorted.
    """
    if plot_title is None: # format title with today's date if none is provided
        today_dtype = date.today()
        today = today_dtype.strftime('%y')+'-'+today_dtype.strftime('%m')+'-'+today_dtype.strftime('%d')
        plot_title = 'C1S04 Cell Status '+today
    # calculate how many cells are good, how many are shorted, and how manyh are missing
    good_cells, miss_cells, short_cells = cell_status(cells)
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

##################### Old functions #################################
# def display_title_cell_no(ax, title, cell_no, fntsz):
#     txtstring = title + '\nCell #: {}'.format(cell_no)
#     title_plt_txt = ax.text(0.5, 1.075, txtstring, fontsize=fntsz,
#                             ha='center', va='center', transform=ax.transAxes)
#     return title_plt_txt
#
# def displayStats(ax, cell, fntsz):
#     stats_txtstring = "RMS: {:.2f} um\nPV: {:.2f} um".format(cell.rms, cell.pv)
#     stats_plt_txt = ax.text(0.03, 0.05, stats_txtstring, fontsize=fntsz,
#                             transform=ax.transAxes)
#     return stats_plt_txt
#
# def displayDetails(ax, cell, fntsz):
#     # details_txtstring = "Coordinates: {}\nRegion: {}\nPin cell #: {}\nCable: {},  Pin: {}\nCable: {},  Pin: {}".format(cell.rc_label, cell.region, cell.pin_cell_no, cell.p_cable, cell.pv_pin, cell.p0x_cable, cell.p0xv_pin)
#     coord_txtstring = "Coordinates: {}\nRegion: {}\nPin cell #: {}".format(cell.rc_label, cell.region, cell.pin_cell_no)
#     border = '\n----------------------------------\n'
#     board_txtstring = "BRD: {}, DAC: {}, CH: {}\nAOUT: {}, PORT: {}-{}".format(cell.board_num, cell.dac_num, cell.channel,
#                                                                                 cell.aout_pin, cell.j_port, cell.j_pin)
#     cables_txtstring = "             (Cable, Pin):\nAIR: ({}, {}), ({}, {})\nVAC: ({}, {}), ({}, {})".format(
#                                             cell.p0xa_cable, cell.p0xa_pin, cell.pa_cable, cell.pa_pin,
#                                             cell.pv_cable, cell.pv_pin, cell.p0xv_cable, cell.p0xv_pin)
#     details_txtstring = coord_txtstring + border + board_txtstring + border + cables_txtstring
#     details_plt_txt = ax.text(0.60, 0.25, details_txtstring, fontsize=fntsz,
#                                 linespacing=1.5, ha='left', va='center',
#                                 transform=ax.transAxes)
#     return details_plt_txt
#
# def displayRadii(ax, ax_fntsz):
#     large_R_text = ax.text(0.5, 0.075, 'Larger R', fontsize=ax_fntsz, color='red',
#                             ha='center', va='center', transform=ax.transAxes)
#     small_R_text = ax.text(0.5, 0.9255, 'Smaller R', fontsize=ax_fntsz, color='red',
#                             ha='center', va='center', transform=ax.transAxes)
#
# def displayIFs(cells, dx, imbounds=None, vbounds=None, colormap='jet',
#             figsize=(8,6), title_fntsz=14, ax_fntsz=12,
#             title='Influence Functions',
#             x_title='Azimuthal Dimension (mm)',
#             y_title='Axial Dimension (mm)',
#             cbar_title='Figure (microns)',
#             frame_time=500, repeat_bool=False, dispR=False,
#             stats=False, details=False):
#     """
#     Displays set of IFs in a single animation on one figure.
#     """
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.set_xlabel(x_title, fontsize = ax_fntsz)
#     ax.set_ylabel(y_title, fontsize = ax_fntsz)
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.10)
#     extent = fp.mk_extent(cells[0].ifunc, dx)
#     if not vbounds:
#         ifs = np.stack(tuple(cell.ifunc for cell in cells), axis=0)
#         vbounds = [np.nanmin(ifs), np.nanmax(ifs)]
#     if imbounds:
#         lbnd = imbounds[0] - 1
#         ubnd = imbounds[1]
#     else:
#         lbnd = 0
#         ubnd = len(cells)
#     ims = []
#     for i in range(lbnd, ubnd):
#         # print(i)
#         im = ax.imshow(cells[i+lbnd].ifunc, extent=extent, aspect='auto', cmap=colormap,
#                         vmin=vbounds[0], vmax=vbounds[1])
#         cell_no = cells[i+lbnd].no #i + 1 + lbnd
#         title_plt_txt = display_title_cell_no(ax, title, cell_no, title_fntsz)
#         stats_plt_txt = ax.text(0, 0, '')
#         details_plt_txt = ax.text(0, 0, '')
#         if stats:
#             # stats_plt_text = displayStats(ax, cells[i+lbnd], ax_fntsz)
#             stats_txtstring = "RMS: {:.2f} um\nPV: {:.2f} um".format(cells[i+lbnd].rms, cells[i+lbnd].pv)
#             stats_plt_txt = ax.text(0.03, 0.05, stats_txtstring, fontsize=ax_fntsz,
#                                     transform=ax.transAxes)
#         if details:
#             details_plt_txt = displayDetails(ax, cells[i+lbnd], ax_fntsz)
#         ims.append([im, title_plt_txt, stats_plt_txt, details_plt_txt])
#         # print('appended:', i)
#     cbar = fig.colorbar(ims[0][0], cax=cax)
#     cbar.set_label(cbar_title, fontsize=ax_fntsz)
#     if dispR:
#         displayRadii(ax, ax_fntsz)
#     ani = animation.ArtistAnimation(fig, ims, interval=frame_time, blit=False,
#                                     repeat=repeat_bool)
#     fps = int(1 / (frame_time/1000))
#
#     return ani, fps
