"""
Used in conjuction with construct_connections.py to control the voltage of cell objects.
"""
try:
    import construct_connections as cc
except:
    import axroHFDFCpy.construct_connections as cc
import numpy as np
import serial
import serial.tools.list_ports
import copy
import time
import os
from datetime import date
import pickle

today_dtype = date.today()
today = today_dtype.strftime('%y')+'-'+today_dtype.strftime('%m')+'-'+today_dtype.strftime('%d')+'_'
# repo_dir = 'C:\\Users\\kbuffo\\OneDrive - University of Iowa\\Research\\repos\\axroHFDFCpy\\'
# test_save_dir = 'C:\\Users\\kbuffo\\OneDrive - University of Iowa\\Research\\PZT_work\\WorkSpace\\C1S04\\control\\voltage_data\\test_cell_voltages\\'

personal_dir = 'C:\\Users\\kbuffo\\'
accufiz_dir = 'C:\\Users\\AccuFiz\\'
if os.path.exists(personal_dir): # check if on AccuFiz pc or laptop
    repo_dir = personal_dir+'OneDrive - University of Iowa\\Research\\repos\\axroHFDFCpy\\'
    test_save_dir = personal_dir+'OneDrive - University of Iowa\\Research\\PZT_work\\WorkSpace\\C1S04\\control\\voltage_data\\test_cell_voltages\\'
    pole_save_dir = personal_dir+'OneDrive - University of Iowa\\Research\\PZT_work\\WorkSpace\\C1S04\\control\\voltage_data\\pole_voltages\\'
else:
    repo_dir = accufiz_dir+'OneDrive - University of Iowa\\Research\\repos\\axroHFDFCpy\\'
    test_save_dir = accufiz_dir+'OneDrive - University of Iowa\\Research\\PZT_work\\WorkSpace\\C1S04\\control\\voltage_data\\test_cell_voltages\\'
    pole_save_dir = accufiz_dir+'OneDrive - University of Iowa\\Research\\PZT_work\\WorkSpace\\C1S04\\control\\voltage_data\\pole_voltages\\'

exec(open(repo_dir+"CommandCheck.py").read())
exec(open(repo_dir+"VetTheCommand_V3.0.py").read())
exec(open(repo_dir+"ProcessCommandFile.py").read())
abs_volt_max = 15.0
standard_read_write_delay = 0.25
# open serial connection
ser_port_opened = False
ser = serial.Serial()
ser.port = 'COM3'
ser.baudrate = 9600
ser.open()


def findCells_wMultParams(cells, attributes, values, printFail=True):
    """
    Returns a list of cells whose attributes matches the given values.
    For example, attributes=['board_num', 'j_port'] and values=[1, 'J2'] will return
    the list of cells that are connected to board 1 and j-port J2.
    """
    for attr, val in zip(attributes, values):
        match_cells = findCells(cells, attr, val, printFail=printFail)
        cells = match_cells
    return cells

def findCells(cells, attribute, value, printFail=True):
    """
    Returns a list of cells whose attribute matches the given parameter.
    For example, attribute='no' and value=1 will return the list of cells whose cell
    number is 1 (which only a single cell in this case).
    """
    matching_cells = []
    for cell in cells:
        found_value = getattr(cell, attribute)
        if found_value == value:
            matching_cells.append(cell)
    if len(matching_cells) == 0 and printFail is True:
        print("Couldn't find any cells with attribute: {} and value: {}."\
                .format(attribute, value))
    return matching_cells

def printCellVolt(cell_no, cells, voltage):
    """
    Print the voltage a cell is at.
    """
    cell = findCells(cells, 'no', cell_no)[0]
    print('Cell #: {} is now at {} V.'.format(cell.no, cell.voltage))

def updateVolts(cell_no, cells, voltage):
    """
    Update the current voltage of a cell and add it to voltage history.
    """
    cell = findCells(cells, 'no', cell_no)[0]
    cell.voltage = voltage
    cell.volt_hist.append(voltage)

def undoVoltStep(cell_no, cells, idx=-1):
    """
    Removes voltage[idx] from voltage history of cell.
    Default is to remove last step
    """
    cell = findCells(cells, 'no', cell_no)[0]
    cell.volt_hist.pop(idx)

def clearVoltHistory(cell_no, cells):
    """
    Clears the voltage history list for a cell.
    """
    cell = findCells(cells, 'no', cell_no)[0]
    cell.volt_hist = []

def setCell(cell_no, cells, voltage, printCellEcho=True, printSerEcho=False,
            write_read_delay=standard_read_write_delay, include_delay=True):
    """
    Issues a command to set the voltage of a cell.
    voltage <= abs(15 V) for command to be exectued.
    If printCellEcho is True, the voltage read from the cell after setting it
    will be printed for that cell number.
    If printSerEcho is True, the result of the command is read from the
    serial monitor
    """
    if np.abs(voltage) <= abs_volt_max:
        cell = findCells(cells, 'no', cell_no)[0]
        cstr = 'VSET %i %i %i %f' % (cell.board_num, cell.dac_num, cell.channel, voltage)
        ser.write(cstr.encode()) # apply the voltage
        _ = serEcho(include_print=printSerEcho)
        if include_delay:
            time.sleep(write_read_delay)
            set_voltage = readCell(cell_no, cells, printSerEcho=printSerEcho) # read the voltage that was applied
            updateVolts(cell_no, cells, set_voltage) # update cell object
        if printCellEcho and include_delay: printCellVolt(cell_no, cells, set_voltage)
    else:
        print('\nInput voltage out of bounds!\n')

def readCell(cell_no, cells, printSerEcho=False):
    """
    Issues a command to read the voltage of a cell.
    If serEcho is true, the result of the command is read from the serial monitor.
    """
    cell = findCells(cells, 'no', cell_no)[0]
    cstr = 'VREAD %i %i %i' % (cell.board_num, cell.dac_num, cell.channel)
    ser.write(cstr.encode())
    return_string = serEcho(include_print=printSerEcho)
    voltage = float(return_string.split()[-1])
    return voltage

def serEcho(include_print=False):
    """"
    Retrieve the response from the board after a command is issued.
    If include_print=True, print the response.
    Return the response as a string.
    """
    cmd_echo = ser.readline()
    if include_print: print("Board Response is: ", cmd_echo)
    return cmd_echo

def init_boards(board_nums, offset=8192, printCellEcho=False, printSerEcho=False):
    """
    Initializes boards by resetting them and their DAC's offset values.
    board_nums should be a list of integers: [1, 2, 3, 4].
    IT IS IMPORTANT TO INITIALIZE BOARDS BEFORE USING THEM FOR MEASUREMENTS.
    """
    print('Initializing boards...')
    for board_num in board_nums:
        cstr = 'RESET %i' % (board_num)
        ser.write(cstr.encode())
        _ = serEcho(include_print=printSerEcho)

        for dac in range(8): # 8192 offset val was used previously as standard
            cstr = 'DACOFF %i %i 0 %i' % (board_num, dac, offset)
            ser.write(cstr.encode())
            _ = serEcho(include_print=printSerEcho)

            cstr = 'DACOFF %i %i 1 %i' % (board_num, dac, offset)
            ser.write(cstr.encode())
            _ = serEcho(include_print=printSerEcho)
        ground_board(board_num, printCellEcho=False, printSerEcho=False)
        print('Initialized board:', board_num)

def ground_board(board_num, printCellEcho=True, printSerEcho=False):
    """
    Grounds all the pins a board.
    """
    cells = cc.construct_cells()
    cells_w_board = findCells(cells, 'board_num', board_num, printFail=False)
    for dac in range(8):
        cells_w_dac = findCells(cells_w_board, 'dac_num', dac, printFail=False) # find the cells with the matching dac
        for channel in range(32):
            matching_cells = findCells(cells_w_dac, 'channel', channel, printFail=False) # find the cell with the matching channel
            if len(matching_cells) > 0: # ground the cell
                cell = matching_cells[0]
                setCell(cell.no, cells, 0.0, printCellEcho=printCellEcho,
                        printSerEcho=printSerEcho, include_delay=False)
            else: # ground pin that is not connected to cell
                cstr = 'VSET %i %i %i %f' % (board_num, dac, channel, 0.0)
                ser.write(cstr.encode())
                _ = serEcho(include_print=printSerEcho)
                cstr = 'VREAD %i %i %i' % (board_num, dac, channel)
                ser.write(cstr.encode())
                return_string = serEcho(include_print=printSerEcho)
                return_voltage = float(return_string.split()[-1])
                if printCellEcho:
                    print('(not a cell) Board: {}, DAC: {}, CH: {} has been set to {} V.'\
                                    .format(board_num, dac, channel, return_voltage))

def test_cells(input_cells, board_nums, tvolts=[-10, -1, 0, 1, 10],
                tdelays=None,
                fname='cells_voltages_tests', save_dir=None):
    """
    Tests all the cells at different voltages. Saves the set of cell objects.
    Sets a cell to tvolts, waits a small delay, measures its voltage, grounds the cell,
    then moves on to the next one.
    """
    start_time = time.time()
    cells = copy.deepcopy(input_cells)
    if tdelays is None:
        tdelays=[standard_read_write_delay]*len(tvolts)
    if len(tvolts) != len(tdelays):
        print('ERROR: tdelays must be a list the same length as tvolts.')
        return None
    init_boards(board_nums) # initialize the boards
    for cell in cells:
        clearVoltHistory(cell.no, cells) # clear the volt history of the cells
    for volt, rw_delay in zip(tvolts, tdelays):
        print('Now testing: {} V, Delay: {} s'.format(volt, rw_delay))
        for cell in cells:
            # print('Testing {} V on cell {}: volt hist before: {}'.format(volt, cell.no, cell.volt_hist))
            setCell(cell.no, cells, 0.0, printCellEcho=False) # ground the cell
            undoVoltStep(cell.no, cells, idx=-1) # remove the ground from the volt history
            setCell(cell.no, cells, volt, write_read_delay=rw_delay) # set the cell to the test voltage
            # print('Testing {} V on cell {}: volt hist after: {}'.format(volt, cell.no, cell.volt_hist))
        print('Tested {} V, Delay: {} s, sleeping breifly...'.format(str(volt), rw_delay))
        if len(tvolts) > 1: time.sleep(10)
    for board_num in board_nums:
        ground_board(board_num, printCellEcho=False)
    save_tested_cells(fname, save_dir, cells)
    dt = time.time() - start_time
    print('--------------------------------------------')
    print('Testing complete. Cells are now grounded.')
    print('Total time elapsed: {:.2f} s = {:.2f} min.'.format(dt, dt/60))
    return cells

def test_cells_new(input_cells, board_nums, tvolts=[-10, -1, 0, 1, 10],
                fname='cells_voltages_tests', save_dir=None):
    """
    Tests all the cells at different voltages. Saves the set of cell objects.
    For all cells: grounds the cell then sets it to tvolt.
    For all cells: reads the voltage
    This method is much faster than test_cells() since it does not include a delay inbetween
    setting cells. However, this method will not reveal shorts during testing.
    The date of the test will be appended to the beginning of fname automatically.
    """
    start_time = time.time()
    cells = copy.deepcopy(input_cells)
    init_boards(board_nums) # initialize the boards
    for volt in tvolts:
        print('Now testing: {} V'.format(volt))
        for cell in cells: # write the voltage to all the cells
            # if cell.no != 8: continue
            setCell(cell.no, cells, 0.0, include_delay=False,
                    printCellEcho=False) # ground the cell
            # undoVoltStep(cell.no, cells, idx=-1) # remove the ground from the volt history
            setCell(cell.no, cells, volt, include_delay=False) # set the cell to the test voltage
        time.sleep(1)
        for cell in cells: # read the voltage from all the cells
            # if cell.no != 8: continue
            read_volt = readCell(cell.no, cells)
            updateVolts(cell.no, cells, read_volt)
            print('Cell #: {} is now at {} V'.format(cell.no, read_volt))
        print('Tested {} V, sleeping breifly...'.format(str(volt)))
        if len(tvolts) > 1: time.sleep(10)
    for board_num in board_nums:
        ground_board(board_num, printCellEcho=False)
    save_tested_cells(fname, save_dir, cells)
    dt = time.time() - start_time
    print('--------------------------------------------')
    print('Testing complete. Cells are now grounded.')
    print('Total time elapsed: {:.2f} s = {:.2f} min.'.format(dt, dt/60))
    return cells


def test_for_shorts(input_cells, board_nums, tvolt=10.0, fname='short_voltage_test',
                    save_dir=None):
    """
    You must test all 288 cells for indexing to be correct when using cc.plot_short_test().
    The date that the test was performed will automatically be appended to the beginning of fname.
    """
    start_time = time.time()
    cells = copy.deepcopy(input_cells)
    init_boards(board_nums)
    for cell in cells:
        clearVoltHistory(cell.no, cells)
    for cell in cells:
        setCell(cell.no, cells, tvolt, printCellEcho=False) # set cell to test voltage
        for other_cell in cells: # read the voltages from all other cells
            if other_cell.no == cell.no: continue
            else:
                read_voltage = readCell(other_cell.no, cells)
                updateVolts(other_cell.no, cells, read_voltage)
        setCell(cell.no, cells, 0.0, printCellEcho=False) # ground the cell
        undoVoltStep(cell.no, cells, idx=-1) # remove the ground from the volt history
        print('Tested Cell: {}'.format(cell.no))
    for board_num in board_nums:
        ground_board(board_num, printCellEcho=False)
    save_tested_cells(fname, save_dir, cells)
    dt = time.time() - start_time
    print('--------------------------------------------')
    print('Testing complete. Cells are now grounded.')
    print('Total time elapsed: {:.2f} s = {:.2f} min.'.format(dt, dt/60))
    return cells

def hot_pole(input_cells, board_nums, tvolt=15, meas_dt=60.0,
             voltTimes_fname='hot_pole_times', voltMaps_fname='hot_pole_voltage_maps',
             save_dir=None):
    cells = copy.deepcopy(input_cells)
    init_boards(board_nums)
    voltMaps = []
    voltTimes = []
    for cell in cells:
        clearVoltHistory(cell.no, cells)
        setCell(cell.no, cells, tvolt, include_delay=False)
    t_0 = time.time()
    i = 0
    try:
        while True:
            voltTime = time.time()
            voltMap = get_voltMap(cells)
            print('---------------------------------------------------------')
            print("iteration: {}, t = {:.2f} min"\
                    .format(i, (voltTime-t_0)/60.))
            print('Some volts: {}'.format(voltMap[0, :4]))
            voltTimes.append(voltTime)
            voltMaps.append(voltMap)
            time.sleep(meas_dt - ((time.time()-t_0)) % meas_dt)
            i += 1
    except KeyboardInterrupt:
        voltTimes = (np.array(voltTimes) - voltTimes[0])/60
        voltMaps = np.stack(voltMaps, axis=0)
        final_time = time.time() - t_0
        print('Polling finished. Time of completion: {:.2f} min. voltMaps shape: {}'\
        .format(final_time/60., voltMaps.shape))
        for board_num in board_nums:
            ground_board(board_num, printCellEcho=False, printSerEcho=False)
        if (voltTimes_fname is not None) and (voltMaps_fname is not None): # save the test
            if not save_dir: save_dir = pole_save_dir
            voltTimes_full_fname = save_dir+today+voltTimes_fname
            voltMaps_full_fname = save_dir+today+voltMaps_fname
            np.save(voltTimes_full_fname, voltTimes)
            np.save(voltMaps_full_fname, voltMaps)
        return voltMaps, voltTimes

def get_voltMap(cells):
    voltMap = np.zeros(cc.cell_order_array.shape)
    for cell in cells:
        read_volt = readCell(cell.no, cells)
        voltMap = np.where(cell.no==cc.cell_order_array, read_volt, voltMap)
    return voltMap

def save_tested_cells(fname, save_dir, cells):
    """
    Saves a list of tested cell objects whose voltage histories have been changed.
    """
    if fname is not None: # save the test
        if save_dir: full_fname = save_dir+today+fname
        else: full_fname = test_save_dir+today+fname
        cc.save_cells(cells, full_fname)
