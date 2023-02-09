"""
Used in conjuction with construct_connections.py to control the voltage of cell objects.
"""
import numpy as np
import serial
import time
import os
from datetime import date
import pickle

today_dtype = date.today()
today = today_dtype.strftime('%y')+'-'+today_dtype.strftime('%m')+'-'+today_dtype.strftime('%d')+'_'
repo_dir = 'C:\\Users\\kbuffo\\OneDrive - University of Iowa\\Research\\repos\\axroHFDFCpy\\'
test_save_dir = 'C:\\Users\\kbuffo\\OneDrive - University of Iowa\\Research\\PZT_work\\WorkSpace\\C1S04\\control\\voltage_data\\test_cell_voltages\\'
exec(open(repo_dir+"CommandCheck.py").read())
exec(open(repo_dir+"VetTheCommand_V3.0.py").read())
exec(open(repo_dir+"ProcessCommandFile.py").read())
abs_volt_max = 15.0

#Establish the serial connection
# ser = serial.Serial('COM3', 9600)
try:
    ser = serial.Serial('COM3', 9600)
    print('it did work')
except:
    print("it didn't work")
    pass
# print('You have imported the changed file!')

def findCells(cells, attribute, value):
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
    if len(matching_cells) == 0:
        print("Couldn't find any cells matching those parameters.")
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

def undoVoltStep(cell_no, cells, voltage, idx=-1):
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

def setCell(cell_no, cells, voltage, printCellEcho=True, printSerEcho=False):
    """
    Issues a command to set the voltage of a cell.
    voltage <= abs(15 V) for command to be exectued.
    If printCellEcho is True, the voltage read from the cell after setting it
    will be printed for that cell number.
    If printSerEcho is True, the result of the command is read from the
    serial monitor
    """
    # print('You called setCell!')
    if np.abs(voltage) <= abs_volt_max:
        cell = findCells(cells, 'no', cell_no)[0]
        cstr = 'VSET %i %i %i %f' % (cell.board_num, cell.dac_num, cell.channel, voltage)
        ser.write(cstr.encode())
        # if printSerEcho: _ = serEcho(include_print=True)
        cmd_echo = ser.readline()
        # print('Here is the response in setCell:')
        # print("Board Response is: ", cmd_echo)
        set_voltage = readCell(cell_no, cells) # read the voltage that was applied
        # print('Here is the set voltage I got back:', set_voltage)
        updateVolts(cell_no, cells, set_voltage) # update cell object
        if printCellEcho: printCellVolt(cell_no, cells, set_voltage)
    else:
        print('\nInput voltage out of bounds!\n')

def readCell(cell_no, cells, printSerEcho=False):
    """
    Issues a command to read the voltage of a cell.
    If serEcho is true, the result of the command is read from the serial monitor.
    """
    # print('You called readCell!')
    cell = findCells(cells, 'no', cell_no)[0]
    cstr = 'VREAD %i %i %i' % (cell.board_num, cell.dac_num, cell.channel)
    ser.write(cstr.encode())
    return_string = ser.readline()
    # print('Here is the response in readCell:')
    # print('Board response is:', return_string)
    # return_string = serEcho(include_print=printSerEcho)
    voltage = float(return_string.split()[-1])
    # print('In readCell: the voltage read is:', voltage)
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

def init_boards(board_nums, cells, offset=8192, printSerEcho=False):
    """
    Initializes boards by resetting them and their DAC's offset values.
    board_nums should be a list of integers: [1, 2, 3, 4].
    IT IS IMPORTANT TO INITIALIZE BOARDS BEFORE USING THEM FOR MEASUREMENTS.
    """
    for board_num in board_nums:
        # print('here is the current board number:', board_num)
        cstr = 'RESET %i' % (board_num)
        ser.write(cstr.encode())
        _ = ser.readline()
        # print('You wrote the reset command for board:', board_num)
        if printSerEcho: _ = serEcho(include_print=printSerEcho)

        for dac in range(8): # 8192 offset val was used previously as standard
            cstr = 'DACOFF %i %i 0 %i' % (board_num, dac, offset)
            ser.write(cstr.encode())
            # print('You wrote the dacoff 0 command for dac:', dac)
            _ = ser.readline()
            if printSerEcho: _ = serEcho(include_print=printSerEcho)

            cstr = 'DACOFF %i %i 1 %i' % (board_num, dac, offset)
            ser.write(cstr.encode())
            _ = ser.readline()
            # print('You wrote the dacoff 1 command for dac:', dac)
            if printSerEcho: _ = serEcho(include_print=printSerEcho)
        ground_board(board_num, cells)
        # print('Initialized board:', board_num)

def ground_board(board_num, cells):
    """
    Grounds all the pins a board.
    """
    # print('You called ground_board!')
    cells_w_board = findCells(cells, 'board_num', board_num)
    for dac in range(8):
        cells_w_dac = findCells(cells_w_board, 'dac_num', dac) # find the cells with the matching dac
        for channel in range(32):
            # print('Current board, dac, channel:', board_num, dac, channel)
            matching_cells = findCells(cells_w_dac, 'channel', channel) # find the cell witht the matching
            if len(matching_cells) > 0: # ground the cell
                cell = matching_cells[0]
                # print('I found cell # {} has board: {}, dac: {}, and channel: {}'.format(cell.no, cell.board_num, cell.dac_num, cell.channel))
                setCell(cell.no, cells, 0.0, printSerEcho=False)
                # print('I grounded a cell!')
            else: # ground pin that is not connected to cell
                cstr = 'VSET %i %i %i %f' % (board_num, dac, channel, 0.0)
                ser.write(cstr.encode())
                _ = ser.readline()
                cstr = 'VREAD %i %i %i' % (board_num, dac, channel)
                ser.write(cstr.encode())
                return_string = ser.readline()
                return_voltage = float(return_string.split()[-1])
                print('(not a cell) Board: {}, DAC: {}, CH: {} has been set to {} V.'\
                                .format(board_num, dac, channel, return_voltage))
                # print('I grounded a pin!')

def test_cells(cells, board_nums, tvolts=[-10, -1, 0, 1, 10],
                savefile=True):
    """
    Tests all the cells at different voltages.
    """
    for cell in cells: # clear the volt history of the cells
        clearVoltHistory(cell.no, cells)
    init_boards(board_nums, cells) # initialize the boards
    for volt in tvolts:
        for cell in cells:
            clearVoltHistory(cell.no, cells) # clear the volt history of the cell
            setCell(cell.no, cells, 0.0) # ground the cell
            setCell(cell.no, cells, volt) # set the cell to the test voltage
        if savefile: # save the test
            fname = today+str(volt)+'V_cells_test'
            date_dir = test_save_dir+today[:-1]
            os.mkdir(date_dir)
            with open(date_dir+fname, 'wb') as f:
                picle.dump(cells, f)
        print('Tested {} V, sleeping breifly...'.format(str(volt)))
        time.sleep(10)
