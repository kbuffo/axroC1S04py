import numpy as np
import serial
import time
import os

exec(open("CommandCheck.py").read())
exec(open("VetTheCommand_V3.0.py").read())
exec(open("ProcessCommandFile.py").read())

verbose = True
abs_volt_max = 15.0

#Establish the serial connection
ser = serial.Serial('COM3', 9600)

cellmap = range(256)

def convChan(chan):
    """
    Convert channel from 0 - 255 to proper DAC and channel number on board.
    """
    dac = int(chan) / 32
    channel = int(chan) % 32
    return dac,channel

def echo():
    """
    Retrieve the response from the board after a command is issued.
    Print the response, also return the response as a string.
    """
    cmd_echo = ser.readline()
    if verbose:
        print("Board Response is: ", cmd_echo)
    return cmd_echo

def encoded_init():
    """
    Change to software directory and run initialization script.
    """
    ProcessCommandFile(CommandCheck(),arddir+'SetUp_DACOFF_FullBoard3.txt',0)
    return None

def setChan(board_num, chan, volt):
    """
    Convert the channel number into board format and then issue
    command to set voltage.
    Limit of <= abs(15 V) is encoded into this function.
    """
    if abs(volt) <= abs_volt_max:
        dac,channel = convChan(chan)
        cstr = 'VSET %i %i %i %f' % (board_num,dac,channel,volt)
        ser.write(cstr.encode())
        echo()
    else:
        print('Voltage out of bounds!')
    return None

def readChan(board_num, chan):
    """
    Convert channel into proper DAC and channel number
    Then issue read command and parse voltage from echo.
    """
    dac,channel = convChan(chan)
    cstr = 'VREAD %i %i %i' % (board_num,dac,channel)
    ser.write(cstr.encode())
    s = echo()
    print('\n==================================\ns string: {}\nconversion: {}\n==================================\n'.format(s, float(s.split()[-1])))
    return float(s.split()[-1])

def close():
    """
    Close the serial connection to the Arduino
    """
    ser.write('QUIT'.encode())
    ser.close()
    return None

#Define higher level functions for interacting with piezo mirror
def ground(board_num):
    """
    Set all channels to zero volts
    """
    for c in range(256):
        setChan(board_num, c,0.0)
    return None

def setVoltArr(board_num, voltage):
    """
    Set all 256 channels using a 256 element voltage vector.
    The indices correspond to piezo cell number.
    """
    for c in range(256):
        setChan(board_num, cellmap[c],voltage[c])
    return None

def setVoltChan(board_num, chan, volt):
    """
    Set individual piezo cell, channel corresponds to piezo cell number
    """
    setChan(board_num, cellmap[chan], volt)
    return None

def readVoltArr(board_num):
    """
    Loop through and read the voltages on all piezo cells. Return
    vector of voltages where index matches cell number (minus one).
    """
    v = []
    for c in range(256):
        v.append(readChan(board_num, cellmap[c]))
    return np.asarray(v)

def readVoltChan(board_num, chan):
    """
    Read individual piezo cell voltage. Chan refers to piezo cell number.
    """
    return readChan(board_num, cellmap[chan])

def init(board_num = 1, offset=8192):
    """
    Change to software directory and run initialization script.
    """
    cstr = 'RESET %i' % (board_num)
    ser.write(cstr.encode())
    echo()

    for dac in range(8): # 8192 offset val was used previously as standard
        cstr = 'DACOFF %i %i 0 %i' % (board_num, dac, offset)
        ser.write(cstr.encode())
        echo()

        cstr = 'DACOFF %i %i 1 %i' % (board_num, dac, offset)
        ser.write(cstr.encode())
        echo()

    ground(board_num)
    return None

def test_board(board_num, tvolts = [0,-10,-1,1,10], savefile=True,
                header = 'hereareyourtestresults'):
    ground(board_num)

    for volt in tvolts:
        ground(board_num)
        setVoltArr(board_num, np.ones(256)*volt)
        rvolts = readVoltArr(board_num)
        if savefile:
            np.savetxt(header + str(volt) + 'V_ReadResponse.txt',rvolts)
        print('Tested ' + str(volt) + 'V, Sleeping Briefly....')
        time.sleep(10)
