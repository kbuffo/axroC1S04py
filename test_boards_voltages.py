# from numpy import *
# import matplotlib.pyplot as plt
import os
import pdb
from datetime import date
import axroHFDFCpy.axroBoardTesting_V2 as axBT   # 'axro Board Test = axBT'
"""
In the control box, board 1 is on bottom and board 3a is on top.
"""
today = str(date.today())
data_dir = "C:\\Users\\kbuffo\\OneDrive - University of Iowa\\Research\\PZT_work\\WorkSpace\\C1S04\\control\\voltage_data\\test_voltages"
boards = ['1','3a'] # entries are strings: '1', '2', '3a', '3b'
# offset_vals = [0, 4096, 8192, 12288, 16383] # [0 to 16383] integer 8192 IS DEFAULT
offset_vals = [8192]
tvolts = [0, -10, -1, 1, 10]
# tvolts = [-5.0]
for board in boards:
    board_num = int([*board][0])
    for offset in offset_vals:
        # print('Board num: {}\nOffset: {}'.format(board_num, offset))
        # Initialize board
        axBT.init(board_num = board_num, offset=offset)
        # test board
        filename = '\\Board_{}\\{}_b{}_{}_'.format(board, today, board, str(offset))
        axBT.test_board(board_num, tvolts=tvolts, savefile=True, header=data_dir+filename)
