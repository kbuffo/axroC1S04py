import axroHFDFCpy.axroBoardTesting_V2 as axBT   # 'axro Board Test = axBT'
import time
board_choice = 2 # specify which board you want to apply voltage to
voltage = -5.0 # voltage to apply

# initialize boards 1 & 3
axBT.init(board_num = board_choice)
# axBT.init(board_num = 3)

# apply voltage to board
# axBT.board_num = board_choice
axBT.setVoltArr(board_choice, np.ones(256)*voltage)
rvolts = axBT.readVoltArr(board_num=board_choice)
print('\n==========Voltage Read Response==========\n', rvolts)
time.sleep(10)
# input('Press ENTER >')
print('done')
