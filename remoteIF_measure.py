import construct_connections as cc
import cellElectronics as ce
import webIF
import copy
import time

def measureIFs(input_cells, meas_dir, cells_fname=None, board_nums=[1,3],
                IF_voltage=10., measCount=32):
    cells = copy.deepcopy(input_cells)
    # full list of cells used to get the voltMaps
    full_cells = cc.construct_cells()
    w = webIF.WebIF() # create instance of 4D web service
    w.setTimeout(150000) # set the timeout to 2.5 minutes
    ce.init_boards(board_nums) # initialize the boards
    start_time = time.time() # start timer
    for cell in cells:
        print('-'*12+'CELL # {}'.format(cell.no)+'-'*12)
        # get the voltage map with all cells grounded
        cell.gnd_voltMap = ce.get_voltMap(full_cells)
        # take an average measurement with all cells grounded
        w.averageMeasure(measCount)
        # save the grounded measurement
        w.saveMeasurement(meas_dir+'cell_{}_grounded'.format(cell.no))
        # energize the cell
        ce.setCell(cell.no, cells, IF_voltage)
        # get the voltage map with the cell energized
        cell.high_voltMap = ce.get_voltMap(full_cells)
        # take an average measurement with the cell energized
        w.averageMeasure(measCount)
        # save the energized measurement
        w.saveMeasurement(meas_dir+'cell_{}_{}V'.format(cell.no,
                            str(int(IF_voltage))))
        # ground all cells
        for board_num in board_nums:
            ce.ground_board(board_num, printCellEcho=False)
        print('Cells are now grounded.')

    # stop timer
    dt = time.time() - start_time
    # save the list of cell objects
    if cells_fname is not None: cc.save_cells(cells, cells_fname)
    print('--------------------------------------------')
    print('Measurements complete. Cells are now grounded.')
    print('Total time elapsed: {:.2f} min = {:.2f} hrs.'.format(dt/60, dt/3600))
    return cells
