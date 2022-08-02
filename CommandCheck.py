################################################################################
#
#  Class CommandCheck - class that takes the commands as typed by the user and
#                       checks to see if they are legal commands. If they are
#                       the appropriate attributes are set and legal_flag
#                       is returned as True.
#
#                       If the command is illegal either due to the command
#                       typed or the arguments specified, then legal_flag
#                       is set to False.
################################################################################

class CommandCheck():

    def __init__(self):
        self.command = ""
        self.board_number = -1
        self.DAC_number = -1
        self.channel = -1
        self.group = -1
        self.volts = -1.0
        self.gain = 65535
        self.offset = 32767
        self.DACoffset = 16383
        self.group = 1

        self.legal_flag = False
        #
        self.legal_command = True
        self.legal_board = True
        self.legal_DAC = True
        self.legal_channel = True
        self.legal_voltage = True
        self.legal_gain = True
        self.legal_offset = True
        self.legal_DACoffset = True
        self.legal_group = True

        # This is the list of all acceptable commands
        self.acceptable_commands = ["DACOFF", "TREAD", "VREAD", "VSET", "RESET", "FILE", "HEARTBEAT", "GAIN", "OFFSET", "QUIT"]
        self.acceptable_board_numbers = [0, 1, 2, 3]
        self.acceptable_DAC_numbers = [0, 1, 2, 3, 4, 5, 6, 7]
        self.acceptable_Channel_numbers = range(32)
        self.acceptable_voltage_range = [-15.25, 15.25]
        self.acceptable_gain_range = [0, 65535]
        self.acceptable_offset_range = [0, 32767]
        self.acceptable_DACoffset_range = [0, 16383]
        self.acceptable_group_numbers = [0,1]  # For the DACOFF command

    #---------------------------------------------------------------------------
    #
    # AcceptableCommand - Check to see if the command is in the acceptale
    #                     command list
    #
    #---------------------------------------------------------------------------
    def AcceptableCommand(self, command_tokens):
        # Extract the command
        self.command = command_tokens[0]

        # Check to see if the command is legal
        if (self.command not in self.acceptable_commands):
            print("\nINPUT ERROR - invalid command! You typed: ", command_line)
            print("    Acceptable commands are: ")
            print("    ", self.acceptable_commands)
            self.legal_command = False
        else:
            self.legal_command = True

        return(self.legal_command)

    #---------------------------------------------------------------------------
    #
    # CheckBoard
    #
    #
    #---------------------------------------------------------------------------
    def CheckBoard(self, command_tokens):
        # Extract the board number
        board_number = int(command_tokens[1])

        # Test the user input for board number  to see if it's legal
        if  board_number not in self.acceptable_board_numbers:
            self.legal_board = False
            print("\nINPUT ERROR = Your input board range of:", command_tokens[1], "is invalid!!\n    Valid board numbers range from: ", self.acceptable_board_numbers[0], " to ",self.acceptable_board_numbers[-1])
        else:
            self.legal_board = True

        return(self.legal_board)

    #---------------------------------------------------------------------------
    #
    # CheckDAC
    #
    #
    #---------------------------------------------------------------------------
    def CheckDAC(self, command_tokens):
        # Extract the DAC argument
        DAC_number = int(command_tokens[2])

        # Test the DAC number to be sure it's in the range
        if DAC_number not in self.acceptable_DAC_numbers:
            self.legal_DAC = False
            print("\nINPUT ERROR - Incorrect DAC number")
            print("You typed in the command: ", command_line)
            print("Acceptable integer DAC value range is from ", self.acceptable_DAC_numbers[0], " to ", self.acceptable_DAC_numbers[-1], " inclusive")
        else:
            self.legal_DAC = True

        # Return the resultant DAC flag
        return(self.legal_DAC)

    #---------------------------------------------------------------------------
    #
    # CheckChannel
    #
    #
    #---------------------------------------------------------------------------
    def CheckChannel(self, command_tokens):
        # Extract the channel number argument
        self.channel = int(command_tokens[3])

        # If it's not within range, bounce it
        if self.channel not in self.acceptable_Channel_numbers:
            self.legal_channel = False
            print("\nERROR - Channel Number out of range")
            #print "You typed in the command: ", command_line
            print("Acceptable range of values for specifying a channel are:")
            print("    ",self.acceptable_Channel_numbers[0]," through ",self.acceptable_Channel_numbers[-1], "inclusive")

        else:
            self.legal_channel = True

        # Return the resultant channel flag
        return(self.legal_channel)


    #---------------------------------------------------------------------------
    #
    # CheckGroup
    #
    #
    #---------------------------------------------------------------------------
    def CheckGroup(self, command_tokens):
        # Extract the Group argument
        self.group = int(command_tokens[3])

        # If it's not within range, bounce it
        if self.group not in self.acceptable_group_numbers:
            self.legal_group = False
            print("\nERROR - Group Number out of range: ",self.group)
            print("Acceptable range of values for specifying a DACOFF group are:")
            print("    0 or 1")
        else:
            self.legal_group = True

        # Return the resultant group flag
        return(self.legal_group)

    #---------------------------------------------------------------------------
    #
    # CheckVolts
    #
    #
    #---------------------------------------------------------------------------
    def CheckVolts(self, command_tokens):
        # Extract the voltage argument
        self.volts = float(command_tokens[4])

        # Check to see if the voltage is within range
        if (self.volts < self.acceptable_voltage_range[0]) or \
           (self.volts > self.acceptable_voltage_range[-1]):
            print("ERROR - Invalid voltage range. Input voltages must be between:")
            print("    ",self.acceptable_voltage_range[0]," and ",self.acceptable_voltage_range[-1])
            self.legal_voltage = False
        else:
            self.legal_voltage = True

        # Return the resultant flag
        return(self.legal_voltage)


    #---------------------------------------------------------------------------
    #
    # CheckGain
    #
    #
    #---------------------------------------------------------------------------
    def CheckGain(self,  command_tokens):
        # Extract the gain argument
        self.gain = int(command_tokens[4])

        # Check it's limits
        if (self.gain < self.acceptable_gain_range[0]) or \
           (self.gain > self.acceptable_gain_range[-1]):
            print("ERROR - Invalid gain value. Input gain must be between:")
            print("    ",self.acceptable_gain_range[0]," and ",self.acceptable_gain_range[-1]   )
            self.legal_gain = False
        else:
            self.legal_gain = True

        # Return the resultant flag
        return(self.legal_gain)



    #---------------------------------------------------------------------------
    #
    # CheckOffset
    #
    #
    #---------------------------------------------------------------------------
    def CheckOffset(self,  command_tokens):
        # Extract the  offset argument
        self.offset = float(command_tokens[4])

        # Check it's limits
        if (self.offset < self.acceptable_offset_range[0]) or \
           (self.offset > self.acceptable_offset_range[-1]):
            print("ERROR - Invalid offset value. Input offset must be between:")
            print("    ",self.acceptable_offset_range[0]," and ",self.acceptable_offset_range[-1]   )
            self.legal_offset = False
        else:
            self.legal_offset = True

        # Return the resultant flag
        return(self.legal_offset)


    #---------------------------------------------------------------------------
    #
    # CheckDACOffset
    #
    #
    #---------------------------------------------------------------------------
    def CheckDACOffset(self,  command_tokens):
        # Extract the  DACoffset argument
        self.DACoffset = int(command_tokens[3])

        # Check it's limits
        if (self.DACoffset < self.acceptable_DACoffset_range[0]) or \
           (self.DACoffset > self.acceptable_DACoffset_range[-1]):
            print("ERROR - Invalid DACoffset value. Input DACoffset must be between:")
            print("    ",self.acceptable_DACoffset_range[0]," and ",self.acceptable_DACoffset_range[-1]   )
            self.legal_DACoffset = False
        else:
            self.legal_DACoffset = True

        # Return the resultant flag
        return(self.legal_DACoffset)
