

#-------------------------------------------------------------------------------
#
# VetTheCommand - function that vets a command line given to it. If the
#                 command is a legal command it will return the string "Legal"
#                 Otherwise it returns the string "Illegal".
#
#                 Also, this function assumes that the command line fed to
#                 it has already been UPCASED. So the commands will all be
#                 in capital letters.
#
#-------------------------------------------------------------------------------
def VetTheCommand(CheckIt, command_line):


    # Set the legal command flag to True. Subsequent tests will set it to
    # False if there is an input error
    legal_flag = True

    # --------------  EMPTY STRING CHECK
    # First check to see if there's a command line there at all
    # If not, inform the user
    if len(command_line) == 0:
        legal_flag = False
        print("\nINPUT ERROR - you gave me an empty command.\n")

    else:  # Else there is some length to the input string so check it out.
        # Upcase the command string
        command_line = command_line.upper()

        # split the command out into a list of tokens
        command_tokens = command_line.split()
        # Grab the first token which must be a legal command
        command = command_tokens[0]

        # Check to see if the command is legal
        legal_command = CheckIt.AcceptableCommand(command_tokens)

        # At this point you have a legal command token. Now check the rest of
        # the command line tokens for:
        #   Right number of arguments for that command line
        #   Correct ranges for each argument
        #
        # The individual command checks will check for correct number of command
        # tokens and the correct value in the 4th argument if one exists.
        #
        # Remember that legal_flag was initialized to True. So any problem
        # found in these checks will set it to False

        # ---------------- QUIT ------------------------
        # Process  "QUIT"
        if command == "QUIT":
            legal_flag = True

        # ---------------- HEARTBEAT ------------------------
        elif command == "HEARTBEAT":  # Check for HEARTBEAT
            legal_flag = True

        # ---------------- HELP ------------------------
        elif command == "HELP":  # Check for HELP
            legal_flag = True

        # ---------------- TREAD ------------------------
        elif (command == "TREAD"):

            # Check for correct number of arguments
            if len(command_tokens) != 2:
                print("\nWrong Number of Arguments for TREAD")
                print("You typed in the command: ", command_line)
                print("Correct Syntax: TREAD <BRD>")
                legal_flag = False
            else: # Check the board argument
                board_flag = CheckIt.CheckBoard(command_tokens)

                if board_flag == True:
                    legal_flag = True
                else:
                    legal_flag = False

        # ---------------- VREAD ------------------------
        elif (command == "VREAD"):
            if len(command_tokens) != 4:
                print("\nWrong Number of Arguments for VREAD")
                print("You typed in the command: ", command_line)
                print("Correct Syntax: VREAD <BRD> <DAQ> <CHANNEL>")
                legal_flag = False
            else:
                board_flag = CheckIt.CheckBoard(command_tokens)
                DAC_flag = CheckIt.CheckDAC(command_tokens)
                channel_flag = CheckIt.CheckChannel(command_tokens)

                # Now check all the flags and return a final legal flag
                if board_flag and DAC_flag and channel_flag:
                    legal_flag = True
                else:
                    legal_flag = False

        # ---------------- VSET ------------------------
        elif command == "VSET":
            if len(command_tokens) != 5:
                print("\nWrong Number of Arguments for VSET")
                print("You typed in the command: ", command_line)
                print("Correct Syntax: VSET <BRD>  <DAQ> <CHANNEL> <VOLTAGE>")
                legal_flag = False
            else:
                board_flag = CheckIt.CheckBoard(command_tokens)
                DAC_flag = CheckIt.CheckDAC(command_tokens)
                channel_flag = CheckIt.CheckChannel(command_tokens)
                volts_flag = CheckIt.CheckVolts(command_tokens)

                # Now check all the flags and return a final legal flag
                if board_flag and DAC_flag and channel_flag and volts_flag:
                    legal_flag = True
                else:
                    legal_flag = False

        # ---------------- RESET ------------------------
        elif command == "RESET":
            if len(command_tokens) != 2:
                print("\nWrong Number of Arguments for VSET")
                print("You typed in the command: ", command_line)
                print("Correct Syntax: RESET <BRD>")
                legal_flag = False
            else: # Check the board argument
                board_flag = CheckIt.CheckBoard(command_tokens)

                if board_flag == True:
                    legal_flag = True
                else:
                    legal_flag = False

        # ---------------- GAIN ------------------------
        elif command == "GAIN":
            if len(command_tokens) != 5:
                print("\nWrong Number of Arguments for GAIN")
                print("You typed in the command: ", command_line)
                print("Correct Syntax: GAIN <BRD> <DAQ> <CHANNEL> <GAINVAL>")
                legal_flag = False
            else:
                board_flag = CheckIt.CheckBoard(command_tokens)
                DAC_flag = CheckIt.CheckDAC(command_tokens)
                channel_flag = CheckIt.CheckChannel(command_tokens)
                gain_flag = CheckIt.CheckGain(command_tokens)

                # Now check all the flags and return a final legal flag
                if board_flag and DAC_flag and channel_flag and gain_flag:
                    legal_flag = True
                else:
                    legal_flag = False

        # ---------------- CHANNEL OFFSET ------------------------
        elif command == "OFFSET":
            if len(command_tokens) != 5:
                print("\nWrong Number of Arguments for OFFSET")
                print("You typed in the command: ", command_line)
                print("Correct Syntax: OFFSET <BRD> <DAQ> <CHANNEL> <OFFSETVAL>")
                legal_flag = False
            else:
                board_flag = CheckIt.CheckBoard(command_tokens)
                DAC_flag = CheckIt.CheckDAC(command_tokens)
                channel_flag = CheckIt.CheckChannel(command_tokens)
                offset_flag = CheckIt.CheckOffset(command_tokens)

                # Now check all the flags and return a final legal flag
                if board_flag and DAC_flag and channel_flag and offset_flag:
                    legal_flag = True
                else:
                    legal_flag = False

        # ---------------- DAC OFFSET - DACOFF <BRD> <DAC> <GROUP> <OFFSETVAL>-------
        elif command == "DACOFF":
            if len(command_tokens) != 5:
                print("\nWrong Number of Arguments for DACOFF")
                print("You typed in the command: ", command_line)
                print("Correct Syntax: DACOFF <BRD> <DAC> <GROUP> <OFFSETVAL>")
                legal_flag = False
            else:
                board_flag = CheckIt.CheckBoard(command_tokens)
                DAC_flag = CheckIt.CheckDAC(command_tokens)
                group_flag = CheckIt.CheckGroup(command_tokens)
                DACoffset_flag = CheckIt.CheckDACOffset(command_tokens)

                # Now check all the flags and return a final legal flag
                if board_flag and DAC_flag and group_flag and DACoffset_flag:
                    legal_flag = True
                else:
                    legal_flag = False


    #print "**********Final value of Legal Flag is: ", legal_flag
    # Return the legal flag
    return(legal_flag)

#-------------------------------------------------------------------------------
#
# ShowLegalCommands - Displays the correct syntax and value ranges for
#                     all commands
#
#-------------------------------------------------------------------------------

def ShowLegalCommands():
    print("\nCommand Syntax:")
    print("\nTo set a voltage:            vset board_number  DAQ_number  Channel_number volts")
    print("\nTo Read a Voltage:           vread board_number  DAQ_number  Channel_number")
    print("\nTo read a board temperature: tread board_number")
    print("\n    Where board_number ranges from ", acceptable_board_numbers[0], " to ", acceptable_board_numbers[-1])
    print("\n    Where DAQ_number ranges from ",acceptable_DAQ_numbers[0]," through ",acceptable_DAQ_numbers[-1]," inclusive")
    print("\n    Where Voltages can range from ",acceptable_voltage_range[0]," through ",acceptable_voltage_range[1], "inclusive")
