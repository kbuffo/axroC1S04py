###############################################################
#
# AXRO_CLI_V2.0.py - This version uses the CommandCheck Class
#                    and VetTheCommand_V3.0,
#                    And has the DACOFF command
###############################################################

from time import sleep
import serial

# execfile("CommandCheck.py")
# execfile("VetTheCommand_V3.0.py")
# execfile("ProcessCommandFile.py")
exec(open("CommandCheck.py").read())
exec(open("VetTheCommand_V3.0.py").read())
exec(open("ProcessCommandFile.py").read())

def Write_File(outfile, command_line, response):
    outfile.write("\n\n"+command_line+"\n")
    outfile.write(response+"\n")


# Some inits
file_open_flag = False

# Create the class for command checking
CheckIt = CommandCheck()

# Establish the connection on a specific port
#ser = serial.Serial('/dev/ttyACM0', 9600)
ser = serial.Serial('COM3', 9600)

#  Get the heartbeat
ser.write(b"HEARTBEAT")
hb_response = ser.readline()
print(hb_response)

# Get a user command
command_line = input("Gimme a full command: ")
split_command = command_line.split()
command = split_command[0].upper().encode()

# Do until the user Control-C's out
while command.upper() != "QUIT":

    # If the command was "FILE" then process the command file
    if command.upper() == "FILE":
        ProcessCommandFile(CheckIt, split_command[1], split_command[2])

    # Check to see if the command is HEARTBEAT. If it is,
    # wait for the response. Eventually we will build in a
    # timeout here.
    elif command.upper() == "HEARTBEAT":
        print("    Sending Heartbeat to the ARDUINO", command_line)
        ser.write(command_line.upper().encode())
        cmd_echo = ser.readline()
        print("Board Response is: ", cmd_echo)

    elif command.upper() == "OPEN":
        filename = input("Give me the name of the output file: ")
        outfile = open(filename, "w")
        comment = input("Give me a comment to place at the top: ")
        outfile.write(comment+"\n")
        file_open_flag = True

    elif command.upper() == "CLOSE":
        if file_open_flag == True:
            outfile.close();
            file_open_flag = False

    elif command.upper() == "DEBUG":
        ser.write(command_line.upper().encode())
        cmd_echo = ser.readline()
        print("Board Response is: ", cmd_echo)

    elif command.upper() == "NODEBUG":
        ser.write(command_line.upper().encode())
        cmd_echo = ser.readline()
        print("Board Response is: ", cmd_echo)

    else:
        # If this is a legal command........
        if (VetTheCommand(CheckIt, command_line.upper())):
            #.... then send it to the Arduino
            #print "    Legal Command - Sending to the ARDUINO", command_line
            ser.write(command_line.upper().encode())
            cmd_echo = ser.readline()
            print("Board Response is: ", cmd_echo)
            if file_open_flag == True:
                Write_File(outfile, command_line, cmd_echo)

        else:
            print("Unauthorized command IGNORING")
            #ser.write(command_line.upper())
            #cmd_echo = ser.readline()
            #print "Board Response is: ", cmd_echo

    # Get a new command
    command_line = input("Gimme a command: ")
    split_command = command_line.split()
    command = split_command[0]

# Command was Quit so close down the serial link cleanly
# Send the command to the Arduino so that it drops into it's wait state
ser.write(command_line.upper().encode())

# close down the serial line from this end
ser.close()

print("Ok then bye for now")
