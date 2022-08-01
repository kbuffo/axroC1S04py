//#################################################################################
//
// AXRO_PRODUCTION_TEMP_LTC_DAQ_V0.2 - This program will read the Board temperature
//                                     read the LTC 1859 voltage
//                                     readback and set the specified
//.                                    DAQ channel
//
//                                     For the rest of the commands it just echos
//                                     them back to the user.
//
//################################################################################

#include <SPI.h>

//FOR USE ON THE SWEAP WINDOWS PC
//#include <C:\Users\sweap\Desktop\Mirror Control\Arduino\AXROINCLUDES\AXRO_BRD1.h>
//#include <C:\Users\sweap\Desktop\Mirror Control\Arduino\AXROINCLUDES\AXRO_Functions.cpp>

//#include <C:\Users\sweap\Desktop\Mirror Control\Arduino\AXROINCLUDES\RigSPIforTREAD.cpp>
//#include <C:\Users\sweap\Desktop\Mirror Control\Arduino\AXROINCLUDES\RigForLTC1859.cpp>
//#include <C:\Users\sweap\Desktop\Mirror Control\Arduino\AXROINCLUDES\RigForAD5372.cpp>

////FOR USE ON MY MAC
//#include </Users/ggermain/Desktop/ARDUINO/AXROINCLUDES/AXRO_BRD1.h>
//#include </Users/ggermain/Desktop/ARDUINO/AXROINCLUDES/AXRO_Functions.cpp>
//
//#include </Users/ggermain/Desktop/ARDUINO/AXROINCLUDES/RigSPIforTREAD.cpp>
//#include </Users/ggermain/Desktop/ARDUINO/AXROINCLUDES/RigForLTC1859.cpp>
//#include </Users/ggermain/Desktop/ARDUINO/AXROINCLUDES/RigForAD5372.cpp>

// For WFS Laptop Ops
#include <C:\Users\Casey\Software\python_repository\axroHFDFCpy\AXRO_BRD1.h>
#include <C:\Users\Casey\Software\python_repository\axroHFDFCpy\AXRO_Functions.cpp>
#include <C:\Users\Casey\Software\python_repository\axroHFDFCpy\RigSPIforTREAD.cpp>
#include <C:\Users\Casey\Software\python_repository\axroHFDFCpy\RigForLTC1859.cpp>
#include <C:\Users\Casey\Software\python_repository\axroHFDFCpy\RigForAD5372.cpp>

boolean debug = true;

int num_cmd_bytes = 0;
int num_board_id_bytes = 0;

#define INPUT_SIZE 80

char command_line[INPUT_SIZE+1];
char *command;
char *first_token;
char *next_token;
 
// ADT7310 Temperature Sensor Definitions              
// Temperature value read from the ADT7310 on the proto board 
unsigned int temperature_value = 0; 

// LTC1859 definitions
float T_deg_C, volts;
float sign = 1;
word conv_result = 0;
float channel_voltage = 0;
byte command_byte;
byte ltc_channels[8] = {B10000100, 
                         B11000100,
                         B10010100, 
                         B11010100, 
                         B10100100,
                         B11100100, 
                         B10110100, 
                         B11110100};
      
// AD5372 DAQ Definitions
byte byte1, byte2, byte3;
unsigned int data_val;
byte group_byte = 64;

int board, group_no, daq, channel;
    
unsigned int gain, offset;

////////////////////////////////////////////////////////////////////////////////
//  
//                           SETUP
//
////////////////////////////////////////////////////////////////////////////////
void setup() 
{

  // Set up LED 13 for digital ops, and set it low to begin with
   pinMode(13, OUTPUT);
   digitalWrite(13, LOW); 
   
  //Initialize the Serial and wait for port to open:
  Serial.begin(9600);
  while (!Serial) 
    {
    ; // wait for serial port to connect. Needed for Leonardo only
    }
  // prints title with ending line break
  
   num_cmd_bytes = 0;

  pinMode(DACCLRbar, OUTPUT);
   
  // 74HC154 function selects are outputs
  pinMode(SSA0, OUTPUT);
  pinMode(SSA1, OUTPUT);
  pinMode(SSA2, OUTPUT);
  pinMode(SSA3, OUTPUT);
  
  // Set the pin directions for E-MUXA3-0
  pinMode(MUXA3, OUTPUT);  // MUX input line
  pinMode(MUXA2, OUTPUT);  // MUX input line
  pinMode(MUXA1, OUTPUT);  // MUX input line
  pinMode(MUXA0, OUTPUT);  // MUX input line

  // Set the pin directions for E-MUXA4 MUX enable/Disable
  pinMode(MUXA4, OUTPUT);  // MUX enable/disable

  //  System Enable is an output and is active low
  pinMode(SYSENbar, OUTPUT);

  // Boar Selects are outputs
  pinMode(BRD_A0, OUTPUT);
  pinMode(BRD_A1, OUTPUT);

  pinMode(13, OUTPUT);
  
  // enable DAC outputs
  digitalWrite(DACCLRbar, HIGH);

  // Select no function for a clean start
  SelectFunction(Y0);
  SystemDisable();
  delay(50);
   
} // END SETUP


////////////////////////////////////////////////////////////////////////////////////
//  
// LOOP
//
///////////////////////////////////////////////////////////////////////////////////
void loop() 
{
 
 
  // Now check for a command from the User
  //
  // Get the number of bytes in the Serial buffer (if any)
  num_cmd_bytes = Serial.available();
  
  // If you have some bytes (i.e. > 0) then parse the command
  if (num_cmd_bytes > 0)
    {
      Serial.readBytes(command_line, num_cmd_bytes);
      
      // Place a NULL at the end of the actual command line
      command_line[num_cmd_bytes] = 0;
     
       // Grab the first token which is always a command
       first_token = strtok(command_line, " ");
       command = strdup(first_token);
          
       // VSET - Process the DAQ Voltage Set command 
       // 
       // VSET <BRD>  <DAQ> <CHANNEL> <VOLTAGE>
       if ( !strcmp(command_line,"VSET") )
          {
              // BOARD -  Get the board id
              next_token = strtok(NULL, " " );
              board = atoi(next_token);
              
              // DAQ -  Get the DAQ on the board
              next_token = strtok(NULL, " " );
              daq = atoi(next_token);
              
              // CHANNEL -  Get the CHANNEL on the DAQ
              next_token = strtok(NULL, " " );
              channel = atoi(next_token);
              
              // VOLTS -  Get the voltage to be set on that DAQ
              next_token = strtok(NULL, " " );
              volts = atof(next_token);
           
              // Rig the SPI setup for the AD5372
              RigSPIforDAQ();
        
              // Select the appropriate board
              Board_Select(board);      
             delay(2);
             
             // Select the DAQ
             SelectFunction(daq);
 
             // Calculate the command byte
             byte1 = Build_Group_Byte(channel);

             // Map the user specified voltage to the DAQ range
             // of 0->65535
             data_val = 65535 * (volts + 15.25) / (30.5);
             
             // Write the data value out.
             write(byte1, data_val);
             // select no function, disable system
             SelectFunction(Y0);
             SystemDisable();             
             delay(1);
             
             if (debug)
             {Serial.print("I'm setting Channel number: ");
              Serial.print(channel);
              Serial.print(" on DAQ number: ");
              Serial.print(daq);
              Serial.print(" on board number: ");
              Serial.print(board);
              Serial.print(" with a voltage of: ");
              Serial.print(volts);
              Serial.print(" Scaled Integer is: ");
              Serial.print(data_val);
              Serial.print(" Command Byte is: ");
              Serial.println(byte1, BIN);
             }

          }

        // VREAD - VREAD <BRD> <DAQ> <CHANNEL>
        else if (!strcmp(command_line,"VREAD") )
                {
                   // Get the board id
                   next_token = strtok(NULL, " " );
                   board = atoi(next_token);
              
                   // Get the DAQ on the board
                   next_token = strtok(NULL, " " );
                   daq = atoi(next_token);
              
                   // CHANNEL -  Get the CHANNEL on the DAQ
                   next_token = strtok(NULL, " " );
                   channel = atoi(next_token);
              
                   // Pick the appropriate command byte
                   command_byte = ltc_channels[daq];
                     
                   // Select the board
                   Board_Select(board);
                   
                   // Read from the specified LTC1859 channel
                   conv_result = ReadLTC1859Channel(command_byte, channel);
                   conv_result = ReadLTC1859Channel(command_byte, channel);
                
                  // Convert it to a voltage
                  if ((conv_result & 0x8000) == 0x8000)
                    {
                      //conv_result < 0: convert ADC code from two's
                      // complement to binary
                      conv_result = (conv_result ^ 0xFFFF)+(1<<(16-16));
                      sign = -1;
                    }
                    else
                      {
                      sign = 1;
                      }
                    
                    // Calculate the voltage
                    channel_voltage = sign * (float)conv_result;
                    channel_voltage = channel_voltage / (pow(2, 16-1)-1);
                    channel_voltage = channel_voltage * 10;
                  
                    float scaled_voltage = (channel_voltage /10.0) * 15.00;
                    
                   // Tell the user what you got
                   if (debug)
                   {
                     Serial.print("OK. The conversion result on channel: ");
                     Serial.print(channel);
                     Serial.print(" on DAQ number: ");
                     Serial.print(daq);
                     Serial.print(" on board number: ");
                     Serial.print(board);
                     Serial.print(" using Command Byte: ");
                     Serial.print(command_byte, BIN);
                     Serial.print(" is: ");
                     Serial.print(conv_result);
                     Serial.print(" and the voltage is: ");
                     Serial.print(channel_voltage);
                     Serial.print(" and the voltage scaled to 15.25 is: ");
                     Serial.println(scaled_voltage);
                   }
                   else
                   {
                     Serial.print("Converted Volts: ");
                     Serial.print(channel_voltage);
                     Serial.print(" Scaled volts: ");
                     Serial.println(scaled_voltage);
                   }

                   // select no function, disable system
                    SelectFunction(Y0);
                    SystemDisable();
                 }
                 
        // DAC OFFSET - Set the DAC Offset of the specified group
        //     DACOFF <BRD> <DAQ> <GROUP> <OFFSETVAL>
        else if( !strcmp(command_line,"DACOFF") )
            {
              // BOARD - Get the board id
              next_token = strtok(NULL, " " );
              board = atoi(next_token);
              
              // Get the DAQ on the board
              next_token = strtok(NULL, " " );
              daq = atoi(next_token);
       
              // GROUP - Get the group number
              next_token = strtok(NULL, " " );
              group_no = atoi(next_token);

              // OFFSET - Get the OFFSET to be set on that DAQ
              next_token = strtok(NULL, " " );
              data_val = atoi(next_token);
  
              // Rig the SPI setup for the AD5372
              RigSPIforDAQ();
        
              // Select the appropriate board
              Board_Select(board);      
              delay(2);
                 
               // Select the DAQ
              SelectFunction(daq);

              // MK ERROR - this prevents the write from occuring because /BRDSEL
              //            is then HIGH and the MISO SPI line is disabled and the
              //            write cannot work
              //SystemDisable();
              
              if (group_no == 1)
                byte1 = 0b00000010;
              else
                byte1 = 0b00000011;
                
              write(byte1, data_val);

              // delay 50 milli-seconds
              delay(50);
              
             // Tell the user what you got
             if(debug)
             {
              Serial.print("DACOFF GROUP: ");
              Serial.print(group_no);
              Serial.print(" Board: ");
              Serial.print(board);
              Serial.print(" Offset: ");
              Serial.print(data_val);
              Serial.print(" Command byte: ");
              Serial.println(byte1, BIN); 
             }
             else
             {
              Serial.print("DACOFF GROUP: ");
              Serial.print(group_no);
              Serial.print(" Board: ");
              Serial.print(board);
              Serial.print(" Offset: ");
              Serial.println(data_val);
             }
              // select no function, disable system
             SelectFunction(Y0);
             SystemDisable();
        } // END DACOFF                    
              

        // TREAD - TREAD <BRD>
        else if (!strcmp(command_line,"TREAD") )
                {                                   
                   // Get the board id
                   next_token = strtok(NULL, " " );
                   board = atoi(next_token);       

                   // Select the board
                   Board_Select(board);
                   delay(1);
                   
                   // RIG FOR  SPI for working with the ADT7310 Temperature chip
                   RigSPIforTREAD();
                   
                   Reset_ADT7310();
                 
                   delay(50);
                   
                   // Read from the board
                   temperature_value = Read_ADT7310(0x02, 16);
                   
                   T_deg_C = temperature(temperature_value,16); 
            
                   // Tell the user what you got

                   if (debug)
                     {
                        Serial.print("Board number: "); 
                        Serial.print(board);
                        Serial.print(" shows a Converted Temperature of: ");
                        Serial.print(T_deg_C,3); 
                        Serial.print(" C, [0x"); 
                        Serial.print(temperature_value,HEX); 
                        Serial.print("], [0b"); 
                        Serial.print(temperature_value,BIN); 
                        Serial.println("]"); 
                     }
                  else
                     {
                        Serial.print("Board number: "); 
                        Serial.print(board);
                        Serial.print(" Temp: ");
                        Serial.println(T_deg_C,3); 
                     }
                  // select no function, disable system
                  SelectFunction(Y0);
                  SystemDisable();
                } // END TREAD

        // HEARTBEAT
        else if (!strcmp(command_line,"HEARTBEAT") )
                {                    
                 // Send a greeting
                 Serial.println("AXRO Arduino Board: Lock Confirmed; Beacon Terra 1.");
  
                } // END COMMAND == HEARTBEAT

        // RESET -  RESET <BRD> 
        else if (!strcmp(command_line,"RESET") )
                {                                   
                  // Get the board id
                  next_token = strtok(NULL, " " );
                  board = atoi(next_token);       

                  // Select the board
                  Board_Select(board);
                  
                  // Now RESET that board
                  SelectFunction(RESET);
                  delay(1);
                  SelectFunction(Y0);
                  SystemDisable();

                  // Tell the user what you got

   	          Serial.print("RESET on Board: "); 
                  Serial.println(board);
                 }              
                       
       // GAIN - Set the Gain of the specified line
       //  GAIN <BRD> <DAQ> <CHANNEL> <GAINVAL>
       else if( !strcmp(command_line,"GAIN") )
            {
              // BOARD - Get the board id
              next_token = strtok(NULL, " " );
              board = atoi(next_token);
              
              // DAQ - Get the DAQ on the board
              next_token = strtok(NULL, " " );
              daq = atoi(next_token);
              
              // CHANNEL - Get the CHANNEL on the DAQ
              next_token = strtok(NULL, " " );
              channel = atoi(next_token);

              // GAIN - Get the GAIN to be set on that DAQ
              next_token = strtok(NULL, " " );
              gain = atoi(next_token);
              
            // Select the board
              Board_Select(board);
              delay(2);
             
              // Select the DAQ
              SelectFunction(daq);
              delay(2);
 
              // Calculate the command byte
              byte1 = Build_Group_Byte(channel);

              // Now set the mode bits for changing offset
              byte1 &= (0b01111111);
              
             // Write the data value out.
             write(byte1, gain);
             
             delay(5);             
              // Tell the user what you got
              if (debug)
               {
                 Serial.print("Gain set on CHANNEL number: ");
                 Serial.print(channel);
                 Serial.print(" on DAQ number: ");
                 Serial.print(daq);
                 Serial.print(" on board number: ");
                 Serial.print(board);
                 Serial.print(" with a gain of: ");
                 Serial.println(gain);    
               }
              else
                 {
                   Serial.print(" Gain Set: ");
                   Serial.println(gain); 
                 }  
                // select no function, disable system
                SelectFunction(Y0);
                SystemDisable();
                          
            }   // END GAIN
    
       // OFFSET - Set the Offset of the specified line
       //     OFFSET <BRD> <DAQ> <CHANNEL> <OFFSETVAL>
       else if( !strcmp(command_line,"OFFSET") )
            {
              // BOARD - Get the board id
              next_token = strtok(NULL, " " );
              board = atoi(next_token);
              
              // DAQ - Get the DAQ on the board
              next_token = strtok(NULL, " " );
              daq = atoi(next_token);
              
              // CHANNEL - Get the CHANNEL on the DAQ
              next_token = strtok(NULL, " " );
              channel = atoi(next_token);

              // OFFSET - Get the OFFSET to be set on that DAQ
              next_token = strtok(NULL, " " );
              offset = atoi(next_token);
              
              // Select the board
              Board_Select(board);
              delay(2);
             
              // Select the DAQ
              SelectFunction(daq);
              delay(2);
 
              // Calculate the command byte
              byte1 = Build_Group_Byte(channel);

              // Now set the mode bits for changing offset
              byte1 &= (0b10111111);
              
             // Map the user specified voltage to the DAQ range
             // of 0->65535
             //data_val = map(volts, -15.25, 15.25, 0, 65535);
             
             // Write the data value out.
             write(byte1, offset);
             
             delay(5);
             
              // Tell the user what you got
              if (debug)
                {
                 Serial.print("I'm going to set the offset of CHANNEL  number: ");
                 Serial.print(channel);
                 Serial.print(" on DAQ number: ");
                 Serial.print(daq);
                 Serial.print(" on board number: ");
                 Serial.print(board);
                 Serial.print(" with an offset of: ");
                 Serial.println(offset);   
                }
              else
                {
                 Serial.print(" with an offset of: ");
                 Serial.println(offset); 
                }    
              // select no function, disable system
              SelectFunction(Y0);
              SystemDisable();
     
            }       

        // QUIT
        else if (!strcmp(command_line,"QUIT") )
                {                    
                 // Send an ACK
                 Serial.println("QUIT received.");
                 Serial.end();
                } // END COMMAND == QUIT
                
        else if (!strcmp(command_line, "DEBUG"))
               {
                //debug = true;
                Serial.println("Debug ON");
               }
       
        else if (!strcmp(command_line, "NODEBUG"))
               {
                //debug = false;
                Serial.println("Debug OFF");
               }        
        else
            {
              Serial.print("***I have no idea what to do with this:");
              Serial.print(command);
              Serial.println("---");
            }
      
    } //ENDIF num_cmd_bytes > 0
    
}// END VOID LOOP
