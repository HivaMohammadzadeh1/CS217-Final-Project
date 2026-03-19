# Environment variable settings
global env
set CATAPULT_HOME "/cad/mentor/2024.2_1/Mgc_home"
## Set the variable for file path prefixing 
set RTL_TOOL_SCRIPT_DIR /home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/vivado_concat_v
set RTL_TOOL_SCRIPT_DIR [file dirname [file normalize [info script] ] ]
puts "-- RTL_TOOL_SCRIPT_DIR is set to '$RTL_TOOL_SCRIPT_DIR' "
# Vivado Non-Project mode script starts here
puts "==========================================="
puts "Catapult driving Vivado in Non-Project mode"
puts "==========================================="
set outputDir /home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/vivado_concat_v
set outputDir $RTL_TOOL_SCRIPT_DIR
create_project -force ip_tcl_concat_v
   read_verilog ../concat_PECore.v
# set up XPM libraries for XPM-related IP like the Catapult Xilinx_FIFO
set_property XPM_LIBRARIES {XPM_CDC XPM_MEMORY XPM_FIFO} [current_project]
read_xdc $outputDir/concat_PECore.v.xv.sdc
set_property part xcvu9p-flgb2104-2-e [current_project]
