//  Catapult Ultra Synthesis 2024.2_1/1143609 (Production Release) Wed Nov 13 18:57:31 PST 2024
//  
//          Copyright (c) Siemens EDA, 1996-2024, All Rights Reserved.
//                        UNPUBLISHED, LICENSED SOFTWARE.
//             CONFIDENTIAL AND PROPRIETARY INFORMATION WHICH IS THE
//                   PROPERTY OF SIEMENS EDA OR ITS LICENSORS.
//  
//  Running on Linux hiva@iron-04:825496 6.14.0-37-generic x86_64 aol
//  
//  Package information: SIFLIBS v27.2_1.0, HLS_PKGS v27.2_1.0, 
//                       SIF_TOOLKITS v27.2_1.0, SIF_XILINX v27.2_1.0, 
//                       SIF_ALTERA v27.2_1.0, CCS_LIBS v27.2_1.0, 
//                       CDS_PPRO v2024.1_3, CDS_DesignChecker v2024.2_1, 
//                       CDS_OASYS v21.1_3.1, CDS_PSR v24.1_0.20, 
//                       DesignPad v2.78_1.0, DesignAnalyzer 2024.2/1130210
//  
solution new -state initial
solution options defaults
solution options set /Interface/DefaultResetClearsAllRegs yes
solution options set /Input/CompilerFlags {-D_SYNTHESIS_ -DHLS_CATAPULT -DHLS_ALGORITHMICC -DCONNECTIONS_ACCURATE_SIM -DSC_INCLUDE_DYNAMIC_PROCESSES }
solution options set /Input/SearchPath {. /cad/mentor/2024.2_1/Mgc_home/shared/pkgs/matchlib/cmod/include /cad/mentor/2024.2_1/Mgc_home/shared/pkgs/matchlib_connections/include /cad/mentor/2024.2_1/Mgc_home/shared/include /cad/xilinx/vitis/2020.1/Vitis/2020.1/cardano/tps/boost_1_64_0 /home/users/hiva/CS217-Final-Project/fpga/src/include}
solution options set /Output/OutputVHDL false
solution options set /Output/PackageOutput true
solution options set /Output/PackageStaticFiles true
solution options set /Output/PrefixStaticFiles true
solution options set /Output/Basename {schedule {cyc${ENTITY}} extract {${ENTITY}}}
solution options set /Output/SubBlockNamePrefix {${ENTITY}_}
solution options set /Flows/QuestaSIM/Path /cad/mentor/2021.4/questasim/bin
solution options set /Flows/SCVerify/USE_QUESTASIM false
solution options set /Flows/SCVerify/USE_OSCI false
solution options set /Flows/SCVerify/USE_VCS true
solution options set /Flows/VCS/VCS_HOME /cad/synopsys/vcs/U-2023.03/
solution options set /Flows/VCS/VG_GNU_PACKAGE /cad/synopsys/vcs_gnu_package/S-2021.09/gnu_9/linux
solution options set /Flows/VCS/VG_ENV64_SCRIPT source_me.csh
solution file add ../../src/PECore/PECore.h -type CHEADER
solution file add ../../src/PECore/testbench.cpp -type C++ -exclude true
directive set -LOGIC_OPT false
directive set -FSM_BINARY_ENCODING_THRESHOLD 64
directive set -REGISTER_SHARING_MAX_WIDTH_DIFFERENCE 8
directive set -CHAN_IO_PROTOCOL use_library
directive set -ARRAY_SIZE 1024
directive set -STALL_FLAG_SV off
directive set -STALL_FLAG false
directive set -READY_FLAG {}
directive set -ON_THE_FLY_PROTOTYPING false
directive set -CLUSTER_ADDTREE_IN_COUNT_THRESHOLD 0
directive set -SPECULATE true
directive set -MERGEABLE true
directive set -REGISTER_THRESHOLD 256
directive set -MEM_MAP_THRESHOLD 32
directive set -REG_MAX_FANOUT 0
directive set -NO_X_ASSIGNMENTS true
directive set -SAFE_FSM false
directive set -REGISTER_SHARING_LIMIT 0
directive set -ASSIGN_OVERHEAD 0
directive set -TIMING_CHECKS true
directive set -MUXPATH true
directive set -REALLOC true
directive set -UNROLL no
directive set -IO_MODE super
directive set -IDLE_SIGNAL {}
directive set -TRANSACTION_DONE_SIGNAL true
directive set -DONE_FLAG {}
directive set -START_FLAG {}
directive set -TRANSACTION_SYNC ready
directive set -CLOCK_OVERHEAD 20.000000
directive set -OPT_CONST_MULTS use_library
directive set -CHARACTERIZE_ROM false
directive set -PROTOTYPE_ROM true
directive set -ROM_THRESHOLD 64
directive set -CLUSTER_ADDTREE_IN_WIDTH_THRESHOLD 0
directive set -CLUSTER_OPT_CONSTANT_INPUTS true
directive set -CLUSTER_RTL_SYN false
directive set -CLUSTER_FAST_MODE false
directive set -CLUSTER_TYPE combinational
directive set -PIPELINE_RAMP_UP true
go new
directive set -DESIGN_GOAL latency
directive set -FSM_ENCODING binary
directive set -REGISTER_IDLE_SIGNAL false
directive set -RESET_CLEARS_ALL_REGS yes
directive set -BUILTIN_MODULARIO false
directive set -GATE_REGISTERS true
directive set -GATE_EFFORT high
directive set -GATE_MIN_WIDTH 4
directive set -GATE_EXPAND_MIN_WIDTH 4
directive set -DSP_EXTRACTION yes
directive set -DSP_EXTRACTION_UNFOLD_MAC true
go analyze
solution design set PECore -top
directive set -CLOCKS {clk {-CLOCK_PERIOD 6.0 -CLOCK_EDGE rising -CLOCK_UNCERTAINTY 0.0 -CLOCK_HIGH_TIME 3.0 -RESET_SYNC_NAME rst -RESET_ASYNC_NAME arst_n -RESET_KIND sync -RESET_SYNC_ACTIVE high -RESET_ASYNC_ACTIVE low -ENABLE_ACTIVE high}}
go compile
directive set -CLOCKS {clk {-CLOCK_PERIOD 6.0 -CLOCK_UNCERTAINTY 0.0 -CLOCK_HIGH_TIME 3.0}}
solution library add mgc_Xilinx-VIRTEX-uplus-2_beh -- -rtlsyntool Vivado -manufacturer Xilinx -family VIRTEX-uplus -speed -2 -part xcvu9p-flgb2104-2-e
solution library add Xilinx_RAMS
solution library add Xilinx_ROMS
solution library add Xilinx_FIFO
solution library add ccs_fpga_hic
go libraries
go extract
