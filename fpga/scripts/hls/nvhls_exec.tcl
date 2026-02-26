# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

source $env(HLS_SCRIPTS)/run_hls_global_setup.tcl

namespace eval nvhls {
    proc run {} {

        # Get all input variables set from Makefile
        global env
        set USER_VARS {TOP_NAME CLK_PERIOD SRC_PATH SEARCH_PATH HLS_CATAPULT RUN_SCVERIFY COMPILER_FLAGS SYSTEMC_DESIGN RUN_CDESIGN_CHECKER}

        echo "***USER SETTINGS***"
        foreach var $USER_VARS {
            if [info exists env($var)] {
                echo "$var = $env($var)"
                set $var $env($var)
            } else {
                echo "Warning: $var not set by user"
                set $var ""
            }
        }
        options set Input/SearchPath ". $SEARCH_PATH"
        set_input_files $SRC_PATH $TOP_NAME $SYSTEMC_DESIGN
        set_compiler_flags $HLS_CATAPULT $COMPILER_FLAGS
        usercmd_pre_analyze
        go analyze
        setup_libs
        setup_clocks $CLK_PERIOD
        setup_hier $TOP_NAME
        usercmd_pre_compile
        go compile
        if { $RUN_CDESIGN_CHECKER eq "1" } { run_design_checker; exit }
        go libraries
        go assembly
        usercmd_post_assembly
        go architect
        usercmd_post_architect
        go allocate
        go schedule
        go dpfsm
        go extract
        project save
        if { $RUN_SCVERIFY eq "0" } {
          flow run /SCVerify/launch_make ./scverify/Verify_concat_sim_${TOP_NAME}_v_vcs.mk SIMTOOL=vcs sim INVOKE_ARGS= CCS_VCD_FILE=./default.fsdb CCS_VCD_TIMES=0,ns,end,ns USE_FSDB=true
          project save
        }

        if { $RUN_SCVERIFY eq "1" } {
          flow run /SCVerify/launch_make ./scverify/Verify_concat_sim_${TOP_NAME}_v_vcs.mk SIMTOOL=vcs sim INVOKE_ARGS= CCS_VCD_FILE=./default.fsdb CCS_VCD_TIMES=0,ns,end,ns USE_FSDB=true
          go switching
          flow run /PowerAnalysis/report_pre_pwropt_Verilog
          go power
          flow run /PowerAnalysis/report_post_pwropt_Verilog
          project save
        }
        if { $RUN_SCVERIFY eq "2" } {
          go switching
          flow run /PowerAnalysis/report_pre_pwropt_Verilog
          project save
        }
        if { $RUN_SCVERIFY eq "3" } {
          go switching
          project save
        }
        if { $RUN_SCVERIFY eq "4" } {
          flow run /SCVerify/launch_make ./scverify/Verify_concat_sim_${TOP_NAME}_v_vcs.mk SIMTOOL=vcs sim INVOKE_ARGS= CCS_VCD_FILE=./default.fsdb CCS_VCD_TIMES=0,ns,end,ns USE_FSDB=true
          project save
        }
        exit
    }

    proc set_input_files {SRC_PATH TOP_NAME SYSC} {
        if { $SYSC eq "1" } {
            set type SYSTEMC
            set ext h
        } else {
            set type C++
            set ext cpp
        }
        solution file add [list $SRC_PATH/$TOP_NAME/$TOP_NAME.$ext] -type $type
        solution file add [list $SRC_PATH/$TOP_NAME/testbench.cpp] -type $type -exclude true
    }

    proc set_compiler_flags {HLS_CATAPULT COMPILER_FLAGS} {
        if { $HLS_CATAPULT eq "1" } {
            set HLS_CATAPULT_FLAG "-DHLS_CATAPULT"
        } else {
            set HLS_CATAPULT_FLAG ""
        }
        set FLAG_STR ""
        foreach flag $COMPILER_FLAGS {
            append FLAG_STR "-D$flag "
        }
        options set Input/CompilerFlags "-D_SYNTHESIS_ $HLS_CATAPULT_FLAG $FLAG_STR"
    }

    proc setup_libs {} {
        # Technology Library
        # solution library add nangate-45nm_beh -- -rtlsyntool OasysRTL -vendor Nangate -technology 045nm
        # solution library add ccs_sample_mem

        # WARNING: Catapult HLS does not have official support for VU47P parts used in AWS F2 instances.
        # Using VU9P part library in AWS F1 instances instead.
        # solution library add mgc_Xilinx-VIRTEX-uplus-2_beh -- -rtlsyntool Vivado -manufacturer Xilinx -family VIRTEX-uplus -speed -2 -part xcvu47p-fsvh2892-2-e
        solution library add mgc_Xilinx-VIRTEX-uplus-2_beh -- -rtlsyntool Vivado -manufacturer Xilinx -family VIRTEX-uplus -speed -2 -part xcvu9p-flgb2104-2-e
        solution library add Xilinx_RAMS
        solution library add Xilinx_ROMS
        solution library add Xilinx_FIFO
        solution library add ccs_fpga_hic
    }

    proc setup_clocks {period} {
        set name clk
        set CLK_PERIODby2 [expr $period/2]
        directive set -CLOCKS "$name \"-CLOCK_PERIOD $period -CLOCK_EDGE rising -CLOCK_UNCERTAINTY 0.0 -CLOCK_HIGH_TIME $CLK_PERIODby2 -RESET_SYNC_NAME rst -RESET_ASYNC_NAME arst_n -RESET_KIND sync -RESET_SYNC_ACTIVE high -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high\"    "
        directive set -CLOCK_NAME $name
    }

    proc setup_hier {TOP_NAME} {
        directive set -DESIGN_HIERARCHY "$TOP_NAME"
    }

    proc run_design_checker {} {
      flow run /CDesignChecker/write_options {{-abr -severity error } {-abw -severity error } {-acc -severity warning } {-acs -severity warning } {-aic -severity warning } {-aob -severity error } {-apt -severity info } {-cas -severity error } {-ccc -severity warning } {-cia -severity warning } {-cns -severity warning } {-cwb -severity warning } {-dbz -severity error } {-fxd -severity warning } {-ise -severity error } {-mxs -severity info } {-ovl -severity error } {-pdd -severity warning } {-rrt -severity error } {-stf -severity info } {-sud -severity warning } {-umr -severity error }}
      #flow run /CDesignChecker/write_options {-abr -abw -aic -aob -apt -cas -ccc -cia -cns -cwb -dbz -fxd -ise -mxs -osl -ovl -pdd -rrt -stf -sud -umr } {-abr -abw -cas -dbz -ise -umr }
      #flow run /CDesignChecker/launch_sleccpc_sh ./CDesignChecker/design_checker.sh
      flow run /CDesignChecker/launch_sleccpc_sh ./SLEC_CPC/slec_cpc.sh
    }

    proc usercmd_pre_analyze {} {}
    proc usercmd_pre_compile {} {}
    proc usercmd_post_assembly {} {}
    proc usercmd_post_architect {} {}
}
