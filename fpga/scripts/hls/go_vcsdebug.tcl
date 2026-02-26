# Copyright (c) 2016-2019, NVIDIA CORPORATION.  All rights reserved.
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

source Catapult.ccs

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
flow run /SCVerify/launch_make ./scverify/Verify_concat_sim_${TOP_NAME}_v_vcs.mk SIMTOOL=vcs SIM_DUMP_FSDB=1 sim
exit
