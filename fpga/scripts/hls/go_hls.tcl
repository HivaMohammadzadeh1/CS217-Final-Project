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

source $env(HLS_SCRIPTS)/nvhls_exec.tcl

namespace eval nvhls {
  # Pre-compile: set directives that must be applied before 'libraries' stage.
  proc usercmd_pre_compile {} {
    # --- Memory: force arrays to BRAM, not registers ---
    directive set -REGISTER_THRESHOLD 256
    directive set -MEM_MAP_THRESHOLD 256

    # --- Scheduling: conservative area-focused goal ---
    directive set -DESIGN_GOAL area

    # --- Components: allow slower but easier-to-place components ---
    directive set -COMPGRADE slow
  }

  # Post-assembly: only directives known to be safe at this stage.
  proc usercmd_post_assembly {} {
    # --- Timing: remove clock overhead margin for arbiter feedback paths ---
    # CLOCK_PERIOD is set to 5ns via Makefile CLK_PERIOD.
    directive set -CLOCK_OVERHEAD 0
  }
}

nvhls::run
