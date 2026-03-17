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
  proc usercmd_post_assembly {} {
    # --- Timing: relax clock to give arbiter feedback paths slack ---
    # FPGA target is 125MHz (8ns), so 5ns (200MHz) HLS target is conservative.
    directive set -CLOCK_OVERHEAD 0
    directive set -CLOCK_PERIOD 5

    # --- Memory: force arrays to BRAM, not registers ---
    # Default thresholds (2048) let arrays up to 256 bytes stay as registers.
    # Lower to 256 bits (32 bytes) so only tiny arrays are registers.
    # This prevents large memory blocks from being synthesized as register files.
    directive set -REGISTER_THRESHOLD 256
    directive set -MEM_MAP_THRESHOLD 256

    # --- Scheduling: conservative area-focused goal ---
    # 'latency' goal aggressively packs operations, making timing harder.
    # 'area' goal is more conservative — easier to schedule, fewer timing violations.
    directive set -DESIGN_GOAL area

    # --- Components: allow slower but easier-to-place components ---
    directive set -COMPGRADE slow
  }
}

nvhls::run
