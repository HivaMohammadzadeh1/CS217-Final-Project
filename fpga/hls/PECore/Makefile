#
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
#

ROOT            := ../..
SRC_PATH        := $(ROOT)/src/
SYSTEMC_DESIGN  := 1
COMPILER_FLAGS  := HLS_ALGORITHMICC MX_ONLY
CLK_PERIOD      := 4.0 
RUN_SCVERIFY    := 0
# Use := to ignore env REPO_TOP from old lab (cs217-lab-1-hiva)
REPO_TOP        := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
HLS_SCRIPTS     := $(REPO_TOP)/scripts/hls
INCLUDE         := $(REPO_TOP)/src/include

include $(HLS_SCRIPTS)/Makefile_hls
