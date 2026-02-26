// ============================================================================
// Amazon FPGA Hardware Development Kit
//
// Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Amazon Software License (the "License"). You may not use
// this file except in compliance with the License. A copy of the License is
// located at
//
//    http://aws.amazon.com/asl/
//
// or in the "license" file accompanying this file. This file is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or
// implied. See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

`ifndef design_top_DEFINES
`define design_top_DEFINES

        // Put module name of the CL design here. This is used to instantiate in top.sv
`define CL_NAME design_top

`define DDR_A_ABSENT
`define DDR_B_ABSENT


`endif
localparam int kIntWordWidth = 8;
localparam int kVectorSize = 16;
localparam int kNumVectorLanes = 16;
localparam int kActWordWidth = 32;
localparam int kAccumWordWidth = 2*kIntWordWidth + kVectorSize - 1;

localparam int kActWordMax = (1 << (kActWordWidth-1)) -1;
localparam int kActWordMin = -kActWordMax;

localparam int WIDTH_DATA_AXI = 32;
localparam int WIDTH_ADDR_AXI = 16;

// Tranfer cycles counter
localparam int ADDR_TRANSFER_COUNTER = 16'h0400;
localparam int ADDR_TRANSFER_COUNTER_EN = 16'h0400;

// Compute cycles counter
localparam int ADDR_COMPUTE_COUNTER = 16'h0404;

// Start cfg
localparam int ADDR_START_CFG = 16'h0404;

// RVA IN Port
localparam int WIDTH_DATA_RVA_IN = kIntWordWidth * kVectorSize;
localparam int WIDTH_ADDR_RVA_IN = 24;
localparam int WIDTH_RVA_IN = WIDTH_DATA_RVA_IN + WIDTH_ADDR_RVA_IN + 1 + (WIDTH_DATA_RVA_IN >> 3); // Data + Addr + Start + Parity
localparam int WIDTH_RVA_IN_32 = 32 * (int((WIDTH_RVA_IN + 31)/ 32));
localparam int LOOP_RVA_IN = WIDTH_RVA_IN_32 / 32;
localparam int ADDR_RVA_IN_START = 16'h408;

localparam int WIDTH_SPILLOVER = WIDTH_RVA_IN - (LOOP_RVA_IN - 1)*32;
localparam int WIDTH_EXISTING = (LOOP_RVA_IN - 1)*32;

// RVA OUT Port
localparam int WIDTH_RVA_OUT = WIDTH_DATA_RVA_IN;
localparam int ADDR_RVA_OUT_START = 16'h408;
localparam int LOOP_RVA_OUT = WIDTH_RVA_OUT / WIDTH_DATA_AXI;

// Activation Port
localparam int WIDTH_ACT_PORT = kActWordWidth * kNumVectorLanes;
localparam int ADDR_ACT_PORT_START = 16'h0440;
localparam int LOOP_ACT_PORT = WIDTH_ACT_PORT / WIDTH_DATA_AXI;

// Testbench randomization
localparam int LOOP_RANDOMIZE = WIDTH_DATA_RVA_IN / WIDTH_DATA_AXI;