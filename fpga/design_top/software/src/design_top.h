#ifndef DESIGN_TOP_H
#define DESIGN_TOP_H

#include <stdint.h>
#include <stdbool.h>

#ifndef DESIGN_TOP_DEFINES_H
#define DESIGN_TOP_DEFINES_H

#include <stdint.h>
#include <math.h>

// From design_top_defines.vh

#define kIntWordWidth 8
#define kVectorSize 16
#define kNumVectorLanes 16
#define kActWordWidth 32
#define kAccumWordWidth (2 * kIntWordWidth + kVectorSize - 1)

#define kActWordMax ((int32_t)(((int64_t)1 << (kActWordWidth - 1)) - 1))
#define kActWordMin (-kActWordMax)

#define WIDTH_DATA_AXI 32
#define WIDTH_ADDR_AXI 16

// Tranfer cycles counter
#define ADDR_TRANSFER_COUNTER 0x0400
#define ADDR_TRANSFER_COUNTER_EN 0x0400

// Compute cycles counter
#define ADDR_COMPUTE_COUNTER 0x0404

// Start cfg
#define ADDR_START_CFG 0x0404

// RVA IN Port
#define WIDTH_DATA_RVA_IN (kIntWordWidth * kVectorSize) // 128
#define WIDTH_ADDR_RVA_IN 24
#define WIDTH_RVA_IN (WIDTH_DATA_RVA_IN + WIDTH_ADDR_RVA_IN + 1 + (WIDTH_DATA_RVA_IN >> 3)) // 128 + 24 + 1 + 16 = 169
#define WIDTH_RVA_IN_32 192
#define LOOP_RVA_IN (WIDTH_RVA_IN_32 / 32) // 6
#define ADDR_RVA_IN_START 0x408

// RVA OUT Port
#define WIDTH_RVA_OUT WIDTH_DATA_RVA_IN // 128
#define ADDR_RVA_OUT_START 0x408
#define LOOP_RVA_OUT (WIDTH_RVA_OUT / WIDTH_DATA_AXI) // 4

// Activation Port
#define WIDTH_ACT_PORT (kActWordWidth * kNumVectorLanes) // 512
#define ADDR_ACT_PORT_START 0x0440
#define LOOP_ACT_PORT (WIDTH_ACT_PORT / WIDTH_DATA_AXI) // 16

#endif // DESIGN_TOP_DEFINES_H

// Function Prototypes (Aligned with SystemVerilog testbench)

// Basic AXI-lite access
int ocl_wr32(int bar_handle, uint16_t addr, uint32_t data);
int ocl_rd32(int bar_handle, uint16_t addr, uint32_t *data);

// RVA message formatting and transport
void rva_format(bool rw, uint32_t addr, const uint64_t data[2], uint32_t rva_msg[LOOP_RVA_IN]);
int ocl_rva_wr32(int bar_handle, const uint32_t rva_msg[LOOP_RVA_IN]);
int ocl_rva_r32(int bar_handle, uint64_t data_cmp[2], const uint32_t rva_in[LOOP_RVA_IN]);

// Golden model and verification
void randomize_data(uint64_t data[2]);
double round(double x);
void calculate_golden_activations(const uint64_t weights[kNumVectorLanes][2], const uint64_t input_written[2], int32_t golden_activations[kNumVectorLanes]);
void compare_act_vectors(const int32_t dut_vec[kNumVectorLanes], const int32_t golden_vec[kNumVectorLanes]);

// Performance counter functions
int start_data_transfer_counter(int bar_handle);
int stop_data_transfer_counter(int bar_handle);
int get_data_transfer_cycles(int bar_handle, uint32_t *cycles);
int get_compute_cycles(int bar_handle, uint32_t *cycles);

#endif // DESIGN_TOP_H
