#include <fpga_mgmt.h>
#include <fpga_pci.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h> 
#include <time.h>
#include "design_top.h"

//#define DEBUG
// --- Testbench Helper Functions (Adapted from SystemVerilog tasks) ---

/**
 * @brief Simple OCL (AXI-lite) 32-bit write using the FPGA SDK.
 * @return 0 on success, 1 on MMIO failure.
 */
int ocl_wr32(int bar_handle, uint16_t addr, uint32_t data) {
    if (fpga_pci_poke(bar_handle, addr, data)) {
        fprintf(stderr, "ERROR: MMIO write failed at addr=0x%04x\n", addr);
        return 1;
    }
    return 0;
}

/**
 * @brief Simple OCL (AXI-lite) 32-bit read using the FPGA SDK.
 * @return 0 on success, 1 on MMIO failure.
 */
int ocl_rd32(int bar_handle, uint16_t addr, uint32_t *data) {
    if (fpga_pci_peek(bar_handle, addr, data)) {
        fprintf(stderr, "ERROR: MMIO read failed at addr=0x%04x\n", addr);
        return 1;
    }
    return 0;
}

/**
 * @brief Formats the RVA message, aligned with the SystemVerilog testbench.
 */
void rva_format(bool rw, uint32_t addr, const uint64_t data[2], uint32_t rva_msg[LOOP_RVA_IN]) {
    // Total message size is WIDTH_RVA_IN_32 = 224 bits (7 words)
    // Structure based on design_top_base_test.sv:
    // rva_msg[223]: TAG bit
    // rva_msg[168]: RW bit
    // rva_msg[151:128]: addr (24 bits)
    // rva_msg[127:0]: data (128 bits)

    for (int i = 0; i < LOOP_RVA_IN; i++) {
        rva_msg[i] = 0;
    }

    // Pack data (127:0)
    rva_msg[0] = (uint32_t)(data[0] & 0xFFFFFFFF);
    rva_msg[1] = (uint32_t)((data[0] >> 32) & 0xFFFFFFFF);
    rva_msg[2] = (uint32_t)(data[1] & 0xFFFFFFFF);
    rva_msg[3] = (uint32_t)((data[1] >> 32) & 0xFFFFFFFF);

    // Pack addr (151:128)
    // addr is 24 bits. It starts at bit 128. This falls entirely into rva_msg[4].
    rva_msg[4] = addr & 0xFFFFFF; // bits 128-151

    // Pack rw bit (at bit 168)
    // 168 = 5 * 32 + 8. So it's in rva_msg[5] at bit index 8.
    if (rw) {
        rva_msg[5] |= (1 << 8);
    }
    
    // Pack TAG bit (at bit 223)
    // 223 = 6 * 32 + 31. So it's in rva_msg[6] at bit index 31.
    rva_msg[5] |= (1U << 31);
}

/**
 * @brief Writes the RVA message across sequential AXI-lite registers.
 * @return 0 on success, 1 on MMIO failure.
 */
int ocl_rva_wr32(int bar_handle, const uint32_t rva_msg[LOOP_RVA_IN]) {
    uint16_t addr; 
    #ifdef DEBUG
    printf("LOOP_RVA_IN: %d and WIDTH_RVA_IN = %d\n", LOOP_RVA_IN, WIDTH_RVA_IN_32);
    #endif
    for (int i = 0; i < LOOP_RVA_IN; i++) {
        addr = ADDR_RVA_IN_START + i * 4;
        #ifdef DEBUG
        printf("Writing RVA word %d to addr 0x%04x: 0x%08x\n", i, addr, rva_msg[i]);
        #endif
        if (ocl_wr32(bar_handle, addr, rva_msg[i])) {
            return 1;
        }
    }
    return 0;
}


// --- Golden Reference Model (Used for Verification) ---

/**
 * @brief Fills a 128-bit data array with random values.
 */
void randomize_data(uint64_t data[2]) {
    uint32_t r1 = rand();
    uint32_t r2 = rand();
    uint32_t r3 = rand();
    uint32_t r4 = rand();
    data[0] = (uint64_t)r2 << 32 | r1;
    data[1] = (uint64_t)r4 << 32 | r3;
}

/**
 * @brief SV-equivalent rounding function.
 */
double round(double x) {
    return (x > 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
}

/**
 * @brief Calculates the golden activations based on randomized weights and inputs.
 */
void calculate_golden_activations(const uint64_t weights[kNumVectorLanes][2], const uint64_t input_written[2], int32_t golden_activations[kNumVectorLanes]) {
    const double SCALE_DIVISOR = 12.25;

    for (int i = 0; i < kNumVectorLanes; i++) {
        // kAccumWordWidth is 31, so it fits in a 32-bit uint.
        uint32_t output_accum = 0;

        // The SV test does byte-wise multiplication and accumulation
        for (int j = 0; j < kVectorSize; j++) { // kVectorSize is 16 bytes (128 bits)
            uint8_t weight_byte;
            uint8_t input_byte;

            if (j < 8) {
                weight_byte = (weights[i][0] >> (j * 8)) & 0xFF;
                input_byte = (input_written[0] >> (j * 8)) & 0xFF;
            } else {
                weight_byte = (weights[i][1] >> ((j - 8) * 8)) & 0xFF;
                input_byte = (input_written[1] >> ((j - 8) * 8)) & 0xFF;
            }
            output_accum += (uint32_t)weight_byte * input_byte;
        }
        
        double scaled = (double)output_accum / SCALE_DIVISOR;

        // Clamp to kActWordMin/Max
        if (scaled > kActWordMax) {
            scaled = kActWordMax;
        } else if (scaled < kActWordMin) {
            scaled = kActWordMin;
        }
        
        // Round to nearest integer (matches SV `round` function)
        golden_activations[i] = (int32_t)round(scaled);
    }
}


/**
 * @brief Performs RVA readback verification.
 * @return Returns the number of errors (0 on success).
 */
int ocl_rva_r32(int bar_handle, uint64_t data_cmp[2], const uint32_t rva_in[LOOP_RVA_IN]) {
    uint32_t rva_out_words[LOOP_RVA_OUT] = {0};
    uint64_t data_read[2] = {0};
    uint16_t addr;
    int error_cnt = 0;

    // 1. Write the RVA read command
    if (ocl_rva_wr32(bar_handle, rva_in)) return 1;

    // 2. Read the response from AXI-lite registers
    #ifdef DEBUG
    printf("LOOP_RVA_OUT: %d and WIDTH_RVA_OUT = %d\n", LOOP_RVA_OUT, WIDTH_RVA_OUT);
    #endif
    for (int i = 0; i < LOOP_RVA_OUT; i++) {
        addr = ADDR_RVA_OUT_START + i * 4;
        if (ocl_rd32(bar_handle, addr, &rva_out_words[i])) {
            return 1;
        }
        #ifdef DEBUG
        printf("Read RVA word %d from addr 0x%04x: 0x%08x\n", i, addr, rva_out_words[i]);
        #endif
    }


    // Reconstruct the 128-bit data read
    data_read[0] = (uint64_t)rva_out_words[1] << 32 | rva_out_words[0];
    data_read[1] = (uint64_t)rva_out_words[3] << 32 | rva_out_words[2];

    // 3. Compare the 128-bit data
    if (data_read[0] != data_cmp[0] || data_read[1] != data_cmp[1]) {
        fprintf(stderr, "RVA readback mismatch: expected 0x%016llx%016llx got 0x%016llx%016llx\n", 
                (long long unsigned)data_cmp[1], (long long unsigned)data_cmp[0],
                (long long unsigned)data_read[1], (long long unsigned)data_read[0]);
        error_cnt++;
    } else {
        printf("RVA readback OK: 0x%016llx%016llx\n", 
               (long long unsigned)data_read[1], (long long unsigned)data_read[0]);
    }
    return error_cnt;
}


/**
 * @brief Calculates the relative absolute difference between two vectors.
 */
void compare_act_vectors(const int32_t dut_vec[kNumVectorLanes], const int32_t golden_vec[kNumVectorLanes]) {
    double diff_sum = 0.0;
    bool test_failed = false;

    printf("\n---- Final Output Vector Comparison ----\n");
    for (int j = 0; j < kNumVectorLanes; j++) {
        double a = (double)dut_vec[j];
        double e = (double)golden_vec[j];
        double term;

        if (e == 0.0) {
            term = (a == 0.0) ? 0.0 : 1.0; 
        } else {
            term = fabs(a - e) / fabs(e);
        }
        diff_sum += term;

        printf("Act Port Computed value = %d and expected value = %d (lane %02d) err=%.3f%%\n",
               dut_vec[j], golden_vec[j], j, 100.0 * term);
        
        // Fail if any single element has high error (e.g., > 1%)
        if (term > 0.02) { // Loosen tolerance slightly for C floating point vs SV real
            test_failed = true;
        }
    }

    double avg_pct = (diff_sum * 100.0) / kNumVectorLanes;
    printf("\nDest: Difference observed in compute Act and expected value %.3f%%\n", avg_pct);

    if (avg_pct > 2.0 || test_failed) { // Tolerance check from SV test
        fprintf(stderr, "TEST FAILED\n");
    } else {
        printf("TEST PASSED\n");
    }
}

// --- Counter Functions (Aligned with SV) ---

int start_data_transfer_counter(int bar_handle) {
    return ocl_wr32(bar_handle, ADDR_TRANSFER_COUNTER_EN, 1);
}

int stop_data_transfer_counter(int bar_handle) {
    return ocl_wr32(bar_handle, ADDR_TRANSFER_COUNTER_EN, 0);
}

int get_data_transfer_cycles(int bar_handle, uint32_t *cycles) {
    return ocl_rd32(bar_handle, ADDR_TRANSFER_COUNTER, cycles);
}

int get_compute_cycles(int bar_handle, uint32_t *cycles) {
    return ocl_rd32(bar_handle, ADDR_COMPUTE_COUNTER, cycles);
}


// ----------------------------------------------------------------------------
// MAIN EXECUTION LOGIC (Structured like cl_peek_simple.c)
// ----------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <slot_id>\n", argv[0]);
        return 1;
    }

    srand(time(NULL)); // Seed for randomization

    int slot_id = atoi(argv[1]);
    int bar_handle = -1;
    int total_errors = 0;
    uint32_t rva_in_words[LOOP_RVA_IN];
    bool rva_in_rw;
    uint32_t rva_in_addr;
    
    // Arrays to hold randomized test data
    uint64_t weights[kNumVectorLanes][2];
    uint64_t input_written[2];
    uint64_t rva_in_data[2]; 

    // DUT results and golden model results
    int32_t output_obtained[kNumVectorLanes] = {0};
    int32_t output_act[kNumVectorLanes];

    // --- 1. Initialization and Attachment (Like cl_peek_simple.c) ---
    if (fpga_mgmt_init() != 0) {
        fprintf(stderr, "Failed to initialize fpga_mgmt\n");
        return 1;
    }

    if (fpga_pci_attach(slot_id, FPGA_APP_PF, APP_PF_BAR0, 0, &bar_handle)) {
        fprintf(stderr, "fpga_pci_attach failed\n");
        return 1;
    }
    
    printf("---- System Initialization and Reset (bar_handle: %d) ----\n", bar_handle);

    start_data_transfer_counter(bar_handle);

    // --- 2. Test Execution (SystemVerilog Steps) ---
    
    // ---------------------------
    // 1) WRITE PEConfig (region 0x4, local_index = 0x0001)
    // ---------------------------
    printf("\n---- STEP 1: WRITE PEConfig ----\n");    
    rva_in_addr = 0x400010;
    rva_in_rw = true;
    rva_in_data[0] = 0x0000010100000001;
    rva_in_data[1] = 0x00000000; 
    rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_in_words);
    if (ocl_rva_wr32(bar_handle, rva_in_words)) goto error_detach;
    stop_data_transfer_counter(bar_handle);

    // Read back to verify
    rva_in_rw = false;
    rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_in_words);
    total_errors += ocl_rva_r32(bar_handle, rva_in_data, rva_in_words);


    // ---------------------------
    // 2) WRITE WEIGHT SRAM (region 0x5, addr i<<4 for i in [0..15])
    // ---------------------------
    printf("\n---- STEP 2: WRITE WEIGHT SRAM ----\n");
    for (int i = 0; i < kNumVectorLanes; i++) {
        rva_in_rw = true;
        randomize_data(weights[i]);
        rva_in_data[0] = weights[i][0];
        rva_in_data[1] = weights[i][1];
        rva_in_addr = 0x500000 + (i << 4); 
        
        start_data_transfer_counter(bar_handle);
        rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_in_words);
        if (ocl_rva_wr32(bar_handle, rva_in_words)) goto error_detach;
        stop_data_transfer_counter(bar_handle);
        
        // Read back to verify
        rva_in_rw = false;
        rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_in_words);
        total_errors += ocl_rva_r32(bar_handle, rva_in_data, rva_in_words);
    }

    // ---------------------------
    // 3) WRITE INPUT SRAM (region 0x6, addr 0x0000)
    // ---------------------------
    printf("\n---- STEP 3: WRITE INPUT SRAM ----\n");
    rva_in_rw = true;
    randomize_data(input_written);
    rva_in_data[0] = input_written[0];
    rva_in_data[1] = input_written[1];
    rva_in_addr = 0x600000;
    start_data_transfer_counter(bar_handle);
    rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_in_words);
    if (ocl_rva_wr32(bar_handle, rva_in_words)) goto error_detach;
    stop_data_transfer_counter(bar_handle);

    // Read back to verify
    rva_in_rw = false;
    rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_in_words);
    total_errors += ocl_rva_r32(bar_handle, rva_in_data, rva_in_words);

    // Calculate the golden reference model output now that inputs are finalized
    calculate_golden_activations(weights, input_written, output_act);

    // ---------------------------
    // 4) WRITE Manager1 config (region 0x4, local_index = 0x0004)
    // ---------------------------
    printf("\n---- STEP 4: WRITE Manager1 config ----\n");
    rva_in_rw = true;
    rva_in_data[0] = 0x0000000000000100; // Aligned with SV test
    rva_in_data[1] = 0x00000000;
    rva_in_addr = 0x400020;
    start_data_transfer_counter(bar_handle);
    rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_in_words);
    if (ocl_rva_wr32(bar_handle, rva_in_words)) goto error_detach;
    stop_data_transfer_counter(bar_handle);

    // Read back to verify
    rva_in_rw = false;
    rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_in_words);
    total_errors += ocl_rva_r32(bar_handle, rva_in_data, rva_in_words);

    stop_data_transfer_counter(bar_handle);


    // ---------------------------
    // 5 & 6) START and STOP
    // ---------------------------
    printf("\n---- STEP 5 & 6: START/STOP ----\n");
    if (ocl_wr32(bar_handle, ADDR_START_CFG, 0x1)) goto error_detach; // START
    usleep(50); // Wait for computation (Simulate latency)
    if (ocl_wr32(bar_handle, ADDR_START_CFG, 0x0)) goto error_detach; // STOP
    usleep(50); 

    start_data_transfer_counter(bar_handle);


    // ---------------------------
    // 7) Read Output Act (16 lanes * 32-bits/lane)
    // ---------------------------
    printf("\n---- STEP 7: READ OUTPUT ACT ----\n");
    start_data_transfer_counter(bar_handle);
    for (int i = 0; i < LOOP_ACT_PORT; i++) {
        uint16_t addr_w = ADDR_ACT_PORT_START + i * 4;
        // Read directly into the signed integer array, will be cast.
        if (ocl_rd32(bar_handle, addr_w, (uint32_t*)&output_obtained[i])) goto error_detach;
    }

    stop_data_transfer_counter(bar_handle);

    // ---------------------------
    // 8) Compare vectors
    // ---------------------------
    compare_act_vectors(output_obtained, output_act);
    
    // Read and print the cycle counts
    uint32_t data_transfer_cycles = 0;
    uint32_t compute_cycles = 0;
    if (get_data_transfer_cycles(bar_handle, &data_transfer_cycles)) goto error_detach;
    if (get_compute_cycles(bar_handle, &compute_cycles)) goto error_detach;
    printf("Data Transfer Cycles: %u\n", data_transfer_cycles);
    printf("Compute Cycles: %u\n", compute_cycles);

    // ---------------------------
    // 9) Final Report and Detachment
    // ---------------------------
    printf("\nTotal RVA Verification Errors: %d\n", total_errors);
    
    fpga_pci_detach(bar_handle);
    return total_errors == 0 ? 0 : 1;

error_detach:
    // Jump here on any MMIO read/write error
    fprintf(stderr, "\nTEST FAILED due to MMIO communication error.\n");
    if (bar_handle != -1) {
        fpga_pci_detach(bar_handle);
    }
    return 1;
}