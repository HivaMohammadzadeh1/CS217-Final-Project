/**
 * Lab 1 FPGA Python Wrapper
 *
 * This C wrapper initializes the FPGA and provides functions
 * that can be called from Python via ctypes.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <fpga_pci.h>
#include <fpga_mgmt.h>

// FPGA slot configuration
static int fpga_slot_id = 0;
static pci_bar_handle_t pci_bar_handle = PCI_BAR_HANDLE_INIT;
static bool fpga_initialized = false;

// Lab 1 hardware parameters
#define VECTOR_SIZE 16
#define NUM_LANES 16
#define TILE_SIZE 16

// Register addresses (from Lab 1)
#define ADDR_PE_CONFIG      0x400010
#define ADDR_MANAGER_CONFIG 0x400020
#define ADDR_WEIGHT_BASE    0x500000
#define ADDR_INPUT_BASE     0x600000
#define ADDR_START_CFG      0x000404
#define ADDR_ACT_PORT_START 0x000440

/**
 * Initialize FPGA connection
 * Returns 0 on success, -1 on failure
 */
int fpga_init(int slot_id) {
    int rc;

    // Already initialized?
    if (fpga_initialized) {
        printf("FPGA already initialized\n");
        return 0;
    }

    fpga_slot_id = slot_id;

    // Initialize FPGA management library
    rc = fpga_mgmt_init();
    if (rc != 0) {
        fprintf(stderr, "Failed to initialize FPGA management library: %d\n", rc);
        return -1;
    }

    // Attach to FPGA BAR
    rc = fpga_pci_attach(fpga_slot_id, FPGA_APP_PF, APP_PF_BAR0, 0, &pci_bar_handle);
    if (rc != 0) {
        fprintf(stderr, "Failed to attach to FPGA slot %d: %d\n", fpga_slot_id, rc);
        return -1;
    }

    fpga_initialized = true;
    printf("✓ FPGA initialized successfully (slot %d)\n", fpga_slot_id);

    return 0;
}

/**
 * Get the bar handle for Python to use
 * Returns the bar handle value
 */
long fpga_get_bar_handle() {
    if (!fpga_initialized) {
        fprintf(stderr, "FPGA not initialized!\n");
        return -1;
    }
    return (long)pci_bar_handle;
}

/**
 * Write 32-bit value to FPGA register
 */
int fpga_write32(uint32_t addr, uint32_t data) {
    if (!fpga_initialized) {
        fprintf(stderr, "FPGA not initialized!\n");
        return -1;
    }

    int rc = fpga_pci_poke(pci_bar_handle, addr, data);
    if (rc != 0) {
        fprintf(stderr, "Write failed at 0x%08x: %d\n", addr, rc);
    }
    return rc;
}

/**
 * Read 32-bit value from FPGA register
 */
int fpga_read32(uint32_t addr, uint32_t *data) {
    if (!fpga_initialized) {
        fprintf(stderr, "FPGA not initialized!\n");
        return -1;
    }

    int rc = fpga_pci_peek(pci_bar_handle, addr, data);
    if (rc != 0) {
        fprintf(stderr, "Read failed at 0x%08x: %d\n", addr, rc);
    }
    return rc;
}

/**
 * Perform 16x16 matrix multiplication on Lab 1 FPGA
 *
 * A: 16x16 weight matrix (float32)
 * B: 16x16 input matrix (float32)
 * C: 16x16 output matrix (float32)
 *
 * Returns 0 on success, -1 on failure
 */
int fpga_matmul_16x16(float *A, float *B, float *C) {
    if (!fpga_initialized) {
        fprintf(stderr, "FPGA not initialized!\n");
        return -1;
    }

    int rc;

    // Step 1: Configure PE
    rc = fpga_write32(ADDR_PE_CONFIG, 0x00000001);
    rc |= fpga_write32(ADDR_PE_CONFIG + 4, 0x01010000);
    if (rc != 0) {
        fprintf(stderr, "PE configuration failed\n");
        return -1;
    }

    // Step 2: Write weight matrix A (quantize to INT8)
    for (int lane = 0; lane < 16; lane++) {
        uint32_t addr = ADDR_WEIGHT_BASE + (lane << 4);

        // Pack 16 float32 weights into INT8
        uint32_t data[4] = {0};
        for (int i = 0; i < 16; i++) {
            int8_t quantized = (int8_t)(A[lane * 16 + i] * 127.0f);
            int word_idx = i / 4;
            int byte_idx = i % 4;
            data[word_idx] |= ((uint32_t)(quantized & 0xFF)) << (byte_idx * 8);
        }

        // Write 4 words (128 bits total)
        for (int w = 0; w < 4; w++) {
            rc = fpga_write32(addr + w * 4, data[w]);
            if (rc != 0) return -1;
        }
    }

    // Step 3: Configure Manager
    rc = fpga_write32(ADDR_MANAGER_CONFIG, 0x00000100);
    if (rc != 0) {
        fprintf(stderr, "Manager configuration failed\n");
        return -1;
    }

    // Step 4: Process each column of B
    for (int col = 0; col < 16; col++) {
        // Pack input vector (column of B) as INT8
        uint32_t input_data[4] = {0};
        for (int i = 0; i < 16; i++) {
            int8_t quantized = (int8_t)(B[i * 16 + col] * 127.0f);
            int word_idx = i / 4;
            int byte_idx = i % 4;
            input_data[word_idx] |= ((uint32_t)(quantized & 0xFF)) << (byte_idx * 8);
        }

        // Write input vector
        for (int w = 0; w < 4; w++) {
            rc = fpga_write32(ADDR_INPUT_BASE + w * 4, input_data[w]);
            if (rc != 0) return -1;
        }

        // Trigger START
        rc = fpga_write32(ADDR_START_CFG, 1);
        if (rc != 0) return -1;

        // Wait for computation (adjust timing as needed)
        usleep(100);  // 100 microseconds

        // Trigger STOP
        rc = fpga_write32(ADDR_START_CFG, 0);
        if (rc != 0) return -1;

        usleep(50);

        // Read output activations (16 INT32 values)
        for (int i = 0; i < 16; i++) {
            uint32_t activation;
            rc = fpga_read32(ADDR_ACT_PORT_START + i * 4, &activation);
            if (rc != 0) return -1;

            // Convert INT32 to float32 (dequantize)
            int32_t signed_act = (int32_t)activation;
            C[i * 16 + col] = (float)signed_act / (127.0f * 127.0f);
        }
    }

    return 0;
}

/**
 * Cleanup FPGA resources
 */
void fpga_cleanup() {
    if (fpga_initialized) {
        fpga_pci_detach(pci_bar_handle);
        fpga_initialized = false;
        printf("FPGA resources released\n");
    }
}

/**
 * Check if FPGA is initialized
 */
int fpga_is_initialized() {
    return fpga_initialized ? 1 : 0;
}
