/*
 * Repo-local FPGA bridge for Python ctypes integration.
 *
 * This wrapper talks to the current project design at:
 *   fpga/design_top/software/src/design_top.c
 *
 * The key difference from the earlier Lab 1 bridge is that this project's
 * design writes weights/config via packed RVA messages over BAR0, not by
 * directly poking logical RVA addresses like 0x500000 into BAR0.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <fpga_mgmt.h>
#include <fpga_pci.h>

#include "design_top.h"

#define PE_CONFIG_ADDR 0x400010u
#define MANAGER_CONFIG_ADDR 0x400020u
#define WEIGHT_BASE_ADDR 0x500000u
#define INPUT_BASE_ADDR 0x600000u
#define SCALE_DIVISOR 12.25f

static int fpga_slot_id = 0;
static pci_bar_handle_t pci_bar_handle = PCI_BAR_HANDLE_INIT;
static bool fpga_initialized = false;

static int check_slot_config(int slot_id) {
    struct fpga_mgmt_image_info info = {0};
    int rc = fpga_mgmt_describe_local_image(slot_id, &info, 0);
    if (rc != 0) {
        fprintf(stderr, "Unable to query FPGA slot %d (rc=%d)\n", slot_id, rc);
        return -1;
    }
    if (info.status != FPGA_STATUS_LOADED) {
        fprintf(stderr, "FPGA slot %d is not loaded with an image\n", slot_id);
        return -1;
    }
    return 0;
}

static int clip_int8(float value) {
    int quantized = (int)lrintf(value * 127.0f);
    if (quantized > 127) {
        return 127;
    }
    if (quantized < -128) {
        return -128;
    }
    return quantized;
}

static void pack_quantized_vector(const float *src, uint64_t packed[2]) {
    packed[0] = 0;
    packed[1] = 0;
    for (int i = 0; i < kVectorSize; ++i) {
        const uint8_t byte = (uint8_t)(clip_int8(src[i]) & 0xFF);
        if (i < 8) {
            packed[0] |= ((uint64_t)byte) << (i * 8);
        } else {
            packed[1] |= ((uint64_t)byte) << ((i - 8) * 8);
        }
    }
}

static int fpga_rva_write128(uint32_t addr, const uint64_t data[2]) {
    uint32_t rva_msg[LOOP_RVA_IN];
    rva_format(true, addr, data, rva_msg);
    return ocl_rva_wr32((int)pci_bar_handle, rva_msg);
}

static int fpga_rva_verify128(uint32_t addr, uint64_t expected[2]) {
    uint32_t rva_msg[LOOP_RVA_IN];
    rva_format(false, addr, expected, rva_msg);
    return ocl_rva_r32((int)pci_bar_handle, expected, rva_msg);
}

static int fpga_protocol_self_test(void) {
    uint64_t cfg_words[2] = {0, 0};
    cfg_words[0] = build_pe_config_word(PE_PRECISION_INT8, 8);
    if (fpga_rva_write128(PE_CONFIG_ADDR, cfg_words) != 0) {
        fprintf(stderr, "Failed to write PE config during FPGA bridge self-test\n");
        return -1;
    }
    if (fpga_rva_verify128(PE_CONFIG_ADDR, cfg_words) != 0) {
        fprintf(stderr,
                "Loaded FPGA image does not match this repo's design_top register protocol.\n");
        return -1;
    }
    return 0;
}

int ocl_wr32(int bar_handle, uint16_t addr, uint32_t data) {
    if (fpga_pci_poke((pci_bar_handle_t)bar_handle, addr, data)) {
        fprintf(stderr, "OCL write failed at addr=0x%04x\n", addr);
        return 1;
    }
    return 0;
}

int ocl_rd32(int bar_handle, uint16_t addr, uint32_t *data) {
    if (fpga_pci_peek((pci_bar_handle_t)bar_handle, addr, data)) {
        fprintf(stderr, "OCL read failed at addr=0x%04x\n", addr);
        return 1;
    }
    return 0;
}

void rva_format(bool rw, uint32_t addr, const uint64_t data[2], uint32_t rva_msg[LOOP_RVA_IN]) {
    for (int i = 0; i < LOOP_RVA_IN; ++i) {
        rva_msg[i] = 0;
    }

    rva_msg[0] = (uint32_t)(data[0] & 0xFFFFFFFFu);
    rva_msg[1] = (uint32_t)((data[0] >> 32) & 0xFFFFFFFFu);
    rva_msg[2] = (uint32_t)(data[1] & 0xFFFFFFFFu);
    rva_msg[3] = (uint32_t)((data[1] >> 32) & 0xFFFFFFFFu);
    rva_msg[4] = addr & 0xFFFFFFu;
    if (rw) {
        rva_msg[5] |= (1u << 8);
    }
    rva_msg[5] |= (1u << 31);
}

int ocl_rva_wr32(int bar_handle, const uint32_t rva_msg[LOOP_RVA_IN]) {
    for (int i = 0; i < LOOP_RVA_IN; ++i) {
        const uint16_t addr = ADDR_RVA_IN_START + (uint16_t)(i * 4);
        if (ocl_wr32(bar_handle, addr, rva_msg[i])) {
            return 1;
        }
    }
    return 0;
}

int ocl_rva_r32(int bar_handle, uint64_t data_cmp[2], const uint32_t rva_in[LOOP_RVA_IN]) {
    uint32_t rva_out_words[LOOP_RVA_OUT] = {0};
    uint64_t data_read[2] = {0, 0};

    if (ocl_rva_wr32(bar_handle, rva_in)) {
        return 1;
    }
    usleep(100);

    for (int i = 0; i < LOOP_RVA_OUT; ++i) {
        const uint16_t addr = ADDR_RVA_OUT_START + (uint16_t)(i * 4);
        if (ocl_rd32(bar_handle, addr, &rva_out_words[i])) {
            return 1;
        }
    }

    data_read[0] = ((uint64_t)rva_out_words[1] << 32) | rva_out_words[0];
    data_read[1] = ((uint64_t)rva_out_words[3] << 32) | rva_out_words[2];

    if (data_read[0] != data_cmp[0] || data_read[1] != data_cmp[1]) {
        fprintf(stderr,
                "RVA readback mismatch: expected 0x%016llx%016llx got 0x%016llx%016llx\n",
                (unsigned long long)data_cmp[1],
                (unsigned long long)data_cmp[0],
                (unsigned long long)data_read[1],
                (unsigned long long)data_read[0]);
        return 1;
    }
    return 0;
}

uint64_t build_pe_config_word(uint8_t precision_mode, int group_size) {
    uint64_t word = 0;
    word |= ((uint64_t)1 << PE_CONFIG_IS_VALID_BIT);
    word |= ((uint64_t)1 << PE_CONFIG_IS_BIAS_BIT);
    word |= ((uint64_t)1 << PE_CONFIG_NUM_MANAGER_BIT);
    word |= ((uint64_t)1 << PE_CONFIG_NUM_OUTPUT_BIT);
    word |= ((uint64_t)(precision_mode & 0x3u) << PE_CONFIG_PRECISION_MODE_BIT);
    if (group_size == 16) {
        word |= ((uint64_t)1 << PE_CONFIG_GROUP_SIZE_IS_16_BIT);
    }
    return word;
}

int fpga_init(int slot_id) {
    if (fpga_initialized) {
        return 0;
    }

    fpga_slot_id = slot_id;
    if (fpga_mgmt_init() != 0) {
        fprintf(stderr, "Failed to initialize fpga_mgmt\n");
        return -1;
    }

    if (check_slot_config(fpga_slot_id) != 0) {
        return -1;
    }

    if (fpga_pci_attach(fpga_slot_id, FPGA_APP_PF, APP_PF_BAR0, 0, &pci_bar_handle) != 0) {
        fprintf(stderr, "Failed to attach to FPGA slot %d\n", fpga_slot_id);
        return -1;
    }

    if (fpga_protocol_self_test() != 0) {
        fpga_pci_detach(pci_bar_handle);
        pci_bar_handle = PCI_BAR_HANDLE_INIT;
        return -1;
    }

    fpga_initialized = true;
    return 0;
}

long fpga_get_bar_handle(void) {
    if (!fpga_initialized) {
        return -1;
    }
    return (long)pci_bar_handle;
}

int fpga_write32(uint32_t addr, uint32_t data) {
    if (!fpga_initialized) {
        return -1;
    }
    return ocl_wr32((int)pci_bar_handle, (uint16_t)addr, data);
}

int fpga_read32(uint32_t addr, uint32_t *data) {
    if (!fpga_initialized) {
        return -1;
    }
    return ocl_rd32((int)pci_bar_handle, (uint16_t)addr, data);
}

int fpga_matmul_16x16(float *A, float *B, float *C) {
    uint64_t payload[2] = {0, 0};
    uint32_t activation = 0;
    int rc = 0;

    if (!fpga_initialized) {
        fprintf(stderr, "FPGA not initialized\n");
        return -1;
    }

    payload[0] = build_pe_config_word(PE_PRECISION_INT8, 8);
    payload[1] = 0;
    if (fpga_rva_write128(PE_CONFIG_ADDR, payload) != 0) {
        return -1;
    }

    payload[0] = 0x100u;
    payload[1] = 0;
    if (fpga_rva_write128(MANAGER_CONFIG_ADDR, payload) != 0) {
        return -1;
    }

    for (int lane = 0; lane < kNumVectorLanes; ++lane) {
        pack_quantized_vector(A + lane * kVectorSize, payload);
        if (fpga_rva_write128(WEIGHT_BASE_ADDR + (uint32_t)(lane << 4), payload) != 0) {
            return -1;
        }
    }

    for (int col = 0; col < kVectorSize; ++col) {
        float column[kVectorSize];
        for (int row = 0; row < kVectorSize; ++row) {
            column[row] = B[row * kVectorSize + col];
        }

        pack_quantized_vector(column, payload);
        if (fpga_rva_write128(INPUT_BASE_ADDR, payload) != 0) {
            return -1;
        }

        rc = fpga_write32(ADDR_START_CFG, 1);
        rc |= fpga_write32(ADDR_START_CFG, 0);
        if (rc != 0) {
            return -1;
        }
        usleep(100);

        for (int row = 0; row < kNumVectorLanes; ++row) {
            if (fpga_read32(ADDR_ACT_PORT_START + (uint32_t)(row * 4), &activation) != 0) {
                return -1;
            }
            C[row * kVectorSize + col] =
                ((float)((int32_t)activation) * SCALE_DIVISOR) / (127.0f * 127.0f);
        }
    }

    return 0;
}

void fpga_cleanup(void) {
    if (fpga_initialized) {
        fpga_pci_detach(pci_bar_handle);
    }
    pci_bar_handle = PCI_BAR_HANDLE_INIT;
    fpga_initialized = false;
}

int fpga_is_initialized(void) {
    return fpga_initialized ? 1 : 0;
}
