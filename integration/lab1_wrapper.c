/**
 * Lab 1 FPGA Python Wrapper (MX-aware)
 *
 * This C wrapper initializes the FPGA and provides functions callable from
 * Python via ctypes.  It supports INT8, MXFP8 and MXFP4 precision modes
 * using the same RVA protocol that design_top.c uses.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fpga_pci.h>
#include <fpga_mgmt.h>

/* ------------------------------------------------------------------ */
/* FPGA slot state                                                     */
/* ------------------------------------------------------------------ */
static int fpga_slot_id = 0;
static pci_bar_handle_t pci_bar_handle = PCI_BAR_HANDLE_INIT;
static bool fpga_initialized = false;

/* Current precision state (mirrors Python-side state). */
static uint8_t current_precision_mode = 0; /* PE_PRECISION_INT8 */
static int     current_group_size     = 8;

/* ------------------------------------------------------------------ */
/* Hardware constants (must match design_top_defines.vh)                */
/* ------------------------------------------------------------------ */
#define kIntWordWidth     8
#define kVectorSize      16
#define kNumVectorLanes  16
#define kActWordWidth    32

#define kActWordMax ((int32_t)(((int64_t)1 << (kActWordWidth - 1)) - 1))
#define kActWordMin (-kActWordMax)

#define WIDTH_DATA_AXI   32
#define WIDTH_DATA_RVA_IN (kIntWordWidth * kVectorSize) /* 128 */
#define WIDTH_ADDR_RVA_IN 24
#define WIDTH_RVA_IN_32  192
#define LOOP_RVA_IN      (WIDTH_RVA_IN_32 / 32)  /* 6 */

#define WIDTH_RVA_OUT    WIDTH_DATA_RVA_IN /* 128 */
#define LOOP_RVA_OUT     (WIDTH_RVA_OUT / WIDTH_DATA_AXI) /* 4 */

#define LOOP_ACT_PORT    (kActWordWidth * kNumVectorLanes / WIDTH_DATA_AXI) /* 16 */

/* Register / address map */
#define ADDR_RVA_IN_START    0x0408
#define ADDR_RVA_OUT_START   0x0408
#define ADDR_ACT_PORT_START  0x0440
#define ADDR_START_CFG       0x0404
#define ADDR_TRANSFER_COUNTER_EN 0x0400

/* RVA region addresses */
#define ADDR_PE_CONFIG       0x400010
#define ADDR_MANAGER_CONFIG  0x400020
#define ADDR_WEIGHT_BASE     0x500000
#define ADDR_INPUT_BASE      0x600000

/* PEConfig bit positions */
#define PE_CONFIG_IS_VALID_BIT          0
#define PE_CONFIG_IS_BIAS_BIT          24
#define PE_CONFIG_NUM_MANAGER_BIT      32
#define PE_CONFIG_NUM_OUTPUT_BIT       40
#define PE_CONFIG_PRECISION_MODE_BIT   48
#define PE_CONFIG_GROUP_SIZE_IS_16_BIT 56

/* Precision mode constants */
#define PE_PRECISION_INT8  0
#define PE_PRECISION_MXFP8 1
#define PE_PRECISION_MXFP4 2

/* MX minifloat format parameters */
#define MXFP8_EXP_BITS  4
#define MXFP8_MANT_BITS 3
#define MXFP8_EXP_BIAS  7

#define MXFP4_EXP_BITS  2
#define MXFP4_MANT_BITS 1
#define MXFP4_EXP_BIAS  1

/* ------------------------------------------------------------------ */
/* Low-level AXI-lite helpers (same as design_top.c)                   */
/* ------------------------------------------------------------------ */
static int ocl_wr32(uint16_t addr, uint32_t data) {
    if (fpga_pci_poke(pci_bar_handle, addr, data)) {
        fprintf(stderr, "ERROR: MMIO write failed at addr=0x%04x\n", addr);
        return 1;
    }
    return 0;
}

static int ocl_rd32(uint16_t addr, uint32_t *data) {
    if (fpga_pci_peek(pci_bar_handle, addr, data)) {
        fprintf(stderr, "ERROR: MMIO read failed at addr=0x%04x\n", addr);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* RVA message helpers (same protocol as design_top.c)                 */
/* ------------------------------------------------------------------ */
static void rva_format(bool rw, uint32_t addr, const uint64_t data[2],
                       uint32_t rva_msg[LOOP_RVA_IN]) {
    for (int i = 0; i < LOOP_RVA_IN; i++) rva_msg[i] = 0;

    rva_msg[0] = (uint32_t)(data[0] & 0xFFFFFFFF);
    rva_msg[1] = (uint32_t)((data[0] >> 32) & 0xFFFFFFFF);
    rva_msg[2] = (uint32_t)(data[1] & 0xFFFFFFFF);
    rva_msg[3] = (uint32_t)((data[1] >> 32) & 0xFFFFFFFF);
    rva_msg[4] = addr & 0xFFFFFF;
    if (rw) rva_msg[5] |= (1 << 8);
    rva_msg[5] |= (1U << 31);
}

static int ocl_rva_wr32(const uint32_t rva_msg[LOOP_RVA_IN]) {
    for (int i = 0; i < LOOP_RVA_IN; i++) {
        uint16_t addr = ADDR_RVA_IN_START + i * 4;
        if (ocl_wr32(addr, rva_msg[i])) return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* PE config word builder (same as design_top.c)                       */
/* ------------------------------------------------------------------ */
static uint64_t build_pe_config_word(uint8_t precision_mode, int group_size) {
    uint64_t word = 0;
    word |= ((uint64_t)1 << PE_CONFIG_IS_VALID_BIT);
    word |= ((uint64_t)1 << PE_CONFIG_IS_BIAS_BIT);
    word |= ((uint64_t)1 << PE_CONFIG_NUM_MANAGER_BIT);
    word |= ((uint64_t)1 << PE_CONFIG_NUM_OUTPUT_BIT);
    word |= ((uint64_t)(precision_mode & 0x3u) << PE_CONFIG_PRECISION_MODE_BIT);
    if (group_size == 16)
        word |= ((uint64_t)1 << PE_CONFIG_GROUP_SIZE_IS_16_BIT);
    return word;
}

/* ------------------------------------------------------------------ */
/* MX minifloat encode / decode (same as design_top.c)                 */
/* ------------------------------------------------------------------ */
static int floor_log2_positive(float x) {
    if (x <= 0.0f) return 0;
    return (int)floor(log2(x));
}

static uint8_t encode_minifloat(float value, int exp_bits, int mant_bits, int exp_bias) {
    int sign_shift = exp_bits + mant_bits;
    int exp_mask  = (1 << exp_bits) - 1;
    int mant_mask = (1 << mant_bits) - 1;

    if (!isfinite(value) || value == 0.0f) return 0;

    int neg = signbit(value) ? 1 : 0;
    float ax = fabsf(value);

    int exponent = floor_log2_positive(ax);
    float normalized = ldexpf(ax, -exponent);
    int exp_field = exponent + exp_bias;
    int mantissa  = 0;

    if (exp_field <= 0) {
        float scaled = ldexpf(ax, -(1 - exp_bias));
        mantissa = (int)roundf(scaled * (float)(1 << mant_bits));
        if (mantissa < 0) mantissa = 0;
        if (mantissa > mant_mask) mantissa = mant_mask;
        exp_field = 0;
    } else {
        float mantissa_f = (normalized - 1.0f) * (float)(1 << mant_bits);
        mantissa = (int)roundf(mantissa_f);
        if (mantissa == (1 << mant_bits)) {
            mantissa = 0;
            exp_field += 1;
        }
        if (exp_field >= exp_mask) {
            exp_field = exp_mask;
            mantissa = mant_mask;
        }
    }

    uint8_t code = (uint8_t)((exp_field & exp_mask) << mant_bits);
    code |= (uint8_t)(mantissa & mant_mask);
    if (neg) code |= (uint8_t)(1u << sign_shift);
    return code;
}

static float decode_minifloat(uint8_t code, int exp_bits, int mant_bits, int exp_bias) {
    int sign_shift = exp_bits + mant_bits;
    int exp_mask  = (1 << exp_bits) - 1;
    int mant_mask = (1 << mant_bits) - 1;

    int neg       = ((code >> sign_shift) & 0x1u) != 0;
    int exp_field = (code >> mant_bits) & exp_mask;
    int mantissa  = code & mant_mask;

    if (exp_field == 0 && mantissa == 0) return 0.0f;

    float value;
    if (exp_field == 0) {
        float frac = (float)mantissa / (float)(1 << mant_bits);
        value = ldexpf(frac, 1 - exp_bias);
    } else {
        float frac = 1.0f + ((float)mantissa / (float)(1 << mant_bits));
        value = ldexpf(frac, exp_field - exp_bias);
    }
    return neg ? -value : value;
}

/* ------------------------------------------------------------------ */
/* Quantisation helpers                                                */
/* ------------------------------------------------------------------ */

/* Encode a float to a single byte in the current precision mode. */
static uint8_t encode_element(float value, uint8_t precision_mode) {
    if (precision_mode == PE_PRECISION_INT8) {
        /* Scale float to [-127, 127] range and clamp. */
        float scaled = value * 127.0f;
        if (scaled >  127.0f) scaled =  127.0f;
        if (scaled < -127.0f) scaled = -127.0f;
        return (uint8_t)(int8_t)roundf(scaled);
    } else if (precision_mode == PE_PRECISION_MXFP8) {
        return encode_minifloat(value, MXFP8_EXP_BITS, MXFP8_MANT_BITS, MXFP8_EXP_BIAS);
    } else { /* MXFP4 */
        return encode_minifloat(value, MXFP4_EXP_BITS, MXFP4_MANT_BITS, MXFP4_EXP_BIAS);
    }
}

/* Decode a single byte to a float in the current precision mode. */
static float decode_element(uint8_t code, uint8_t precision_mode) {
    if (precision_mode == PE_PRECISION_INT8) {
        return (float)(int8_t)code;
    } else if (precision_mode == PE_PRECISION_MXFP8) {
        return decode_minifloat(code, MXFP8_EXP_BITS, MXFP8_MANT_BITS, MXFP8_EXP_BIAS);
    } else { /* MXFP4 */
        return decode_minifloat(code, MXFP4_EXP_BITS, MXFP4_MANT_BITS, MXFP4_EXP_BIAS);
    }
}

/* Pack 16 floats into a 128-bit RVA data payload (two uint64_t). */
static void pack_vector(const float *values, uint64_t data[2],
                        uint8_t precision_mode) {
    data[0] = 0;
    data[1] = 0;
    for (int j = 0; j < 16; j++) {
        uint8_t encoded = encode_element(values[j], precision_mode);
        if (j < 8)
            data[0] |= (uint64_t)encoded << (j * 8);
        else
            data[1] |= (uint64_t)encoded << ((j - 8) * 8);
    }
}

/* ------------------------------------------------------------------ */
/* Dequantisation of output activations                                */
/* ------------------------------------------------------------------ */

/* For INT8 the FPGA accumulates (w_int8 * x_int8) and the result is an
 * INT32 activation.  We divide by 127^2 to undo the two scale factors
 * that were applied during quantisation of weights and inputs.
 *
 * For MX modes the FPGA internally decodes minifloats, multiplies as
 * fixed-point, and produces an INT32 activation that represents a
 * fixed-point value.  The design_top golden model simply casts to
 * float, so we return it as-is (the Python layer can apply any further
 * scaling if needed). */
static float dequantize_activation(int32_t raw, uint8_t precision_mode) {
    if (precision_mode == PE_PRECISION_INT8) {
        return (float)raw / (127.0f * 127.0f);
    }
    /* MX modes: return raw activation as float. */
    return (float)raw;
}

/* ================================================================== */
/* PUBLIC API (called from Python via ctypes)                          */
/* ================================================================== */

/**
 * Initialize FPGA connection.
 * Returns 0 on success, -1 on failure.
 */
int fpga_init(int slot_id) {
    int rc;

    if (fpga_initialized) {
        printf("FPGA already initialized\n");
        return 0;
    }

    fpga_slot_id = slot_id;

    rc = fpga_mgmt_init();
    if (rc != 0) {
        fprintf(stderr, "Failed to initialize FPGA management library: %d\n", rc);
        return -1;
    }

    rc = fpga_pci_attach(fpga_slot_id, FPGA_APP_PF, APP_PF_BAR0, 0,
                         &pci_bar_handle);
    if (rc != 0) {
        fprintf(stderr, "Failed to attach to FPGA slot %d: %d\n",
                fpga_slot_id, rc);
        return -1;
    }

    fpga_initialized = true;
    printf("FPGA initialized successfully (slot %d)\n", fpga_slot_id);
    return 0;
}

/**
 * Configure precision mode and group size.
 * Must be called before fpga_matmul_16x16_mx().
 * Returns 0 on success.
 */
int fpga_configure_precision(int precision_mode, int group_size) {
    if (precision_mode < 0 || precision_mode > 2) {
        fprintf(stderr, "Invalid precision mode %d\n", precision_mode);
        return -1;
    }
    if (group_size != 8 && group_size != 16) {
        fprintf(stderr, "Invalid group size %d\n", group_size);
        return -1;
    }
    current_precision_mode = (uint8_t)precision_mode;
    current_group_size     = group_size;
    return 0;
}

/**
 * Perform 16x16 matmul on real FPGA hardware with current precision mode.
 *
 * A: 16x16 weight matrix   (row-major float32, 256 elements)
 * B: 16x16 input matrix    (row-major float32, 256 elements)
 * C: 16x16 output matrix   (row-major float32, 256 elements)
 *
 * The function processes one column of B at a time (matching the
 * design_top test flow), which means 16 START/STOP sequences per call.
 *
 * Returns 0 on success, -1 on failure.
 */
int fpga_matmul_16x16_mx(float *A, float *B, float *C) {
    if (!fpga_initialized) {
        fprintf(stderr, "FPGA not initialized!\n");
        return -1;
    }

    uint32_t rva_msg[LOOP_RVA_IN];
    uint64_t rva_data[2];

    /* ---- 1) Write PEConfig with current precision/group_size ---- */
    rva_data[0] = build_pe_config_word(current_precision_mode,
                                        current_group_size);
    rva_data[1] = 0;
    rva_format(true, ADDR_PE_CONFIG, rva_data, rva_msg);
    if (ocl_rva_wr32(rva_msg)) {
        fprintf(stderr, "fpga_matmul: PEConfig write failed (mode=%d gs=%d)\n",
                current_precision_mode, current_group_size);
        return -1;
    }

    /* ---- 2) Write weight matrix (16 lanes × 128-bit vectors) ---- */
    for (int lane = 0; lane < 16; lane++) {
        pack_vector(&A[lane * 16], rva_data, current_precision_mode);
        rva_format(true, ADDR_WEIGHT_BASE + (lane << 4), rva_data, rva_msg);
        if (ocl_rva_wr32(rva_msg)) {
            fprintf(stderr, "fpga_matmul: weight write failed at lane %d\n", lane);
            return -1;
        }
    }

    /* ---- 3) Write Manager config ---- */
    rva_data[0] = 0x0000000000000100ULL;
    rva_data[1] = 0;
    rva_format(true, ADDR_MANAGER_CONFIG, rva_data, rva_msg);
    if (ocl_rva_wr32(rva_msg)) {
        fprintf(stderr, "fpga_matmul: manager config write failed\n");
        return -1;
    }

    /* ---- 4) Process each column of B ---- */
    for (int col = 0; col < 16; col++) {
        /* Extract column vector from row-major B. */
        float col_vec[16];
        for (int r = 0; r < 16; r++)
            col_vec[r] = B[r * 16 + col];

        /* Write input vector via RVA. */
        pack_vector(col_vec, rva_data, current_precision_mode);
        rva_format(true, ADDR_INPUT_BASE, rva_data, rva_msg);
        if (ocl_rva_wr32(rva_msg)) {
            fprintf(stderr, "fpga_matmul: input write failed at col %d\n", col);
            return -1;
        }

        /* START */
        if (ocl_wr32(ADDR_START_CFG, 0x1)) return -1;
        usleep(50);
        /* STOP */
        if (ocl_wr32(ADDR_START_CFG, 0x0)) return -1;
        usleep(50);

        /* Read 16 INT32 activations. */
        for (int i = 0; i < 16; i++) {
            uint32_t raw;
            if (ocl_rd32(ADDR_ACT_PORT_START + i * 4, &raw)) return -1;
            C[i * 16 + col] = dequantize_activation((int32_t)raw,
                                                     current_precision_mode);
        }
    }

    return 0;
}

/**
 * Legacy INT8-only matmul (calls the new MX-aware version with INT8).
 * Kept for backward compatibility so existing Python code still works.
 */
int fpga_matmul_16x16(float *A, float *B, float *C) {
    uint8_t saved_mode = current_precision_mode;
    int     saved_gs   = current_group_size;

    current_precision_mode = PE_PRECISION_INT8;
    current_group_size     = 8;

    int rc = fpga_matmul_16x16_mx(A, B, C);

    current_precision_mode = saved_mode;
    current_group_size     = saved_gs;
    return rc;
}

/* Low-level register access (kept for diagnostics / Python use). */
int fpga_write32(uint32_t addr, uint32_t data) {
    if (!fpga_initialized) { fprintf(stderr, "FPGA not initialized!\n"); return -1; }
    return fpga_pci_poke(pci_bar_handle, addr, data) ? -1 : 0;
}

int fpga_read32(uint32_t addr, uint32_t *data) {
    if (!fpga_initialized) { fprintf(stderr, "FPGA not initialized!\n"); return -1; }
    return fpga_pci_peek(pci_bar_handle, addr, data) ? -1 : 0;
}

void fpga_cleanup() {
    if (fpga_initialized) {
        fpga_pci_detach(pci_bar_handle);
        fpga_initialized = false;
        printf("FPGA resources released\n");
    }
}

int fpga_is_initialized() {
    return fpga_initialized ? 1 : 0;
}
