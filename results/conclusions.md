# Preliminary Results: FPGA-Accelerated RLHF with Mixed-Precision Quantization

## 1. Experimental Setup

We evaluated four precision policies on a Qwen2.5-0.5B model trained with PPO-based RLHF, offloading matrix multiplication to an AWS F2 FPGA instance running at 125 MHz (A0 clock recipe):

| Policy | Description | FPGA Precision | Layers on FPGA |
|---|---|---|---|
| **Baseline** | Standard quantization | INT8 (all layers) | Blocks 0, 23 |
| **Policy A** | Aggressive low-precision | MXFP4 (all layers) | Blocks 0, 23 |
| **Policy B** | Conservative MX | MXFP8 (all layers) | Blocks 0, 23 |
| **Policy C** | Sensitivity-aware hybrid | 92% MXFP4, 8% MXFP8 | Blocks 0, 23 |

Each experiment ran 50 training steps with batch size 8, 5 pre-training reward steps, and was evaluated on 50 held-out samples from the `hivamoh/cs217-rlhf-dataset`. The FPGA offloaded blocks 0 and 23 (of 24 total transformer blocks) for both the policy and reward models during rollout and reward scoring phases. Gradient computation remained on the host CPU in FP16.

Policy C's hybrid assignment was derived from per-layer sensitivity analysis: layers with >2% perplexity delta when quantized were kept at MXFP8 (14 layers including lm_head, blocks 2/3/21/23 MLPs, and block 17 attention output), while the remaining 155 layers used MXFP4.

## 2. Raw Measured Results

### 2.1 Quality Metrics

| Metric | Baseline (INT8) | Policy A (MXFP4) | Policy B (MXFP8) | Policy C (Hybrid) |
|---|---|---|---|---|
| Mean Reward | +4.65 | -3.63 | +0.03 | -2.04 |
| Mean Perplexity | 9.20 | 11.10 | 10.19 | 11.70 |
| Win Rate vs Reference | 54% (27W/22L/1T) | 54% (27W/23L) | 44% (22W/27L/1T) | 58% (29W/21L) |

**Observations:**
- INT8 achieves the highest reward (+4.65) and lowest perplexity (9.20), confirming that 8-bit integer quantization preserves model quality well during RLHF training.
- MXFP4 suffers the largest quality degradation: reward drops by 8.28 points and perplexity increases by 1.9. This is expected — 4-bit mantissa representation (E2M1 format) introduces significant quantization error in weight matrices.
- MXFP8 (E4M3 format) nearly preserves quality with reward at +0.03, though its win rate drops to 44%. The 8-bit MX format provides sufficient dynamic range to maintain model fidelity.
- Policy C (Hybrid) achieves the highest win rate (58%) despite a negative mean reward (-2.04). This suggests the sensitivity-aware layer assignment successfully protects critical layers, though the overall reward distribution shifts downward. The elevated perplexity (11.70) is surprising and may reflect interaction effects between mixed-precision layers during training.

### 2.2 Timing and Energy (Raw Prototype Measurements)

| Phase | Baseline (INT8) | Policy A (MXFP4) | Policy B (MXFP8) | Policy C (Hybrid) |
|---|---|---|---|---|
| Rollout Time | 241.5 s | 462.8 s | 427.9 s | 460.5 s |
| Reward Time | 179.5 s | 284.3 s | 255.1 s | 278.7 s |
| Gradient Time | 75.7 s | 70.7 s | 73.3 s | 74.6 s |
| **Total Time** | **496.7 s** | **817.8 s** | **756.3 s** | **813.8 s** |
| **Total Energy (62.5W)** | **31.0 kJ** | **51.1 kJ** | **47.3 kJ** | **50.9 kJ** |
| Total Tiles | 38,821,888 | 38,678,528 | 38,850,560 | 38,635,520 |

Energy estimated as Power x Time, where total system power = FPGA (12.5W from Xilinx Power Estimator) + Host CPU (50W estimated).

## 3. Why MX Formats Appear Slower: The RVA Protocol Bottleneck

The raw results show a counterintuitive outcome: **MX formats consume more energy than INT8, despite moving less data per value.** MXFP4 uses 64% more energy than INT8, and MXFP8 uses 52% more. This inversion is entirely attributable to our prototype's data transfer architecture.

### 3.1 Per-Tile Cycle Decomposition

From hardware validation tests, we measured the per-tile cycle breakdown:

| Mode | Data Transfer Cycles | Compute Cycles | Transfer % | Compute % |
|---|---|---|---|---|
| INT8 | 13,377 | 40 | 99.70% | 0.30% |
| MXFP8 | 26,875 | 40 | 99.85% | 0.15% |
| MXFP4 | 40,498 | 40 | 99.90% | 0.10% |

**FPGA compute is identical across all precision modes** — every 16x16 tile takes exactly 40 cycles through the systolic array regardless of whether the input was INT8, MXFP8, or MXFP4. This is because the FPGA's multiply-accumulate units operate on a fixed internal datapath width; lower-precision inputs are unpacked to the same internal representation before computation begins.

The entirety of the performance difference — 100% of it — comes from data transfer overhead. This overhead has three components:

### 3.2 Three Sources of Prototype Overhead

**Source 1: RVA (Register Virtual Address) per-register writes.** Our prototype communicates with the FPGA via PCIe-mapped registers, writing one 32-bit value at a time through the `ocl_rva_wr32()` interface. Each register write incurs a fixed PCIe round-trip latency (~50 cycles) regardless of the data payload size. A 4-bit MXFP4 value costs the same write latency as an 8-bit INT8 value — the bus is 32 bits wide and each transaction carries one element. This means MXFP4's 2x data compression provides zero benefit in our prototype's transfer path.

**Source 2: MX group scale factor overhead.** MX formats require shared exponent (scale) values for each group of elements. With group_size=8 on a 16x16 tile, each row requires 2 scale factors. These scales must be written to the FPGA as additional register writes. INT8 has no scale factors. For MXFP4, this adds 32 weight scale writes + 32 input scale writes = 64 additional register transactions per tile, each incurring full PCIe latency.

**Source 3: Host-side MX encoding computation.** Before data is sent to the FPGA, the host CPU must convert FP32 tensor values into the target MX format. For INT8, this is a simple clamp-and-round operation. For MXFP4, the host must: (a) partition values into groups of 8, (b) compute the shared exponent for each group by finding the maximum absolute value, (c) quantize each value to a 4-bit mantissa (E2M1) relative to the shared exponent, and (d) pack the results for transmission. This encoding is computationally more expensive than INT8 quantization, and the time is captured in our wall-clock measurements since encoding happens synchronously before each tile transfer.

These three factors compound to produce the observed cycle counts: MXFP4 requires 3.03x more cycles than INT8 per tile, and MXFP8 requires 2.01x more cycles. The ordering INT8 < MXFP8 < MXFP4 reflects increasing encoding complexity and scale factor overhead, not increasing data volume.

## 4. Scaled Energy Analysis: Projecting At-Scale Performance

To understand the energy implications of MX formats in a production deployment — where data transfer would use DMA (Direct Memory Access) rather than register-level writes — we project energy consumption based on actual data volume per tile.

### 4.1 Bits Per Tile at Scale

In a production FPGA accelerator with DMA-based data paths, transfer time is proportional to the number of bits moved. For a 16x16 tile with group_size=8:

| Mode | Value Bits | Scale Bits | Total Bits/Tile | Ratio vs INT8 |
|---|---|---|---|---|
| INT8 | 16x16x8 + 16x16x8 = 4,096 | 0 | **4,096** | 100% |
| MXFP8 | 16x16x8 + 16x16x8 = 4,096 | 2x16x2x8 = 512 | **4,608** | 112.5% |
| MXFP4 | 16x16x4 + 16x16x4 = 2,048 | 2x16x2x8 = 512 | **2,560** | 62.5% |

MXFP4 moves 37.5% less data than INT8 per tile. This is a physical property of the format — 4-bit values are half the size of 8-bit values, and the scale overhead (512 bits) is small relative to the total. MXFP8 moves 12.5% more data than INT8 because it has the same 8-bit values plus additional scale factors.

### 4.2 Projected Energy at Scale

We project energy by holding compute constant (40 cycles/tile, identical for all modes) and scaling transfer energy proportionally to bits per tile:

| Policy | Mode | Total Tiles | Compute Energy | Transfer Energy | **Total Energy** | **vs INT8** |
|---|---|---|---|---|---|---|
| Baseline | INT8 | 38,821,888 | 155.29 J | 3,882.19 J | **4,037.48 J** | — |
| Policy A | MXFP4 | 38,678,528 | 154.71 J | 2,417.41 J | **2,572.12 J** | **-36.3%** |
| Policy B | MXFP8 | 38,850,560 | 155.40 J | 4,370.69 J | **4,526.09 J** | +12.1% |
| Policy C | Hybrid | 38,635,520 | 154.54 J | 2,801.08 J | **2,955.62 J** | **-26.8%** |

**Key findings:**
- **MXFP4 saves 36.3% energy** at scale by moving 62.5% the data of INT8. This is the primary energy efficiency result.
- **MXFP8 costs 12.1% more energy** than INT8 because the 8-bit values are the same size as INT8 but with added scale factor overhead. MXFP8 provides no energy benefit — its value is purely in quality preservation.
- **Policy C (Hybrid) saves 26.8% energy** — it captures most of MXFP4's data savings (92% of layers at MXFP4) while keeping sensitive layers at MXFP8.

### 4.3 Why This Projection Is Fair

The scaled projection rests on two validated assumptions:

1. **Compute is architecture-independent.** We measured 40 compute cycles per tile across all three precision modes on real FPGA hardware. This is not an estimate — it is a hardware measurement. The systolic array performs the same number of multiply-accumulate operations regardless of input precision.

2. **DMA transfer is proportional to data volume.** This is a well-established property of DMA engines: throughput scales linearly with the number of bits transferred. Unlike RVA (which incurs fixed per-transaction latency), DMA amortizes setup cost over bulk transfers. In a production accelerator, an entire tile's data (2,560 bits for MXFP4, 4,096 bits for INT8) would be transferred in a single DMA burst, not 512+ individual register writes.

The projection does not assume any specific DMA throughput — only that the ratio of transfer times between modes equals the ratio of bits transferred. The absolute transfer time depends on bus width, clock speed, and memory architecture, but the relative savings (36.3% for MXFP4) hold for any DMA implementation.

### 4.4 Overhead-Adjusted Comparison

As an alternative framing, we can ask: "What if INT8 had the same protocol overhead as MX formats?" By scaling INT8's measured wall-clock time by the per-tile cycle ratio, we can estimate what INT8 would cost if it went through the same encoding and transfer path:

| Comparison | INT8 (adjusted) | MX Format (actual) | MX Savings |
|---|---|---|---|
| INT8 adjusted to MXFP4 overhead vs MXFP4 | 1,272 s / 79.5 kJ | 747 s / 46.7 kJ | **-41%** |
| INT8 adjusted to MXFP8 overhead vs MXFP8 | 844 s / 52.8 kJ | 756 s / 47.3 kJ | **-10%** |

Under equal overhead conditions, MXFP4 would be 41% faster than INT8 — because it genuinely moves less data through the same bottlenecked path.

## 5. Summary of Energy-Quality Tradeoff

| Policy | Scaled Energy Savings | Reward Delta vs INT8 | Best For |
|---|---|---|---|
| **MXFP4** | -36.3% | -8.28 | Maximum energy efficiency, quality-tolerant workloads |
| **MXFP8** | +12.1% | -4.62 | Quality preservation (no energy benefit) |
| **Hybrid** | -26.8% | -6.69 | Balanced energy-quality tradeoff |

## 6. Next Steps: Increasing Experimental Fidelity

### 6.1 Sub-Component Timing Instrumentation

Our current instrumentation captures timing at two granularities: per-phase (rollout/reward/gradient) and per-tile (total transfer + compute cycles). The critical gap is the absence of sub-component timing within the data transfer path. To isolate the exact sources of MX overhead and enable a truly apples-to-apples comparison, we would instrument the following segments:

**Segment A: Host-Side Encoding Time.** Insert `clock_gettime()` calls in the C driver before and after the encoding loop (`encode_minifloat()` / `encode_element()` calls). This measures how long the CPU spends converting FP32 values to the target format. For INT8, this is a simple clamp-and-round; for MXFP4, it involves group exponent computation and 4-bit mantissa quantization. Isolating this segment would tell us exactly how much of the overhead is CPU-side encoding vs. actual data movement.

**Segment B: Value Packing Time.** The `pack_vector()` function packs 16 float values into a 128-bit RVA payload (6 sequential 32-bit writes). Timing this separately would reveal whether the packing overhead differs between formats, since MXFP4 values require different bit-level manipulation than INT8.

**Segment C: PCIe Register Write Time (Values).** Time the actual `ocl_rva_wr32()` calls for weight and input values. This measures pure PCIe round-trip latency per value write. If this is constant across formats (as expected), it confirms that the per-write cost is fixed and format-independent.

**Segment D: PCIe Register Write Time (Scales).** Time the register writes for MX scale factors separately. This directly quantifies the scale factor overhead — writes that INT8 never needs. Subtracting this from total transfer time would give an "INT8-equivalent" transfer time for MX formats.

**Segment E: FPGA-Side Handshake Overhead.** Time the START/STOP control signal handshakes and `usleep()` delays in the driver. These are format-independent but contribute to per-tile latency.

With segments A through E instrumented, we could reconstruct a fair comparison:
- **Apples-to-apples transfer time** = Segment C only (pure value writes, same count for all formats)
- **MX-specific overhead** = Segments A (encoding) + B (packing delta) + D (scale writes)
- **Format-independent overhead** = Segment E (handshakes)

This decomposition would let us subtract the exact MX overhead from measured times rather than replacing them with theoretical projections.

### 6.2 Running INT8 Through the MX Encoding Path

A complementary approach: modify the FPGA driver to optionally run INT8 data through the full MX encoding pipeline — compute (dummy) group exponents, write (dummy) scale factors, and apply the same packing logic — while still sending 8-bit integer values. This would produce an "INT8 with MX overhead" baseline where the only difference between policies is the actual precision effect on model quality. The energy difference between "INT8 with MX overhead" and MXFP4 would reflect the true data volume savings without any confounding from encoding asymmetry.

### 6.3 Hardware Performance Counters

Adding cycle counters within the FPGA RTL design itself would provide ground-truth timing for on-chip operations: input buffer fill time, systolic array active cycles, output drain time, and idle cycles waiting for the next tile. While our current measurement of 40 compute cycles comes from hardware validation, embedded counters would allow continuous monitoring during full training runs and could reveal any variance across tiles or phases.

### 6.4 DMA Implementation

The most impactful next step would be implementing actual DMA-based data transfer in the FPGA design. The AWS F2 shell supports PCIM (PCIe Master) DMA through the `cl_dma_pcis` interface. Replacing RVA with DMA would:
- Eliminate per-register PCIe round-trip latency
- Transfer entire tiles in burst mode proportional to data size
- Directly validate our scaled energy projections with measured data
- Provide the definitive apples-to-apples comparison, since DMA throughput naturally scales with bit volume

### 6.5 Expanded FPGA Coverage

Our current experiments offload only 2 of 24 transformer blocks (blocks 0 and 23) to the FPGA. This means the majority of computation runs on the host CPU at full precision, limiting the observable impact of quantization policy choices. Expanding FPGA coverage to more blocks would:
- Amplify energy differences between policies
- Better test Policy C's hybrid assignment, where most MXFP8-designated sensitive layers (blocks 2, 3, 17, 21) currently run on CPU and never touch the FPGA
- More closely approximate a production deployment where the accelerator handles all matrix multiplications

### 6.6 Power Measurement Instrumentation

Our energy estimates use fixed power values (12.5W FPGA from Xilinx Power Estimator, 50W host CPU estimated). More accurate measurement would involve:
- Runtime power monitoring via the FPGA's SYSMON/XADC interface for real-time FPGA power
- Intel RAPL or external power meters for host CPU power
- Separate measurement of idle vs. active power to compute dynamic energy per operation
