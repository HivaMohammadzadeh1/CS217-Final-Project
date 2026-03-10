/*
 * All rights reserved - Stanford University. 
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef __DATAPATH__
#define __DATAPATH__

#include <nvhls_int.h>
#include <nvhls_types.h>
#include "Spec.h"

// ---- MX Minifloat Fixed-Point Decode (4 fractional bits) ----

// E4M3 (MXFP8): 1 sign + 4 exponent + 3 mantissa, bias=7
// Returns signed 16-bit fixed-point with 4 fractional bits.
// Max magnitude: 15 << 8 = 3840 (for exp=14, mant=7), fits in NVINT13.
inline NVINTW(16) DecodeE4M3Fixed(NVUINTW(8) code) {
  NVUINTW(1) sign_bit = nvhls::get_slc<1>(code, 7);
  NVUINTW(4) exp_f    = nvhls::get_slc<4>(code, 3);
  NVUINTW(3) mant     = nvhls::get_slc<3>(code, 0);

  NVUINTW(16) abs_val = 0;
  NVUINTW(16) full_mant = 8 | (NVUINTW(16))mant;

  if (exp_f == 0) {
    abs_val = 0;
  } else if (exp_f == 15 && mant == 7) {
    abs_val = 0;
  } else if (exp_f >= 6) {
    NVUINTW(4) shift_l = exp_f - 6;
    abs_val = full_mant << shift_l;
  } else {
    NVUINTW(4) shift_r = 6 - exp_f;
    abs_val = full_mant >> shift_r;
  }

  NVINTW(16) result = (NVINTW(16))abs_val;
  if (sign_bit) result = -result;
  return result;
}

// E2M1 (MXFP4): 1 sign + 2 exponent + 1 mantissa, bias=1
// Lower 4 bits of the byte are used; upper 4 are ignored.
// Max magnitude: 3 << 5 = 96 (for exp=3, mant=1), fits easily.
inline NVINTW(16) DecodeE2M1Fixed(NVUINTW(8) code) {
  NVUINTW(1) sign_bit = nvhls::get_slc<1>(code, 3);
  NVUINTW(2) exp_f    = nvhls::get_slc<2>(code, 1);
  NVUINTW(1) mant     = nvhls::get_slc<1>(code, 0);

  NVUINTW(16) abs_val = 0;

  if (exp_f == 0) {
    abs_val = mant ? (NVUINTW(16))8 : (NVUINTW(16))0;
  } else {
    NVUINTW(16) full_mant = 2 | (NVUINTW(16))mant;
    NVUINTW(3) shift_l = (NVUINTW(3))exp_f + 2;
    abs_val = full_mant << shift_l;
  }

  NVINTW(16) result = (NVINTW(16))abs_val;
  if (sign_bit) result = -result;
  return result;
}

inline NVINTW(16) DecodeMXByte(NVUINTW(8) code, NVUINTW(2) precision_mode) {
  if (precision_mode == spec::kPrecisionMXFP8) {
    return DecodeE4M3Fixed(code);
  } else {
    return DecodeE2M1Fixed(code);
  }
}

// ---- Product-Sum Functions ----

// INT8 baseline: unsigned byte multiply-accumulate
inline void ProductSum(const spec::VectorType in_1, const spec::VectorType in_2, spec::AccumScalarType& out) {
  spec::AccumScalarType out_tmp = 0;
#pragma hls_unroll yes
  for (int i = 0; i < spec::kVectorSize; i++) {
    out_tmp += static_cast<spec::AccumScalarType>(in_1[i]) * static_cast<spec::AccumScalarType>(in_2[i]);
  }
  out = out_tmp;
}

// MX product-sum: decode each byte as minifloat, multiply in fixed-point,
// accumulate, then remove fractional bits (4+4=8 from the two operands).
inline void ProductSumMX(const spec::VectorType in_1, const spec::VectorType in_2,
                         const NVUINTW(2) precision_mode,
                         spec::AccumScalarType& out) {
  NVINTW(32) acc = 0;
#pragma hls_unroll yes
  for (int i = 0; i < spec::kVectorSize; i++) {
    NVINTW(16) w_val = DecodeMXByte(in_1[i], precision_mode);
    NVINTW(16) i_val = DecodeMXByte(in_2[i], precision_mode);
    acc += (NVINTW(32))w_val * (NVINTW(32))i_val;
  }
  out = (spec::AccumScalarType)(acc >> 8);
}

// ---- Top-Level Datapath ----

inline void Datapath(spec::VectorType weight_in[spec::kNumVectorLanes],
              spec::VectorType input_in,
              const NVUINTW(2) precision_mode,
              const NVUINTW(1) mx_group_size_is_16,
              spec::AccumVectorType& accum_out)
{
  spec::AccumVectorType accum_out_tmp;
  (void)mx_group_size_is_16;

#pragma hls_unroll yes
  for (int i = 0; i < spec::kNumVectorLanes; i++) {
    if (precision_mode == spec::kPrecisionINT8) {
      ProductSum(weight_in[i], input_in, accum_out_tmp[i]);
    } else {
      ProductSumMX(weight_in[i], input_in, precision_mode, accum_out_tmp[i]);
    }
  }
  accum_out = accum_out_tmp;
}


#endif
