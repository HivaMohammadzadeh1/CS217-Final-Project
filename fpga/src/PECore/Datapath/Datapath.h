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

// Datapath that supports baseline INT8 MAC plus staged MXFP8/MXFP4
// quantize-dequantize compute over the existing 8-bit storage interface.
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <stdint.h>
#include "Spec.h"

inline float AbsFloat(float value) {
  return (value < 0.0f) ? -value : value;
}

inline float ScaleByPow2(float value, int exponent) {
  float scaled = value;
  if (exponent >= 0) {
    for (int i = 0; i < exponent; ++i) {
      scaled *= 2.0f;
    }
  } else {
    for (int i = 0; i < -exponent; ++i) {
      scaled *= 0.5f;
    }
  }
  return scaled;
}

inline long long RoundToLongLong(float value) {
  return static_cast<long long>((value >= 0.0f) ? (value + 0.5f) : (value - 0.5f));
}

inline int RawLaneToSignedInt(const spec::ScalarType& value) {
  const unsigned int raw = static_cast<unsigned int>(value);
  const unsigned int sign_bit = 1u << (spec::kIntWordWidth - 1);
  const unsigned int full_range = 1u << spec::kIntWordWidth;
  if ((raw & sign_bit) != 0u) {
    return static_cast<int>(raw) - static_cast<int>(full_range);
  }
  return static_cast<int>(raw);
}

inline int FloorLog2Positive(float x) {
  if (x <= 0.0f) {
    return 0;
  }
  int exponent = 0;
  float normalized = x;

  if (normalized >= 1.0f) {
    while (normalized >= 2.0f) {
      normalized *= 0.5f;
      exponent += 1;
    }
  } else {
    while (normalized < 1.0f) {
      normalized *= 2.0f;
      exponent -= 1;
    }
  }

  return exponent;
}

inline uint8_t EncodeMinifloat(float value, int exp_bits, int mant_bits, int exp_bias) {
  const int sign_shift = exp_bits + mant_bits;
  const int exp_mask = (1 << exp_bits) - 1;
  const int mant_mask = (1 << mant_bits) - 1;

  if (value == 0.0f) {
    return 0;
  }

  const bool neg = value < 0.0f;
  const float ax = AbsFloat(value);

  int exponent = FloorLog2Positive(ax);
  float normalized = ScaleByPow2(ax, -exponent);
  int exp_field = exponent + exp_bias;
  int mantissa = 0;

  if (exp_field <= 0) {
    const float scaled = ScaleByPow2(ax, -(1 - exp_bias));
    mantissa = static_cast<int>(RoundToLongLong(scaled * static_cast<float>(1 << mant_bits)));
    if (mantissa < 0) {
      mantissa = 0;
    }
    if (mantissa > mant_mask) {
      mantissa = mant_mask;
    }
    exp_field = 0;
  } else {
    const float mantissa_f = (normalized - 1.0f) * static_cast<float>(1 << mant_bits);
    mantissa = static_cast<int>(RoundToLongLong(mantissa_f));
    if (mantissa == (1 << mant_bits)) {
      mantissa = 0;
      exp_field += 1;
    }
    if (exp_field >= exp_mask) {
      exp_field = exp_mask;
      mantissa = mant_mask;
    }
  }

  uint8_t code = static_cast<uint8_t>((exp_field & exp_mask) << mant_bits);
  code |= static_cast<uint8_t>(mantissa & mant_mask);
  if (neg) {
    code |= static_cast<uint8_t>(1u << sign_shift);
  }
  return code;
}

inline float DecodeMinifloat(uint8_t code, int exp_bits, int mant_bits, int exp_bias) {
  const int sign_shift = exp_bits + mant_bits;
  const int exp_mask = (1 << exp_bits) - 1;
  const int mant_mask = (1 << mant_bits) - 1;

  const bool neg = ((code >> sign_shift) & 0x1u) != 0;
  const int exp_field = (code >> mant_bits) & exp_mask;
  const int mantissa = code & mant_mask;

  if (exp_field == 0 && mantissa == 0) {
    return 0.0f;
  }

  float value = 0.0f;
  if (exp_field == 0) {
    const float frac = static_cast<float>(mantissa) / static_cast<float>(1 << mant_bits);
    value = ScaleByPow2(frac, 1 - exp_bias);
  } else {
    const float frac = 1.0f + (static_cast<float>(mantissa) / static_cast<float>(1 << mant_bits));
    value = ScaleByPow2(frac, exp_field - exp_bias);
  }

  return neg ? -value : value;
}

inline void QuantizeDequantizeMXGroup(const spec::VectorType in,
                                      const int start_idx,
                                      const int group_size,
                                      const int exp_bits,
                                      const int mant_bits,
                                      const int exp_bias,
                                      float out[spec::kVectorSize]) {
  float max_abs = 0.0f;
  for (int i = start_idx; i < start_idx + group_size; ++i) {
    const float lane_value = static_cast<float>(RawLaneToSignedInt(in[i]));
    const float lane_abs = AbsFloat(lane_value);
    if (lane_abs > max_abs) {
      max_abs = lane_abs;
    }
  }

  const int shared_exp = (max_abs > 0.0f) ? FloorLog2Positive(max_abs) : 0;
  for (int i = start_idx; i < start_idx + group_size; ++i) {
    const float lane_value = static_cast<float>(RawLaneToSignedInt(in[i]));
    const float scaled = ScaleByPow2(lane_value, -shared_exp);
    const uint8_t code = EncodeMinifloat(scaled, exp_bits, mant_bits, exp_bias);
    out[i] = ScaleByPow2(DecodeMinifloat(code, exp_bits, mant_bits, exp_bias), shared_exp);
  }
}

inline void QuantizeDequantizeMX(const spec::VectorType in,
                                 const NVUINT2 precision_mode,
                                 const NVUINT1 mx_group_size_is_16,
                                 float out[spec::kVectorSize]) {
  const int exp_bits = (precision_mode == spec::kPrecisionMXFP8) ? 4 : 2;
  const int mant_bits = (precision_mode == spec::kPrecisionMXFP8) ? 3 : 1;
  const int exp_bias = (precision_mode == spec::kPrecisionMXFP8) ? 7 : 1;

  if (mx_group_size_is_16 == spec::kMXGroupSize16) {
    QuantizeDequantizeMXGroup(in, 0, 16, exp_bits, mant_bits, exp_bias, out);
  } else {
    QuantizeDequantizeMXGroup(in, 0, 8, exp_bits, mant_bits, exp_bias, out);
    QuantizeDequantizeMXGroup(in, 8, 8, exp_bits, mant_bits, exp_bias, out);
  }
}

inline void ProductSumINT8(const spec::VectorType in_1,
                           const spec::VectorType in_2,
                           spec::AccumScalarType& out) {
  spec::AccumScalarType out_tmp = 0;

#pragma hls_unroll yes
  for (int i = 0; i < spec::kVectorSize; i++) {
    out_tmp += static_cast<spec::AccumScalarType>(in_1[i]) *
               static_cast<spec::AccumScalarType>(in_2[i]);
  }
  out = out_tmp;
}

inline void ProductSumMX(const spec::VectorType in_1,
                         const spec::VectorType in_2,
                         const NVUINT2 precision_mode,
                         const NVUINT1 mx_group_size_is_16,
                         spec::AccumScalarType& out) {
  float qdq_in_1[spec::kVectorSize];
  float qdq_in_2[spec::kVectorSize];

  QuantizeDequantizeMX(in_1, precision_mode, mx_group_size_is_16, qdq_in_1);
  QuantizeDequantizeMX(in_2, precision_mode, mx_group_size_is_16, qdq_in_2);

  float accum = 0.0f;
#pragma hls_unroll yes
  for (int i = 0; i < spec::kVectorSize; ++i) {
    accum += qdq_in_1[i] * qdq_in_2[i];
  }

  const long long accum_max = (1LL << (spec::kAccumWordWidth - 1)) - 1LL;
  const long long accum_min = -accum_max;
  if (accum > static_cast<float>(accum_max)) {
    accum = static_cast<float>(accum_max);
  } else if (accum < static_cast<float>(accum_min)) {
    accum = static_cast<float>(accum_min);
  }

  out = static_cast<spec::AccumScalarType>(RoundToLongLong(accum));
}

inline void Datapath(spec::VectorType weight_in[spec::kNumVectorLanes], 
              spec::VectorType input_in,
              const NVUINT2 precision_mode,
              const NVUINT1 mx_group_size_is_16,
              spec::AccumVectorType& accum_out)
{
  spec::AccumVectorType accum_out_tmp;

#pragma hls_unroll yes
  for (int i = 0; i < spec::kNumVectorLanes; i++) {
    if (precision_mode == spec::kPrecisionINT8) {
      ProductSumINT8(weight_in[i], input_in, accum_out_tmp[i]);
    } else {
      ProductSumMX(weight_in[i], input_in, precision_mode, mx_group_size_is_16, accum_out_tmp[i]);
    }
  }
  accum_out = accum_out_tmp;
}


#endif
