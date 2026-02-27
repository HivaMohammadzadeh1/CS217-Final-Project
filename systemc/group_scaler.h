#ifndef CS217_SYSTEMC_GROUP_SCALER_H_
#define CS217_SYSTEMC_GROUP_SCALER_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "mx_types.h"

namespace mx {

struct QuantizedVector {
  std::size_t group_size = 8;
  std::vector<int> shared_exponents;
  std::vector<uint8_t> packed_values;
};

inline int FloorLog2Positive(float x) {
  if (x <= 0.0f) {
    throw std::invalid_argument("FloorLog2Positive requires x > 0.");
  }
  return static_cast<int>(std::floor(std::log2(x)));
}

inline uint8_t EncodeMinifloat(float value, const MiniFloatSpec& spec) {
  const int exp_bits = spec.exponent_bits;
  const int mant_bits = spec.mantissa_bits;
  const int sign_shift = exp_bits + mant_bits;
  const int exp_mask = (1 << exp_bits) - 1;
  const int mant_mask = (1 << mant_bits) - 1;

  if (!std::isfinite(value) || value == 0.0f) {
    return 0;
  }

  const bool neg = std::signbit(value);
  const float ax = std::fabs(value);

  int exponent = FloorLog2Positive(ax);
  float normalized = std::ldexp(ax, -exponent);  // ax / (2^exponent)
  int exp_field = exponent + spec.exponent_bias;

  int mantissa = 0;

  if (exp_field <= 0) {
    // Subnormal region.
    const float scaled = std::ldexp(ax, -(1 - spec.exponent_bias));
    mantissa = static_cast<int>(std::round(scaled * (1 << mant_bits)));
    mantissa = std::max(0, std::min(mantissa, mant_mask));
    exp_field = 0;
  } else {
    const float mantissa_f = (normalized - 1.0f) * static_cast<float>(1 << mant_bits);
    mantissa = static_cast<int>(std::round(mantissa_f));

    // Carry case: 1.111 rounds to 10.000.
    if (mantissa == (1 << mant_bits)) {
      mantissa = 0;
      exp_field += 1;
    }

    if (exp_field >= exp_mask) {
      // Saturate to max finite value.
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

inline float DecodeMinifloat(uint8_t code, const MiniFloatSpec& spec) {
  const int exp_bits = spec.exponent_bits;
  const int mant_bits = spec.mantissa_bits;
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
    // Subnormal decode.
    const float frac = static_cast<float>(mantissa) / static_cast<float>(1 << mant_bits);
    value = std::ldexp(frac, 1 - spec.exponent_bias);
  } else {
    const float frac = 1.0f + (static_cast<float>(mantissa) / static_cast<float>(1 << mant_bits));
    value = std::ldexp(frac, exp_field - spec.exponent_bias);
  }

  return neg ? -value : value;
}

inline QuantizedVector QuantizeVector(const std::vector<float>& values,
                                      const MiniFloatSpec& spec,
                                      std::size_t group_size) {
  ValidateGroupSize(group_size);

  QuantizedVector q;
  q.group_size = group_size;
  q.packed_values.resize(values.size(), 0);

  const std::size_t num_groups = (values.size() + group_size - 1) / group_size;
  q.shared_exponents.resize(num_groups, 0);

  for (std::size_t group = 0; group < num_groups; ++group) {
    const std::size_t start = group * group_size;
    const std::size_t end = std::min(start + group_size, values.size());

    float max_abs = 0.0f;
    for (std::size_t i = start; i < end; ++i) {
      max_abs = std::max(max_abs, std::fabs(values[i]));
    }

    if (max_abs == 0.0f) {
      q.shared_exponents[group] = 0;
      continue;
    }

    const int shared_exp = FloorLog2Positive(max_abs);
    q.shared_exponents[group] = shared_exp;

    for (std::size_t i = start; i < end; ++i) {
      const float scaled = std::ldexp(values[i], -shared_exp);
      q.packed_values[i] = EncodeMinifloat(scaled, spec);
    }
  }

  return q;
}

inline std::vector<float> DequantizeVector(const QuantizedVector& q,
                                           const MiniFloatSpec& spec) {
  std::vector<float> out(q.packed_values.size(), 0.0f);
  const std::size_t num_groups = q.shared_exponents.size();

  for (std::size_t group = 0; group < num_groups; ++group) {
    const std::size_t start = group * q.group_size;
    const std::size_t end = std::min(start + q.group_size, q.packed_values.size());
    const int shared_exp = q.shared_exponents[group];

    for (std::size_t i = start; i < end; ++i) {
      const float scaled = DecodeMinifloat(q.packed_values[i], spec);
      out[i] = std::ldexp(scaled, shared_exp);
    }
  }

  return out;
}

inline std::vector<float> QuantizeDequantizeVector(const std::vector<float>& values,
                                                   const MiniFloatSpec& spec,
                                                   std::size_t group_size) {
  return DequantizeVector(QuantizeVector(values, spec, group_size), spec);
}

inline float DotProductReference(const std::vector<float>& a,
                                 const std::vector<float>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("DotProductReference: size mismatch.");
  }

  float sum = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

inline float DotProductQuantized(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 const MiniFloatSpec& spec,
                                 std::size_t group_size) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("DotProductQuantized: size mismatch.");
  }

  const std::vector<float> aq = QuantizeDequantizeVector(a, spec, group_size);
  const std::vector<float> bq = QuantizeDequantizeVector(b, spec, group_size);
  return DotProductReference(aq, bq);
}

}  // namespace mx

#endif  // CS217_SYSTEMC_GROUP_SCALER_H_

