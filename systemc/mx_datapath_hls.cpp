#include <cmath>
#include <cstdint>

namespace {

inline int floor_log2_positive(float x) {
  if (x <= 0.0f) {
    return 0;
  }
  return static_cast<int>(std::floor(std::log2(x)));
}

inline uint8_t encode_minifloat(float value, int exp_bits, int mant_bits, int exp_bias) {
  const int sign_shift = exp_bits + mant_bits;
  const int exp_mask = (1 << exp_bits) - 1;
  const int mant_mask = (1 << mant_bits) - 1;

  if (!std::isfinite(value) || value == 0.0f) {
    return 0;
  }

  const bool neg = std::signbit(value);
  const float ax = std::fabs(value);

  int exponent = floor_log2_positive(ax);
  float normalized = std::ldexp(ax, -exponent);
  int exp_field = exponent + exp_bias;
  int mantissa = 0;

  if (exp_field <= 0) {
    const float scaled = std::ldexp(ax, -(1 - exp_bias));
    mantissa = static_cast<int>(std::round(scaled * (1 << mant_bits)));
    if (mantissa < 0) {
      mantissa = 0;
    }
    if (mantissa > mant_mask) {
      mantissa = mant_mask;
    }
    exp_field = 0;
  } else {
    const float mantissa_f = (normalized - 1.0f) * static_cast<float>(1 << mant_bits);
    mantissa = static_cast<int>(std::round(mantissa_f));
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

inline float decode_minifloat(uint8_t code, int exp_bits, int mant_bits, int exp_bias) {
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
    value = std::ldexp(frac, 1 - exp_bias);
  } else {
    const float frac = 1.0f + (static_cast<float>(mantissa) / static_cast<float>(1 << mant_bits));
    value = std::ldexp(frac, exp_field - exp_bias);
  }

  return neg ? -value : value;
}

}  // namespace

// mode: 0 -> MXFP8(E4M3), 1 -> MXFP4(E2M1)
extern "C" void mx_datapath_top(const float input[16],
                                const float weight[16],
                                int mode,
                                int group_size,
                                float* output) {
  const int exp_bits = (mode == 0) ? 4 : 2;
  const int mant_bits = (mode == 0) ? 3 : 1;
  const int exp_bias = (mode == 0) ? 7 : 1;
  if (group_size != 8 && group_size != 16) {
    group_size = 8;
  }

  float input_qdq[16];
  float weight_qdq[16];

  for (int g = 0; g < 16; g += group_size) {
    float max_abs_input = 0.0f;
    float max_abs_weight = 0.0f;

    for (int i = g; i < g + group_size && i < 16; ++i) {
      max_abs_input = std::fmax(max_abs_input, std::fabs(input[i]));
      max_abs_weight = std::fmax(max_abs_weight, std::fabs(weight[i]));
    }

    const int shared_exp_input = (max_abs_input > 0.0f) ? floor_log2_positive(max_abs_input) : 0;
    const int shared_exp_weight =
        (max_abs_weight > 0.0f) ? floor_log2_positive(max_abs_weight) : 0;

    for (int i = g; i < g + group_size && i < 16; ++i) {
      const float scaled_input = std::ldexp(input[i], -shared_exp_input);
      const float scaled_weight = std::ldexp(weight[i], -shared_exp_weight);

      const uint8_t input_code = encode_minifloat(scaled_input, exp_bits, mant_bits, exp_bias);
      const uint8_t weight_code = encode_minifloat(scaled_weight, exp_bits, mant_bits, exp_bias);

      input_qdq[i] =
          std::ldexp(decode_minifloat(input_code, exp_bits, mant_bits, exp_bias), shared_exp_input);
      weight_qdq[i] = std::ldexp(
          decode_minifloat(weight_code, exp_bits, mant_bits, exp_bias), shared_exp_weight);
    }
  }

  float acc = 0.0f;
  for (int i = 0; i < 16; ++i) {
    acc += input_qdq[i] * weight_qdq[i];
  }
  *output = acc;
}

