#ifndef CS217_SYSTEMC_MX_TYPES_H_
#define CS217_SYSTEMC_MX_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace mx {

enum class PrecisionMode { MXFP8 = 0, MXFP4 = 1 };

inline const char* ModeToString(PrecisionMode mode) {
  switch (mode) {
    case PrecisionMode::MXFP8:
      return "MXFP8";
    case PrecisionMode::MXFP4:
      return "MXFP4";
    default:
      return "UNKNOWN";
  }
}

struct MiniFloatSpec {
  const char* name;
  int exponent_bits;
  int mantissa_bits;
  int exponent_bias;
};

inline MiniFloatSpec MXFP8Spec() {
  // E4M3: 1 sign + 4 exponent + 3 mantissa.
  return {"E4M3", 4, 3, 7};
}

inline MiniFloatSpec MXFP4Spec() {
  // E2M1: 1 sign + 2 exponent + 1 mantissa.
  return {"E2M1", 2, 1, 1};
}

inline void ValidateGroupSize(std::size_t group_size) {
  if (group_size != 8 && group_size != 16) {
    throw std::invalid_argument(
        "Group size must be 8 or 16 for this milestone model.");
  }
}

}  // namespace mx

#endif  // CS217_SYSTEMC_MX_TYPES_H_

