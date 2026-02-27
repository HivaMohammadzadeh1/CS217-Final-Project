#ifndef CS217_SYSTEMC_MXFP_PE_H_
#define CS217_SYSTEMC_MXFP_PE_H_

#include <cstddef>
#include <vector>

#include "group_scaler.h"
#include "mx_types.h"

namespace mx {

class MXProcessingElement {
 public:
  MXProcessingElement(MiniFloatSpec spec, std::size_t group_size)
      : spec_(spec), group_size_(group_size) {
    ValidateGroupSize(group_size_);
  }

  float MAC(const std::vector<float>& input, const std::vector<float>& weight) const {
    return DotProductQuantized(input, weight, spec_, group_size_);
  }

  const MiniFloatSpec& spec() const { return spec_; }
  std::size_t group_size() const { return group_size_; }

 private:
  MiniFloatSpec spec_;
  std::size_t group_size_;
};

class MXFP8PE final : public MXProcessingElement {
 public:
  explicit MXFP8PE(std::size_t group_size = 8)
      : MXProcessingElement(MXFP8Spec(), group_size) {}
};

class MXFP4PE final : public MXProcessingElement {
 public:
  explicit MXFP4PE(std::size_t group_size = 8)
      : MXProcessingElement(MXFP4Spec(), group_size) {}
};

}  // namespace mx

#endif  // CS217_SYSTEMC_MXFP_PE_H_

