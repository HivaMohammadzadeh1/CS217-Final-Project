#ifndef CS217_SYSTEMC_MX_DATAPATH_H_
#define CS217_SYSTEMC_MX_DATAPATH_H_

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "mx_types.h"
#include "mxfp_pe.h"

namespace mx {

class DualPrecisionMXDatapath {
 public:
  explicit DualPrecisionMXDatapath(std::size_t group_size = 8)
      : active_mode_(PrecisionMode::MXFP8),
        pending_mode_(PrecisionMode::MXFP8),
        switch_pending_(false),
        flush_count_(0),
        fp8_pe_(group_size),
        fp4_pe_(group_size) {}

  PrecisionMode active_mode() const { return active_mode_; }
  bool flush_required() const { return switch_pending_; }
  std::size_t flush_count() const { return flush_count_; }
  std::size_t group_size() const { return fp8_pe_.group_size(); }

  // Requests a mode switch. New mode takes effect only after flush_pipeline().
  void RequestMode(PrecisionMode mode) {
    pending_mode_ = mode;
    switch_pending_ = (pending_mode_ != active_mode_);
  }

  // Models draining/clearing state when precision mode changes.
  void FlushPipeline() {
    if (!switch_pending_) {
      return;
    }
    active_mode_ = pending_mode_;
    switch_pending_ = false;
    ++flush_count_;
  }

  float MAC(const std::vector<float>& input, const std::vector<float>& weight) const {
    if (switch_pending_) {
      throw std::runtime_error(
          "Mode switch pending. Call FlushPipeline() before MAC().");
    }
    if (active_mode_ == PrecisionMode::MXFP8) {
      return fp8_pe_.MAC(input, weight);
    }
    return fp4_pe_.MAC(input, weight);
  }

  std::vector<std::vector<float>> GEMM(
      const std::vector<std::vector<float>>& a,
      const std::vector<std::vector<float>>& b) const {
    if (switch_pending_) {
      throw std::runtime_error(
          "Mode switch pending. Call FlushPipeline() before GEMM().");
    }
    ValidateMatrices(a, b);

    const std::size_t m = a.size();
    const std::size_t n = b[0].size();
    const std::size_t k = a[0].size();

    std::vector<std::vector<float>> bt(n, std::vector<float>(k, 0.0f));
    for (std::size_t row = 0; row < b.size(); ++row) {
      for (std::size_t col = 0; col < n; ++col) {
        bt[col][row] = b[row][col];
      }
    }

    std::vector<std::vector<float>> out(m, std::vector<float>(n, 0.0f));
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        out[i][j] = MAC(a[i], bt[j]);
      }
    }
    return out;
  }

 private:
  static void ValidateMatrices(const std::vector<std::vector<float>>& a,
                               const std::vector<std::vector<float>>& b) {
    if (a.empty() || b.empty() || a[0].empty() || b[0].empty()) {
      throw std::invalid_argument("GEMM requires non-empty matrices.");
    }

    const std::size_t a_cols = a[0].size();
    for (const auto& row : a) {
      if (row.size() != a_cols) {
        throw std::invalid_argument("Matrix A is not rectangular.");
      }
    }

    const std::size_t b_cols = b[0].size();
    for (const auto& row : b) {
      if (row.size() != b_cols) {
        throw std::invalid_argument("Matrix B is not rectangular.");
      }
    }

    if (a_cols != b.size()) {
      throw std::invalid_argument(
          "GEMM dimension mismatch: A columns must match B rows.");
    }
  }

  PrecisionMode active_mode_;
  PrecisionMode pending_mode_;
  bool switch_pending_;
  std::size_t flush_count_;
  MXFP8PE fp8_pe_;
  MXFP4PE fp4_pe_;
};

}  // namespace mx

#endif  // CS217_SYSTEMC_MX_DATAPATH_H_

