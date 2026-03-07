#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "group_scaler.h"
#include "mx_datapath.h"
#include "mx_types.h"

namespace {

struct ErrorStats {
  float mean = 0.0f;
  float max = 0.0f;
};

std::vector<float> RandomVector(std::size_t n,
                                std::mt19937& rng,
                                float lo = -1.0f,
                                float hi = 1.0f) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> out(n, 0.0f);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = dist(rng);
  }
  return out;
}

std::vector<std::vector<float>> RandomMatrix(std::size_t rows,
                                             std::size_t cols,
                                             std::mt19937& rng,
                                             float lo = -1.0f,
                                             float hi = 1.0f) {
  std::vector<std::vector<float>> out(rows, std::vector<float>(cols, 0.0f));
  for (std::size_t r = 0; r < rows; ++r) {
    out[r] = RandomVector(cols, rng, lo, hi);
  }
  return out;
}

std::vector<std::vector<float>> GemmReference(
    const std::vector<std::vector<float>>& a,
    const std::vector<std::vector<float>>& b) {
  const std::size_t m = a.size();
  const std::size_t n = b[0].size();
  const std::size_t k = a[0].size();

  std::vector<std::vector<float>> out(m, std::vector<float>(n, 0.0f));
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (std::size_t t = 0; t < k; ++t) {
        acc += a[i][t] * b[t][j];
      }
      out[i][j] = acc;
    }
  }
  return out;
}

ErrorStats EvaluateMacError(mx::PrecisionMode mode,
                            std::size_t group_size,
                            std::size_t num_trials,
                            std::mt19937& rng) {
  mx::DualPrecisionMXDatapath dp(group_size);
  dp.RequestMode(mode);
  dp.FlushPipeline();

  float error_sum = 0.0f;
  float error_max = 0.0f;

  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    const auto a = RandomVector(16, rng);
    const auto b = RandomVector(16, rng);

    const float ref = mx::DotProductReference(a, b);
    const float got = dp.MAC(a, b);

    const float scale =
        std::sqrt(std::fabs(mx::DotProductReference(a, a) * mx::DotProductReference(b, b))) + 1e-6f;
    const float norm_error = std::fabs(got - ref) / scale;

    error_sum += norm_error;
    error_max = std::max(error_max, norm_error);
  }

  return {error_sum / static_cast<float>(num_trials), error_max};
}

ErrorStats EvaluateGemmError(mx::PrecisionMode mode,
                             std::size_t group_size,
                             std::size_t num_trials,
                             std::mt19937& rng) {
  mx::DualPrecisionMXDatapath dp(group_size);
  dp.RequestMode(mode);
  dp.FlushPipeline();

  float error_sum = 0.0f;
  float error_max = 0.0f;
  std::size_t count = 0;

  for (std::size_t trial = 0; trial < num_trials; ++trial) {
    const auto a = RandomMatrix(16, 16, rng);
    const auto b = RandomMatrix(16, 16, rng);
    const auto ref = GemmReference(a, b);
    const auto got = dp.GEMM(a, b);

    for (std::size_t i = 0; i < 16; ++i) {
      for (std::size_t j = 0; j < 16; ++j) {
        const float abs_err = std::fabs(got[i][j] - ref[i][j]);
        error_sum += abs_err;
        error_max = std::max(error_max, abs_err);
        ++count;
      }
    }
  }

  return {error_sum / static_cast<float>(count), error_max};
}

bool CheckLEQ(const std::string& label, float value, float limit) {
  const bool pass = value <= limit;
  std::cout << (pass ? "[PASS] " : "[FAIL] ") << std::left << std::setw(46) << label
            << " value=" << std::setw(10) << value << " limit=" << limit << "\n";
  return pass;
}

bool CheckTrue(const std::string& label, bool condition) {
  std::cout << (condition ? "[PASS] " : "[FAIL] ") << label << "\n";
  return condition;
}

}  // namespace

int main() {
  std::mt19937 rng(42);
  int failures = 0;

  std::cout << "============================================================\n";
  std::cout << "Milestone 3 Testbench: Dual-Precision MX Datapath\n";
  std::cout << "============================================================\n\n";

  {
    std::cout << "1) Quantize/dequantize sanity check\n";
    const auto sample = RandomVector(16, rng);

    const auto fp8_qdq = mx::QuantizeDequantizeVector(sample, mx::MXFP8Spec(), 8);
    const auto fp4_qdq = mx::QuantizeDequantizeVector(sample, mx::MXFP4Spec(), 8);

    float fp8_mae = 0.0f;
    float fp4_mae = 0.0f;
    for (std::size_t i = 0; i < sample.size(); ++i) {
      fp8_mae += std::fabs(sample[i] - fp8_qdq[i]);
      fp4_mae += std::fabs(sample[i] - fp4_qdq[i]);
    }
    fp8_mae /= static_cast<float>(sample.size());
    fp4_mae /= static_cast<float>(sample.size());

    std::cout << "  FP8 mean abs reconstruction error: " << fp8_mae << "\n";
    std::cout << "  FP4 mean abs reconstruction error: " << fp4_mae << "\n\n";
    if (!CheckLEQ("FP8 reconstruction MAE", fp8_mae, 0.10f)) {
      ++failures;
    }
    if (!CheckLEQ("FP4 reconstruction MAE", fp4_mae, 0.35f)) {
      ++failures;
    }
  }

  {
    std::cout << "\n2) MAC accuracy checks\n";
    const auto fp8_err = EvaluateMacError(mx::PrecisionMode::MXFP8, 8, 300, rng);
    const auto fp4_err = EvaluateMacError(mx::PrecisionMode::MXFP4, 8, 300, rng);

    std::cout << "  FP8 normalized error mean/max: " << fp8_err.mean << " / " << fp8_err.max << "\n";
    std::cout << "  FP4 normalized error mean/max: " << fp4_err.mean << " / " << fp4_err.max << "\n\n";

    if (!CheckLEQ("FP8 MAC mean normalized error", fp8_err.mean, 0.08f)) {
      ++failures;
    }
    if (!CheckLEQ("FP4 MAC mean normalized error", fp4_err.mean, 0.25f)) {
      ++failures;
    }
  }

  {
    std::cout << "\n3) GEMM accuracy checks (16x16)\n";
    const auto fp8_err = EvaluateGemmError(mx::PrecisionMode::MXFP8, 8, 20, rng);
    const auto fp4_err = EvaluateGemmError(mx::PrecisionMode::MXFP4, 8, 20, rng);

    std::cout << "  FP8 GEMM abs error mean/max: " << fp8_err.mean << " / " << fp8_err.max << "\n";
    std::cout << "  FP4 GEMM abs error mean/max: " << fp4_err.mean << " / " << fp4_err.max << "\n\n";

    if (!CheckLEQ("FP8 GEMM mean abs error", fp8_err.mean, 0.25f)) {
      ++failures;
    }
    if (!CheckLEQ("FP4 GEMM mean abs error", fp4_err.mean, 0.85f)) {
      ++failures;
    }
  }

  {
    std::cout << "\n4) Mode switching safety checks\n";
    mx::DualPrecisionMXDatapath dp(8);
    const auto a = RandomVector(16, rng);
    const auto b = RandomVector(16, rng);

    bool threw = false;
    dp.RequestMode(mx::PrecisionMode::MXFP4);
    try {
      (void)dp.MAC(a, b);
    } catch (const std::runtime_error&) {
      threw = true;
    }

    if (!CheckTrue("MAC blocked until FlushPipeline() after mode request", threw)) {
      ++failures;
    }

    dp.FlushPipeline();
    const bool mode_ok = dp.active_mode() == mx::PrecisionMode::MXFP4;
    if (!CheckTrue("Mode updated to MXFP4 after flush", mode_ok)) {
      ++failures;
    }

    const float out = dp.MAC(a, b);
    (void)out;
    if (!CheckTrue("MAC succeeds after flush", std::isfinite(out))) {
      ++failures;
    }
  }

  {
    std::cout << "\n5) Group size comparison (quality trend)\n";
    const auto fp8_g8 = EvaluateMacError(mx::PrecisionMode::MXFP8, 8, 200, rng);
    const auto fp8_g16 = EvaluateMacError(mx::PrecisionMode::MXFP8, 16, 200, rng);
    std::cout << "  FP8 mean normalized error: group8=" << fp8_g8.mean
              << " group16=" << fp8_g16.mean << "\n";

    // Not a hard mathematical guarantee, but group_size=8 should usually be
    // at least as accurate as group_size=16 for this workload.
    const bool trend_ok = fp8_g8.mean <= (fp8_g16.mean + 0.03f);
    if (!CheckTrue("Group size 8 is not worse than group size 16 (within margin)", trend_ok)) {
      ++failures;
    }
  }

  std::cout << "\n============================================================\n";
  if (failures == 0) {
    std::cout << "ALL CHECKS PASSED\n";
  } else {
    std::cout << failures << " CHECK(S) FAILED\n";
  }
  std::cout << "============================================================\n";

  return failures == 0 ? 0 : 1;
}

