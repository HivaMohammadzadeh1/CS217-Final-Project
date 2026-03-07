#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

extern "C" void mx_datapath_top(const float input[16],
                                const float weight[16],
                                int mode,
                                int group_size,
                                float* output);

namespace {

float reference_dot(const float a[16], const float b[16]) {
  float acc = 0.0f;
  for (int i = 0; i < 16; ++i) {
    acc += a[i] * b[i];
  }
  return acc;
}

}  // namespace

int main() {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  float input[16];
  float weight[16];

  for (int i = 0; i < 16; ++i) {
    input[i] = dist(rng);
    weight[i] = dist(rng);
  }

  const float ref = reference_dot(input, weight);

  float out_fp8 = 0.0f;
  mx_datapath_top(input, weight, 0, 8, &out_fp8);

  float out_fp4 = 0.0f;
  mx_datapath_top(input, weight, 1, 8, &out_fp4);

  std::cout << "Reference dot = " << ref << "\n";
  std::cout << "MXFP8 dot     = " << out_fp8 << "  abs_err=" << std::fabs(out_fp8 - ref) << "\n";
  std::cout << "MXFP4 dot     = " << out_fp4 << "  abs_err=" << std::fabs(out_fp4 - ref) << "\n";

  // Loose thresholds for a smoke test.
  if (std::fabs(out_fp8 - ref) > 2.0f) {
    std::cerr << "MXFP8 smoke test failed.\n";
    return EXIT_FAILURE;
  }
  if (std::fabs(out_fp4 - ref) > 5.0f) {
    std::cerr << "MXFP4 smoke test failed.\n";
    return EXIT_FAILURE;
  }

  std::cout << "HLS C-sim smoke test passed.\n";
  return EXIT_SUCCESS;
}

