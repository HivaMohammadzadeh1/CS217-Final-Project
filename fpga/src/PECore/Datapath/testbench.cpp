/* Integer MAC self-checking testbench */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <systemc.h>
#include <mc_scverify.h>
#include <match_scverify.h>
#include "Spec.h"
#include "Datapath.h"
#include <bitset>
#include <testbench/nvhls_rand.h>

CCS_MAIN (int argc, char *argv[]) {
  // Seed RNG
  nvhls::set_random_seed();
  srand(time(NULL));

  // DUT containers
  spec::VectorType dp_weight[spec::kNumVectorLanes];
  spec::VectorType dp_input;
  spec::AccumVectorType dp_output;

  // Reference containers
  long long ref_weight[spec::kNumVectorLanes][spec::kVectorSize];
  long long ref_input[spec::kVectorSize];
  long long ref_output[spec::kNumVectorLanes];

  // Randomize weights
  for (int i = 0; i < spec::kNumVectorLanes; i++) {
    for (int j = 0; j < spec::kVectorSize; j++) {
      dp_weight[i][j] = nvhls::get_rand<spec::kIntWordWidth>();
      if (j % 5 == 0) dp_weight[i][j] = 0; // zero injection
      ref_weight[i][j] = (long long) dp_weight[i][j];
    }
  }

  // Randomize inputs
  for (int i = 0; i < spec::kVectorSize; i++) {
    dp_input[i] = nvhls::get_rand<spec::kIntWordWidth>();
    if (i % 4 == 0) dp_input[i] = 0; // zero injection
    ref_input[i] = (long long) dp_input[i];
  }

  // Compute reference outputs
  for (int i = 0; i < spec::kNumVectorLanes; i++) {
    long long acc = 0;
    for (int j = 0; j < spec::kVectorSize; j++) {
      acc += ref_weight[i][j] * ref_input[j];
    }
    ref_output[i] = acc;
  }

  // Run DUT
  CCS_DESIGN(Datapath)(dp_weight, dp_input, dp_output);

  // Compare outputs
  int total_tests = spec::kNumVectorLanes;
  int failures = 0;

  cout << endl << "Lane\tDUT Output\tRef Output\tStatus" << endl;
  for (int i = 0; i < spec::kNumVectorLanes; i++) {
    long long dut_val = (long long) dp_output[i];
    bool pass = (dut_val == ref_output[i]);
    if (!pass) failures++;
    cout << i << "\t" << dut_val << "\t\t" << ref_output[i]
         << "\t\t" << (pass ? "PASS" : "FAIL") << endl;
  }

  // Summary
  cout << "\n====================================" << endl;
  cout << "Test Summary: " << (total_tests - failures) << "/" 
       << total_tests << " passed." << endl;
  if (failures > 0) {
    cout << "FAILURE: " << failures << " mismatches detected." << endl;
    CCS_RETURN(1); // non-zero exit for CI
  } else {
    cout << "SUCCESS: All outputs match reference." << endl;
    CCS_RETURN(0);
  }
}
