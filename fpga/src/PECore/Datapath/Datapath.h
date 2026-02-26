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

//  Datapath that does computation on adpfloat 
#include <nvhls_int.h>
#include <nvhls_types.h>
#include "Spec.h"

// Zero skipping 
inline void ProductSum(const spec::VectorType in_1, const spec::VectorType in_2, spec::AccumScalarType& out) {
  spec::AccumScalarType out_tmp = 0; 
  
  // TODO 1:
  //  1. Implement the product sum operation between in_1 and in_2, store the result in out
  //  2. Consider the type of in_1, in_2, and out when implementing the function
  //  3. You can use pragmas to optimize the loop for HLS synthesis
  //  4. We want to have a adder tree structure for the summation operation (this impacts performace)
  //  5. This can be achieved through proper use of pragmas

  //////// YOUR CODE STARTS HERE ////////
#pragma hls_unroll yes
for (int i = 0; i < spec::kVectorSize; i++) {
  out_tmp += static_cast<spec::AccumScalarType>(in_1[i]) * static_cast<spec::AccumScalarType>(in_2[i]);
}
out = out_tmp;
  //////// YOUR CODE ENDS HERE ////////
}

inline void Datapath(spec::VectorType weight_in[spec::kNumVectorLanes], 
              spec::VectorType input_in,
              spec::AccumVectorType& accum_out)
{
  spec::AccumVectorType accum_out_tmp; 
  
  // TODO 2:
  //  1. Implement the datapath that computes the product sum between each weight vector lane and the input vector
  //  2. Store the result in accum_out
  //  3. You can use pragmas to optimize the loop for HLS synthesis 
  //  4. It is important to remember that input is broadcasted to all lanes

  //////// YOUR CODE STARTS HERE ////////
#pragma hls_unroll yes
for (int i = 0; i < spec::kNumVectorLanes; i++) {
  ProductSum(weight_in[i], input_in, accum_out_tmp[i]);
}
accum_out = accum_out_tmp;
  //////// YOUR CODE ENDS HERE ////////
}


#endif
