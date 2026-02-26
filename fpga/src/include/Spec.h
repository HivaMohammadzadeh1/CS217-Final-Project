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


#ifndef __SPEC__
#define __SPEC__

#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <TypeToBits.h>
#include <connections/marshaller.h>

namespace spec {
 
  const int kIntWordWidth = 8; 
  const int kVectorSize = 16;
  const int kNumVectorLanes = 16;

  // TODO 6:
  //  1. Define the accumulation scale and shift values used in the accumulation unit
  //  2. The intention is to achieve functionality similar to dividing by 12.25 in the accumulation unit

  //////// YOUR CODE STARTS HERE ////////
const int kAccumScale = 167;  // We'll optimize this in Task 2
const int kAccumShift = 11;    // We'll optimize this in Task 2
  
  //////// YOUR CODE ENDS HERE ////////

  const int kNumPE = 4;     
  
  typedef NVUINTW(kIntWordWidth) ScalarType;
  typedef typename nvhls::nv_scvector<ScalarType, kVectorSize> VectorType;

  typedef NVUINTW(kIntWordWidth/2) HalfType;
  typedef typename nvhls::nv_scvector<HalfType, kVectorSize> HalfVectorType;

  const int kAccumWordWidth = 2*kIntWordWidth + kVectorSize - 1;
  typedef NVINTW(kAccumWordWidth) AccumScalarType;
  typedef typename nvhls::nv_scvector<AccumScalarType, kNumVectorLanes> AccumVectorType;

  // Activation unit reg type
  const int kActWordWidth = 32;
  const int kActWordMax = (1 << (kActWordWidth-1)) -1;
  const int kActWordMin = -kActWordMax;
  typedef NVINTW(kActWordWidth) ActScalarType;
  typedef typename nvhls::nv_scvector<ActScalarType, kNumVectorLanes> ActVectorType;

  const int rvaBytes = (kVectorSize * kIntWordWidth) / 8;   


  // K-keans cluster LUT
  const int kNumClusterEntries = 16;
  typedef typename nvhls::nv_scvector<ScalarType, kNumClusterEntries> ClusterType;

  // Standard datatype for streaming protacol between GB and PEs 
  // data: VectorType
  // index: the index to locate memory manager ONLY for PE
  // logical_addr: the logical address, same as vector index

  // Update 02142020
  // Customized datatype for channels  Need to inherit nvhls_message
  class StreamType : public nvhls_message {
   public:
    VectorType data;
    NVUINT2 index;
    NVUINT8 logical_addr;
    static const unsigned int width = 2 + 8 + VectorType::width;
    
    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m & data;
      m & index;
      m & logical_addr;
    }
    StreamType() {
      data = 0;
      index = 0;
      logical_addr = 0;
    }  
    
    StreamType operator= (const NVUINTW(width)& in) {
      StreamType _st(in);
      return _st;
    }
  
    StreamType(const NVUINTW(width) & rawbits) {
      *this = NVUINTToType<StreamType>(rawbits);
    }

    NVUINTW(width) to_rawbits() const {
      return TypeToNVUINT(*this);
    }  
  };

  inline bool operator==(const StreamType& lhs, const StreamType& rhs)
  {
    bool is_equal = true;
    is_equal &= (lhs.data == rhs.data);
    is_equal &= (lhs.index == rhs.index);
    is_equal &= (lhs.logical_addr == rhs.logical_addr);

    return is_equal;
  }

  inline std::ostream& operator<<(std::ostream& os,
                                  const StreamType& _st) {
    os << hex << " data = " << _st.data << " index = " << _st.index << " logical_addr = " << _st.logical_addr << endl;
    
    return os;
  }

}

#endif
