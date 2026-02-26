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

#ifndef PECORE_H
#define PECORE_H

// SystemC and MatchLib includes
#include <ArbitratedScratchpadDP.h>
#include <nvhls_int.h>
#include <nvhls_module.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <systemc.h>

// Project includes
#include "AxiSpec.h"
#include "Datapath/Datapath.h"
#include "PECoreSpec.h"
#include "Spec.h"


// PECore module definition
// Please read all comments carefully for implementation details
// and complete the TODO sections.
class PECore : public match::Module {
  static const int kDebugLevel = 4;
  SC_HAS_PROCESS(PECore);

public:

  Connections::In<bool> start;
  Connections::In<spec::StreamType> input_port;
  Connections::In<spec::Axi::SubordinateToRVA::Write> rva_in;
  Connections::Out<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::Out<spec::ActVectorType> act_port;
  sc_in<NVUINT32> SC_SRAM_CONFIG;

  // Use weight address width as the address format of PEManager
  PEManager<spec::PE::Weight::kAddressWidth>
      pe_manager[spec::PE::kNumPEManagers];

  // PE configuration registers
  PEConfig pe_config;

  // FSM
  enum FSM {
    IDLE,   // wait for start signal
    PRE,    // pre-computation setup
    MAC,    // MAC computation
    SCALE,  // scaling computation
    OUT     // output results
  };
  FSM state;

  // accumulator regs
  spec::AccumVectorType accum_vector;
  spec::ActVectorType act_port_reg;

  // Indicate the Computation part is activated
  bool is_start;

  // while loop control signal (including SRAM I/O)
  // True if need to push out AXI response
  bool w_axi_rsp;
  // RVA input register
  spec::Axi::SubordinateToRVA::Write rva_in_reg;
  // RVA output register
  spec::Axi::SubordinateToRVA::Read rva_out_reg;


  // Single port weight SRAM
  ArbitratedScratchpadDP<
      spec::PE::Weight::kNumBanks,
      spec::PE::Weight::kNumReadPorts,
      spec::PE::Weight::kNumWritePorts,
      spec::PE::Weight::kEntriesPerBank,
      spec::PE::Weight::WordType,
      false,
      true>
      weight_mem;

  // Single port input SRAM
  ArbitratedScratchpadDP<
      spec::PE::Input::kNumBanks,
      spec::PE::Input::kNumReadPorts,
      spec::PE::Input::kNumWritePorts,
      spec::PE::Input::kEntriesPerBank,
      spec::PE::Input::WordType,
      false,
      true>
      input_mem;

  // Weight buffer signals
  // Read address for weight buffer
  spec::PE::Weight::Address weight_read_addrs[spec::PE::Weight::kNumReadPorts];
  // Read request valid for weight buffer
  bool weight_read_req_valid[spec::PE::Weight::kNumReadPorts];
  // Write address for weight buffer
  spec::PE::Weight::Address weight_write_addrs[spec::PE::Weight::kNumWritePorts];
  // Write request valid for weight buffer
  bool weight_write_req_valid[spec::PE::Weight::kNumWritePorts];
  // Write data for weight buffer
  spec::PE::Weight::WordType weight_write_data[spec::PE::Weight::kNumWritePorts];
  // Read acknowledge for weight buffer
  bool weight_read_ack[spec::PE::Weight::kNumReadPorts];
  // Write acknowledge for weight buffer
  bool weight_write_ack[spec::PE::Weight::kNumWritePorts];
  // Read ready for weight buffer
  bool weight_read_ready[spec::PE::Weight::kNumReadPorts];
  // Read data output for weight buffer
  spec::PE::Weight::WordType
      weight_port_read_out[spec::PE::Weight::kNumReadPorts];
  // Read data valid for weight buffer
  bool weight_port_read_out_valid[spec::PE::Weight::kNumReadPorts];

  // Input Buffer signals
  // Read address for input buffer
  spec::PE::Input::Address input_read_addrs[spec::PE::Input::kNumReadPorts];
  // Read request valid for input buffer
  bool input_read_req_valid[spec::PE::Input::kNumReadPorts];
  // Write address for input buffer
  spec::PE::Input::Address input_write_addrs[spec::PE::Input::kNumWritePorts];
  // Write request valid for input buffer
  bool input_write_req_valid[spec::PE::Input::kNumWritePorts];
  // Write data for input buffer
  spec::PE::Input::WordType input_write_data[spec::PE::Input::kNumWritePorts];
  // Read acknowledge for input buffer
  bool input_read_ack[spec::PE::Input::kNumReadPorts];
  // Write acknowledge for input buffer
  bool input_write_ack[spec::PE::Input::kNumWritePorts];
  // Read ready for input buffer
  bool input_read_ready[spec::PE::Input::kNumReadPorts];
  // Read data output for input buffer
  spec::PE::Input::WordType input_port_read_out[spec::PE::Input::kNumReadPorts];
  // Read data valid for input buffer
  bool input_port_read_out_valid[spec::PE::Input::kNumReadPorts];


  // Constructor
  PECore(sc_module_name nm) :
      match::Module(nm),
      start("start"),
      input_port("input_port"),
      rva_in("rva_in"),
      rva_out("rva_out"),
      act_port("act_port"),
      SC_SRAM_CONFIG("SRAM_CONFIG") {
    SC_THREAD(PECoreRun);              // Main PECore process
    sensitive << clk.pos();            // Sensitive to positive clock edge
    async_reset_signal_is(rst, false); // async reset active low

    /////////////////////////////////////////////////////////////////
    // We named our reset signal "rst" here for compatibility purposes,
    // but please know that this is NOT ideal in industry practice.
    // Conventions would prefer that you use clearer names such as
    // "arst_b" (for *async* reset *bar*).
    /////////////////////////////////////////////////////////////////
  } // PECore constructor

  // Reset module
  void Reset() {
    state    = IDLE; // reset state
    is_start = 0;    // reset start signal
    for (unsigned i = 0; i < spec::PE::kNumPEManagers; i++) {
      pe_manager[i].Reset(); // reset PE manager counters
    }
    pe_config.Reset(); // reset PE configuration registers
    ResetAccum();      // reset accumulator registers
    ResetPorts();      // reset input/output ports
  } // Reset

  // Reset input/output ports
  void ResetPorts() {
    start.Reset();
    input_port.Reset();
    rva_in.Reset();
    rva_out.Reset();
    act_port.Reset();
  } // ResetPorts

  // Reset accumulator registers
  void ResetAccum() {
    accum_vector = 0;
    act_port_reg = 0;
  } // ResetAccum

  // Reset SRAM buffer interface signals
  void ResetBufferInputs() {

    // Reset all read ports for weight buffer
#pragma hls_unroll yes
    for (unsigned i = 0; i < spec::PE::Weight::kNumReadPorts; i++) {
      weight_read_addrs[i]     = 0;
      weight_read_req_valid[i] = 0;
      weight_read_ready[i]     = 0;
    }

    // Reset all write ports for weight buffer even though only one is used
#pragma hls_unroll yes
    for (unsigned i = 0; i < spec::PE::Weight::kNumWritePorts; i++) {
      weight_write_addrs[i]     = 0;
      weight_write_req_valid[i] = 0;
      weight_write_data[i]      = 0;
    }

    // Reset all read ports for input buffer even though only one is used
#pragma hls_unroll yes
    for (unsigned i = 0; i < spec::PE::Input::kNumReadPorts; i++) {
      input_read_addrs[i]     = 0;
      input_read_req_valid[i] = 0;
      input_read_ready[i]     = 0;
    }

    // Reset all write ports for input buffer even though only one is used
#pragma hls_unroll yes
    for (unsigned i = 0; i < spec::PE::Input::kNumWritePorts; i++) {
      input_write_addrs[i]     = 0;
      input_write_req_valid[i] = 0;
      input_write_data[i]      = 0;
    }

  } // ResetBufferInputs


  void DecodeAxiWrite(const spec::Axi::SubordinateToRVA::Write &rva_in_reg)
  {
    NVUINT4 tmp = nvhls::get_slc<4>(rva_in_reg.addr, 20);
    NVUINT16 local_index = nvhls::get_slc<16>(rva_in_reg.addr, 4);
    //NVUINT32 addr = nvhls::get_slc<32>(rva_in_reg.addr, 0);

    switch (tmp)
    {
    case 0x4:
    { // PEconfig
      switch (local_index)
      {
      case 0x1:
      {
        pe_config.PEConfigWrite(rva_in_reg.data);
        break;
      }
      case 0x2:
      { // manager 0 config
        pe_manager[0].PEManagerWrite(rva_in_reg.data);
        break;
      }
      default:
      {
        break;
      }
      }
      break;
    }
    case 0x5:
    { // Weight Buffer
      weight_write_addrs[0] = local_index;
      weight_write_req_valid[0] = 1;
      weight_write_data[0] = rva_in_reg.data;
      break;
    }
    case 0x6:
    { // Input Buffer (lock datapath)
      input_write_addrs[0] = local_index;
      input_write_req_valid[0] = 1;
      input_write_data[0] = rva_in_reg.data;
      break;
    }
    default:
    {
      break;
    }
    }
  }

  void DecodeAxiRead(const spec::Axi::SubordinateToRVA::Write &rva_in_reg)
  {
    NVUINT4 tmp = nvhls::get_slc<4>(rva_in_reg.addr, 20);
    NVUINT16 local_index = nvhls::get_slc<16>(rva_in_reg.addr, 4);

    // Set Push Response
    w_axi_rsp = 1;
    rva_out_reg.data = 0;

    switch (tmp)
    {
    case 0x3:
    { // Write implemented inside PEModule
      rva_out_reg.data = SC_SRAM_CONFIG.read();
      break;
    }
    case 0x4:
    { // PEconfig
      switch (local_index)
      {
      case 0x1:
      {
        pe_config.PEConfigRead(rva_out_reg.data);
        break;
      }
      case 0x2:
      { // manager 0 config
        pe_manager[0].PEManagerRead(rva_out_reg.data);
        break;
      }
      default:
      {
        break;
      }
      }
      break;
    }
    case 0x5:
    { // Weight Buffer
      // w_axir_weight = 1;
      weight_read_addrs[0] = local_index;
      weight_read_req_valid[0] = 1;
      weight_read_ready[0] = 1;
      break;
    }
    case 0x6:
    { // Input Buffer (lock datapath)
      // w_axir_input = 1;
      input_read_addrs[0] = local_index;
      input_read_req_valid[0] = 1;
      input_read_ready[0] = 1;
      break;
    }
    default:
    {
      break;
    }
    }
  }


  void Initialize()
  {
    ResetBufferInputs();
    w_axi_rsp = 0;
  }

  void DecodeAxi()
  {
    if (rva_in.PopNB(rva_in_reg))
    {
      // w_axi_req = 1;
      CDCOUT(sc_time_stamp() << " PECore: " << name() << "RVA Pop " << endl, kDebugLevel);
      if (rva_in_reg.rw)
      {
        DecodeAxiWrite(rva_in_reg);
      }
      else
      {
        DecodeAxiRead(rva_in_reg);
      }
    }
  }

  void RunFSM()
  {
    // Can do FSM only when and no Axi on input
    // Can only move forward to computation if is_start = 1
    // Can only pop message from GB buffer in IDLE state

    switch (state)
    {
    case IDLE:
    {
      spec::StreamType input_port_reg;
      if (input_port.PopNB(input_port_reg))
      {
        NVUINT4 m_index = input_port_reg.index;
        input_write_addrs[0] = pe_manager[m_index].GetInputAddr(input_port_reg.logical_addr);
        input_write_req_valid[0] = 1;
        input_write_data[0] = input_port_reg.data;
      }
      break;
    }
    case PRE:
    {
      break;
    }
    case MAC:
    {
      NVUINT4 m_index = pe_config.ManagerIndex();
      // Do MAC (Datapath)
      // set weight SRAM read
      spec::PE::Weight::Address weight_base;      
        weight_base = pe_manager[m_index].GetWeightAddr(pe_config.InputIndex(), pe_config.OutputIndex(), 0);
#pragma hls_unroll yes
        for (int i = 0; i < spec::kNumVectorLanes; i++)
        {
          weight_read_addrs[i] = weight_base + i;
          weight_read_req_valid[i] = 1;
          weight_read_ready[i] = 1;
        }
      
      // set input SRAM read
      input_read_ready[0] = 1;
      input_read_addrs[0] = pe_manager[m_index].GetInputAddr(pe_config.InputIndex());
      input_read_req_valid[0] = 1;

      break;
    }

    case SCALE:
    {
      break;
    }
    case OUT:
    {
      break;
    }

    default:
    {
      break;
    }
    }
  }

  void BufferAccess()
  {
    weight_mem.run(
        weight_read_addrs,
        weight_read_req_valid,
        weight_write_addrs,
        weight_write_req_valid,
        weight_write_data,
        weight_read_ack,
        weight_write_ack,
        weight_read_ready,
        weight_port_read_out,
        weight_port_read_out_valid);
    input_mem.run(
        input_read_addrs,
        input_read_req_valid,
        input_write_addrs,
        input_write_req_valid,
        input_write_data,
        input_read_ack,
        input_write_ack,
        input_read_ready,
        input_port_read_out,
        input_port_read_out_valid);
  }

  void RunMac()
  {
    if (state == MAC)
    {
      spec::VectorType dp_in0[spec::kNumVectorLanes]; // weight lanes
      spec::VectorType dp_in1; // input vector
      spec::AccumVectorType dp_out; // Datapath output

      // TODO 3: 
      // 1. Implement the operation that gets the weight vector lanes and input vector from the SRAM read outputs
      // 2. Call the Datapath function to compute the product sum between the weight vector lanes and input vector
      // 3. Store the result in accum_vector
      // 4. You can use pragmas to optimize the loop used for getting the weight vector lanes and to store the result in accum_vector

      //////// YOUR CODE STARTS HERE ////////
// Read weight vectors from SRAM
#pragma hls_unroll yes
for (int i = 0; i < spec::kNumVectorLanes; i++) {
  dp_in0[i] = weight_port_read_out[i];
}

// Read input vector from SRAM  
dp_in1 = input_port_read_out[0];

// Call Datapath to compute product sum
Datapath(dp_in0, dp_in1, dp_out);

// Store result in accumulator
accum_vector = dp_out;
      //////// YOUR CODE ENDS HERE ////////
    }
      
  }

  void RunScale()
  {
    if (state == SCALE)
    {

      NVUINT8 scale = spec::kAccumScale;
      NVUINT8 right_shift = spec::kAccumShift;
      spec::AccumVectorType accum_vector_out;

      // TODO 5:
      // 1. Implement the scaling operation on accum_vector using scale and right_shift
      // 2. Store the result in act_port_reg after checking for overflow and cutting to the activation unit range
      // 3. You can use pragmas to optimize the loop used for scaling and storing the result
      // 4. If overflow occurs, set the value to kActWordMax or kActWordMin accordingly (these can be found in Spec.h)


      /////// YOUR CODE STARTS HERE ////////
#pragma hls_unroll yes
for (int i = 0; i < spec::kNumVectorLanes; i++) {
  // Perform scaling: (accum * scale) >> right_shift
  spec::ActScalarType scaled_val = (accum_vector[i] * scale) >> right_shift;
  
  // Check overflow and clamp to activation range
  if (scaled_val > spec::kActWordMax) {
    accum_vector_out[i] = spec::kActWordMax;
  } else if (scaled_val < spec::kActWordMin) {
    accum_vector_out[i] = spec::kActWordMin;
  } else {
    accum_vector_out[i] = scaled_val;
  }
}

// Store result in output register
#pragma hls_unroll yes
for (int i = 0; i < spec::kNumVectorLanes; i++) {
  act_port_reg[i] = accum_vector_out[i];
}
      /////// YOUR CODE ENDS HERE ////////

    }
  }

  void PushOutput()
  {
    if (state == OUT)
    {
      act_port.Push(act_port_reg);
    }
  }

  void PushAxiRsp()
  {
    if (w_axi_rsp)
    {
      // process read rsp from SRAM
      if (weight_port_read_out_valid[0])
      {
        rva_out_reg.data = weight_port_read_out[0].to_rawbits();
      }
      // MERGE SCALE
      else if (input_port_read_out_valid[0])
      {
        rva_out_reg.data = input_port_read_out[0].to_rawbits();
      }
      rva_out.Push(rva_out_reg);
    }
  }

  // Update FSM State and PE_config counters
  void UpdateFSM()
  {
    FSM next_state;
    switch (state)
    {
    case IDLE:
    {
      // Check start signal only in IDLE state
      bool start_reg;
      is_start = 0; // reset is_start
      if (start.PopNB(start_reg)) {
        is_start = pe_config.is_valid && start_reg;
        CDCOUT(sc_time_stamp() << " PECore: " << name() << " Start" << endl, kDebugLevel);
      }
      next_state = is_start ? PRE : IDLE;
      break;
    }
    case PRE:
    {
      ResetAccum();
      NVUINT4 m_index = pe_config.ManagerIndex();
      if (pe_manager[m_index].zero_active && pe_config.is_zero_first)
      {
        // skip MAC
        next_state = SCALE;
      }
      else
      {
        next_state = MAC;
      }
      break;
    }

    case MAC:
    {
      NVUINT4 m_index = pe_config.ManagerIndex();
      bool is_input_end = 0;
      pe_config.UpdateInputCounter(pe_manager[m_index].num_input, is_input_end);
      if (is_input_end)
      {
        next_state = SCALE;
      }
      else
      {
        next_state = MAC;
      }
      break;
    }
    case SCALE:
    {
      next_state = OUT;
      break;
    }

    case OUT:
    {
      // TODO 4:
      // 1. Call pe_config.UpdateManagerCounter() to update output counter
      // 2. If all outputs are done, move to IDLE state
      // 3. Else, move to PRE state for next computation
      
      //////// YOUR CODE STARTS HERE ////////
bool is_output_end = 0;
pe_config.UpdateManagerCounter(is_output_end);

if (is_output_end) {
  next_state = IDLE;
} else {
  next_state = PRE;
}
      /////// YOUR CODE ENDS HERE ////////
      break;
    }
    default:
    {
      next_state = IDLE; // Minor fix 02262019
      break;
    }
    }
    state = next_state;
  }

  void PECoreRun()
  {
    Reset();

#pragma hls_pipeline_init_interval 1
    while (1) {
      Initialize();

      // Decode AXI requests with highest priority (mutually exclusive with FSM)
      if (rva_in.PopNB(rva_in_reg)) {
        CDCOUT(sc_time_stamp() << " PECore: " << name() << "RVA Pop " << endl, kDebugLevel);
        if (rva_in_reg.rw) {
          DecodeAxiWrite(rva_in_reg);
        } else {
          DecodeAxiRead(rva_in_reg);
        }
        BufferAccess();
      } else {
        // Only run FSM when no AXI request is pending
        // Can only pop message from GB buffer in IDLE state (handled in RunFSM)
        // Can only move forward to computation if is_start = 1 (handled in UpdateFSM)
        RunFSM();
        BufferAccess();
        RunMac();
        RunScale();
        PushOutput();
        UpdateFSM();
      }
      PushAxiRsp();

      wait();
    }
  }
};
#endif
