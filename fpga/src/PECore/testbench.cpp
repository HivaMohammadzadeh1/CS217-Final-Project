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

#include <systemc.h>
#include <mc_scverify.h>
#include <testbench/nvhls_rand.h>
#include <nvhls_connections.h>
#include <map>
#include <vector>
#include <deque>
#include <utility>
#include <sstream>
#include <string>
#include <cstdlib>
#include <math.h> // testbench only
#include <queue>

#include "PECore.h"
#include "Spec.h"
#include "AxiSpec.h"
#include "helper.h"

#include <iostream>
#include <sstream>
#include <iomanip>

#define NVHLS_VERIFY_BLOCKS (PECore)
#include <nvhls_verify.h>

#ifdef COV_ENABLE
#pragma CTC SKIP
#endif

// ---------------- Source ----------------
SC_MODULE(Source)
{
  sc_in<bool> clk;
  sc_in<bool> rst;
  Connections::Out<bool> start;
  Connections::Out<spec::StreamType> input_port;
  Connections::Out<spec::Axi::SubordinateToRVA::Write> rva_in;

  std::vector<spec::Axi::SubordinateToRVA::Write> src_vec;

  SC_CTOR(Source)
  {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void run()
  {
    // Reset handshakes
    start.Reset();
    input_port.Reset();
    rva_in.Reset();

    wait(); // wait for reset release

    for (unsigned i = 0; i < src_vec.size(); i++)
    {
      if (src_vec[i].rw == 1)
      {
        /*std::cout << std::hex << sc_time_stamp()
                  << " Source WRITE data " << src_vec[i].data
                  << " and addr " << src_vec[i].addr << std::endl;*/
        start.Push(false);
        rva_in.Push(src_vec[i]);
      }

      wait();
    }


    wait(100, SC_NS); // wait for DUT to process the start signal

    start.Push(true);
    wait(100, SC_NS); // wait for DUT to process the start signal

    for (unsigned i = 0; i < src_vec.size(); i++)
    {
      if (src_vec[i].rw == 0)
      {
        /*std::cout << std::hex << sc_time_stamp()
                  << " Source READ addr " << src_vec[i].addr << std::endl;*/
        start.Push(false);
        rva_in.Push(src_vec[i]);
      }

      wait();
    }
  }
};

// ---------------- Dest ----------------
SC_MODULE(Dest)
{
  sc_in<bool> clk;
  sc_in<bool> rst;
  Connections::In<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::In<spec::ActVectorType> act_port;

  spec::ActVectorType act_port_reg;
  spec::ActVectorType act_vector;
  std::vector<spec::Axi::SubordinateToRVA::Read> dest_vec;
  spec::Axi::SubordinateToRVA::Read rva_out_dest;

  SC_CTOR(Dest)
  {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void run()
  {
    rva_out.Reset();
    act_port.Reset();

    wait();

    unsigned i = 0;
    while (1)
    {
      if (act_port.PopNB(act_port_reg))
      {
        double diff = 0;
        for (int j = 0; j < spec::kNumVectorLanes; j++)
        {
          std::cout << "ActPort Computed value = " << act_port_reg[j] << " and expected value = " << act_vector[j] << std::endl;
          diff += fabs(double(act_port_reg[j]) - double(act_vector[j]))/
                        double(act_vector[j]);
        }
        std::cout << "Dest: Difference observed in compute Act and expected value " << diff * 100 / spec::kNumVectorLanes <<  "%" << std::endl;
      }
      if (rva_out.PopNB(rva_out_dest))
      {
        /*std::cout << std::hex << sc_time_stamp()
                  << " Dest got rva data = " << rva_out_dest.data
                  << " (resp_idx=" << std::dec << i << ")" << std::endl;*/

        // Defensive bounds check
        assert(i < dest_vec.size() && "Received more responses than expected.");

        if (rva_out_dest.data != dest_vec[i].data)
        {
          std::ostringstream oss;
          oss << "Mismatch at resp " << i
              << ": expected 0x" << std::hex << dest_vec[i].data
              << " got 0x" << rva_out_dest.data;
          SC_REPORT_ERROR("Dest", oss.str().c_str());
          assert(false);
        }
        /*else
        {
          std::cout << "Dest: Response " << i << " matches expected value." << std::endl;
        }*/
        i++;
        if (i >= dest_vec.size())
        {
              sc_stop();
              return;
        }
      }
      wait();
    }
  }
};

// ---------------- TB ----------------
SC_MODULE(testbench)
{
  SC_HAS_PROCESS(testbench);
  sc_clock clk;
  sc_signal<bool> rst;

  Connections::Combinational<bool> start;
  Connections::Combinational<spec::StreamType> input_port;
  Connections::Combinational<spec::Axi::SubordinateToRVA::Write> rva_in;
  Connections::Combinational<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::Combinational<spec::ActVectorType> act_port;
  sc_signal<NVUINT32> SC_SRAM_CONFIG;

  NVHLS_DESIGN(PECore)
  dut;
  Source source;
  Dest dest;

  testbench(sc_module_name name)
      : sc_module(name),
        clk("clk", 1.0, SC_NS, 0.5, 0, SC_NS, true),
        rst("rst"),
        dut("dut"),
        source("source"),
        dest("dest")
  {

    // DUT binds
    dut.clk(clk);
    dut.rst(rst);
    dut.start(start);
    dut.input_port(input_port);
    dut.rva_in(rva_in);
    dut.rva_out(rva_out);
    dut.act_port(act_port);
    dut.SC_SRAM_CONFIG(SC_SRAM_CONFIG);

    // SRC binds
    source.clk(clk);
    source.rst(rst);
    source.start(start);
    source.input_port(input_port);
    source.rva_in(rva_in);

    // DEST binds
    dest.clk(clk);
    dest.rst(rst);
    dest.rva_out(rva_out);
    dest.act_port(act_port);

    testset();

    SC_THREAD(run);
  }

  void testset()
  {
    spec::Axi::SubordinateToRVA::Write rva_write_tmp;
    spec::Axi::SubordinateToRVA::Read rva_read_tmp;

    // Keep local copies of what we write so we can expect readbacks
    NVUINTW(spec::VectorType::width) peconfig_written = 0;
    NVUINTW(spec::VectorType::width) manager1_cfg_written = 0;
    NVUINTW(spec::VectorType::width) weight_written[spec::kNumVectorLanes];
    NVUINTW(spec::VectorType::width) input_written = 0;

    // ---------------------------
    // 1) WRITE PEConfig (region 0x4, local_index = 0x0001)
    // ---------------------------
    rva_write_tmp.rw = 1;
    rva_write_tmp.data = set_bytes<8>("00_00_01_01_00_00_00_01");
    rva_write_tmp.addr = set_bytes<3>("40_00_10"); // correct local_index
    peconfig_written = rva_write_tmp.data;
    source.src_vec.push_back(rva_write_tmp);

    // ---------------------------
    // 2) WRITE WEIGHT SRAM (region 0x5, addr 0x0010)
    // ---------------------------
    for (int i = 0; i < spec::kNumVectorLanes; i++)
    {
      rva_write_tmp.rw = 1;
      rva_write_tmp.data = nvhls::get_rand<spec::VectorType::width>();
      rva_write_tmp.addr = (0x5u << 20) | (static_cast<uint32_t>(i) << 4);
      source.src_vec.push_back(rva_write_tmp);
      weight_written[i] = rva_write_tmp.data; // accumulate for expected readback
    }

    // ---------------------------
    // 3) WRITE INPUT SRAM (region 0x6, addr 0x0020)
    // ---------------------------
    rva_write_tmp.rw = 1;
    rva_write_tmp.data = nvhls::get_rand<spec::VectorType::width>();
    rva_write_tmp.addr = set_bytes<3>("60_00_00");
    input_written = rva_write_tmp.data;
    source.src_vec.push_back(rva_write_tmp);

    // ---------------------------
    // 4) WRITE Manager1 config (region 0x4, local_index = 0x0004)
    // ---------------------------
    rva_write_tmp.rw = 1;
    rva_write_tmp.data = set_bytes<8>("00_00_00_00_00_00_01_00");
    rva_write_tmp.addr = set_bytes<3>("40_00_20");
    manager1_cfg_written = rva_write_tmp.data;
    source.src_vec.push_back(rva_write_tmp);

    // Expected Act Vector
    spec::ActVectorType act_vector;
    spec::AccumScalarType accum ;
    for (int i = 0; i < spec::kNumVectorLanes; i++)
    {
      accum = 0;
      for (int j = 0; j < spec::kVectorSize; j++)
      {
        spec::ScalarType weight = (weight_written[i] >> spec::kIntWordWidth*j) & ((1 << spec::kIntWordWidth) - 1);
        spec::ScalarType input = (input_written >> spec::kIntWordWidth*j) & ((1 << spec::kIntWordWidth) - 1);
        accum += weight * input;
        //cout << "Weight = " << weight << " Input = " << input << " Partial sum = " << accum << std::endl;
      }
      act_vector[i] = int(double(accum) /  12.25); // 12.25 is the scale factor
    }
    dest.act_vector = act_vector;

    // ---- NOTE ----
    // We DO NOT push expected responses for writes.

    // ---------------------------
    // Now issue READs and push expected responses in the same order.
    // ---------------------------

    // A) READ PEConfig (0x4:0x0001)
    rva_write_tmp.rw = 0;
    rva_write_tmp.addr = set_bytes<3>("40_00_10");
    source.src_vec.push_back(rva_write_tmp);
    rva_read_tmp.data = peconfig_written;
    dest.dest_vec.push_back(rva_read_tmp);

    // B) READ WEIGHT SRAM (0x5:0x0010)
    for (int i = 0; i < spec::kNumVectorLanes; i++)
    {
      rva_write_tmp.rw = 0;
      rva_write_tmp.addr = (0x5u << 20) | (static_cast<uint32_t>(i) << 4);
      source.src_vec.push_back(rva_write_tmp);
      rva_read_tmp.data = weight_written[i];
      dest.dest_vec.push_back(rva_read_tmp);
    }

    // C) READ INPUT SRAM (0x6:0x0020)
    rva_write_tmp.rw = 0;
    rva_write_tmp.addr = set_bytes<3>("60_00_00");
    source.src_vec.push_back(rva_write_tmp);
    rva_read_tmp.data = input_written;
    dest.dest_vec.push_back(rva_read_tmp);

    // D) READ Manager1 config (0x4:0x0004)
    rva_write_tmp.rw = 0;
    rva_write_tmp.addr = set_bytes<3>("40_00_20");
    source.src_vec.push_back(rva_write_tmp);
    rva_read_tmp.data = manager1_cfg_written;
    dest.dest_vec.push_back(rva_read_tmp);
  }

  void run()
  {
    wait(2, SC_NS);
    std::cout << "@" << sc_time_stamp() << " Asserting reset" << std::endl;
    rst.write(false);
    wait(2, SC_NS);
    rst.write(true);
    std::cout << "@" << sc_time_stamp() << " De-Asserting reset" << std::endl;

    // Plenty of time for AXI and scratchpad latencies
    wait(10000, SC_NS);
    std::cout << "@" << sc_time_stamp() << " sc_stop" << std::endl;
    sc_stop();
  }
};

int sc_main(int argc, char *argv[])
{
  nvhls::set_random_seed();

  testbench tb("tb");

  sc_report_handler::set_actions(SC_ERROR, SC_DISPLAY);

  sc_start();

  bool rc = (sc_report_handler::get_count(SC_ERROR) > 0);

  if (rc)
    DCOUT("TESTBENCH FAIL" << endl);
  else
    DCOUT("TESTBENCH PASS" << endl);
  return rc;
}
