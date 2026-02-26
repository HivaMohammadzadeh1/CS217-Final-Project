// ============================================================================
// Amazon FPGA Hardware Development Kit
// ============================================================================

`include "common_base_test.svh"
`include "design_top_defines.vh"

module design_top_base_test();
import tb_type_defines_pkg::*;

  function automatic real rabs(input real x);
    return (x < 0.0) ? -x : x;
  endfunction

  function automatic int round(input real x);
    return (x > 0.0) ? $floor(x + 0.5) : $ceil(x - 0.5);
  endfunction

  task automatic start_data_transfer_counter();
    logic [WIDTH_ADDR_AXI - 1 : 0] addr;
    logic [WIDTH_DATA_AXI - 1:0] data;
    addr = ADDR_TRANSFER_COUNTER_EN;
    data = 32'h1;
    ocl_wr32(addr, data);
  endtask

  task automatic stop_data_transfer_counter();
    logic [WIDTH_ADDR_AXI - 1 : 0] addr;
    logic [WIDTH_DATA_AXI - 1:0] data;
    addr = ADDR_TRANSFER_COUNTER_EN;
    data = 32'h0;
    ocl_wr32(addr, data);
  endtask

  task automatic get_data_transfer_cycles();
    logic [WIDTH_ADDR_AXI - 1 : 0] addr;
    logic [WIDTH_DATA_AXI - 1:0] data;
    addr = ADDR_TRANSFER_COUNTER;
    ocl_rd32(addr, data);
    $display("Data transfer cycles: %0d", data);
  endtask

  task automatic get_compute_cycles();
    logic [WIDTH_ADDR_AXI - 1 : 0] addr;
    logic [WIDTH_DATA_AXI - 1:0] data;
    addr = ADDR_COMPUTE_COUNTER;
    ocl_rd32(addr, data);
    $display("Compute cycles: %0d", data);
  endtask


  task automatic compare_act_vectors(
    input [kActWordWidth- 1 : 0] dut_vec   [0:kNumVectorLanes - 1],
    input [kActWordWidth- 1 : 0] goldenVec [0:kNumVectorLanes - 1],
    inout logic [WIDTH_DATA_AXI - 1:0] error_cnt
  );
    real diff_sum = 0.0;
    real avg_pct;

    for (int j = 0; j < kNumVectorLanes; j++) begin
      // cast lane values to int, then to real
      int  act_i = int'($signed(dut_vec[j]));
      int  exp_i = int'($signed(goldenVec[j]));
      real a     = $itor(act_i);
      real e     = $itor(exp_i);

      // per-lane relative error with divide-by-zero guard
      real term;
      if (e == 0.0)
        term = (a == 0.0) ? 0.0 : 1.0; // 0% if both zero, else 100%
      else
        term = rabs(a - e) / rabs(e);

      diff_sum += term;

      $display("ActPort Computed value = %0d and expected value = %0d (lane %0d)  err=%0.3f%%",
              act_i, exp_i, j, 100.0*term);
    end

    avg_pct = (diff_sum * 100.0) / kNumVectorLanes;
    $display("Dest: Difference observed in compute Act and expected value %0.3f%%", avg_pct);
    if (avg_pct > 2.0) begin
      $error("ERROR: Average percentage difference %0.3f%% exceeds threshold", avg_pct);
      error_cnt++;
    end
    else begin
      $display("PASS: Average percentage difference %0.3f%% within threshold", avg_pct);
    end
  endtask


  task automatic ocl_wr32(input logic [WIDTH_ADDR_AXI - 1 : 0] addr, input logic [WIDTH_DATA_AXI - 1:0] data);
    tb.poke_ocl(.addr(addr), .data(data));
    //$display("OCL Write 32-bits: addr=0x%04h data=0x%08x", addr, data);
  endtask

  task automatic ocl_rva_wr32(input logic [WIDTH_RVA_IN_32 - 1 : 0] data);
  logic [WIDTH_DATA_AXI - 1:0] w;
  logic [WIDTH_ADDR_AXI - 1 : 0] addr;
    for (int i = 0; i < LOOP_RVA_IN; i++)
      begin
        addr = ADDR_RVA_IN_START + i*4; // 0x400, 0x404, 0x408, ...
        w = data[i*WIDTH_DATA_AXI +: WIDTH_DATA_AXI];
        ocl_wr32(addr, w);
        #1ns;
        //$display("OCL RVA Write 32-bits: addr=0x%04h data=0x%08x", addr, w);
      end
  endtask

  task automatic randomize(output logic [WIDTH_DATA_RVA_IN - 1 : 0] randomised_data);
  for (int i = 0; i < LOOP_RANDOMIZE; i++)
      randomised_data[i*WIDTH_DATA_AXI +: WIDTH_DATA_AXI] = $urandom();
  endtask

  task automatic rva_format(input logic rw,
                             input logic [WIDTH_ADDR_RVA_IN - 1 : 0] addr,
                             input logic [WIDTH_DATA_RVA_IN - 1 : 0] data,
                             output logic [WIDTH_RVA_IN_32 - 1 : 0] rva_msg);
    rva_msg = 0;
    rva_msg[WIDTH_RVA_IN_32 - 1]   = 1; 
    rva_msg[WIDTH_RVA_IN - 1]  = rw; 
    rva_msg[WIDTH_ADDR_RVA_IN + WIDTH_DATA_RVA_IN - 1:0] = {addr, data};
  endtask

  task automatic ocl_rva_r32(inout logic [WIDTH_DATA_AXI - 1:0] error_cnt, input logic [WIDTH_DATA_RVA_IN - 1 : 0] data_cmp, input logic [WIDTH_RVA_IN_32 - 1 : 0] rva_in);
  logic [WIDTH_DATA_AXI - 1:0] r, w;
  logic [WIDTH_ADDR_AXI - 1 : 0] addr;
  logic [WIDTH_RVA_IN_32 - 1 : 0] data;

  ocl_rva_wr32(rva_in);
  #100ns; // Wait for PECore to process and respond

    for (int i = 0; i < LOOP_RVA_OUT; i++)
      begin
        addr = ADDR_RVA_OUT_START + i*4;
        ocl_rd32(addr, r);
        data[i*32 +: 32] = r;
        #1ns;
      end
    if (data[WIDTH_DATA_RVA_IN - 1 : 0] !== data_cmp[WIDTH_DATA_RVA_IN - 1 : 0]) begin
      $error("RVA readback mismatch: exp=0x%032x got=0x%032x", data_cmp[WIDTH_DATA_RVA_IN - 1 : 0], data[WIDTH_DATA_RVA_IN - 1 : 0]);
      error_cnt++;
    end
    else
      $display("RVA readback OK: 0x%032x", data[WIDTH_DATA_RVA_IN - 1 : 0]);
  endtask

  task automatic ocl_rd32(input logic [WIDTH_ADDR_AXI - 1 : 0] addr, output logic [WIDTH_DATA_AXI - 1:0] data);
    tb.peek_ocl(.addr(addr), .data(data));
    //$display("OCL Read 32-bits: addr=0x%04h data=0x%08x", addr, data);
  endtask

  logic [WIDTH_DATA_AXI - 1:0] w;
  logic [WIDTH_ADDR_AXI - 1 : 0] addr_w;
  logic [WIDTH_DATA_AXI - 1:0] input_word;
  logic [WIDTH_ADDR_AXI - 1 : 0] addr_in;
  logic [WIDTH_ADDR_AXI - 1 : 0] addr_cfg;
  logic [WIDTH_DATA_AXI - 1:0] cfg_lo; 
  logic [WIDTH_DATA_AXI - 1:0] rd;
  logic [WIDTH_DATA_AXI - 1:0] sramcfg;

  logic rva_in_rw;
  logic [WIDTH_DATA_RVA_IN - 1 : 0] rva_in_data;
  logic [WIDTH_ADDR_RVA_IN - 1 : 0] rva_in_addr;
  logic [WIDTH_RVA_IN_32 - 1 : 0] rva_in, rva_out;

  logic [WIDTH_DATA_AXI - 1:0] error_cnt;

  logic [WIDTH_DATA_RVA_IN - 1:0] weight [0:kNumVectorLanes - 1];
  logic [WIDTH_DATA_RVA_IN - 1:0] input_written;
  logic [kIntWordWidth - 1 : 0] weight_byte, input_byte;
  logic [kAccumWordWidth - 1 : 0] output_accum;
  logic [kActWordWidth - 1 : 0] output_act [0:kNumVectorLanes - 1];
  logic [kActWordWidth - 1 : 0] output_obtained [0:kNumVectorLanes - 1];
  logic [WIDTH_ACT_PORT- 1 : 0] output_obtained_flat;

  real scaled;

  logic [63:0] cfg;


  initial begin
  $dumpfile("waves.vcd");
  $dumpvars(0, tb);
  end

  initial
  begin
    error_cnt = 'b0;
    tb.power_up(.clk_recipe_a(ClockRecipe::A0),
              .clk_recipe_b(ClockRecipe::B0),
              .clk_recipe_c(ClockRecipe::C0));

    // Reset design
    #500ns;

    // ---------------------------
    // 1) WRITE PEConfig (region 0x4, local_index = 0x0001)
    // ---------------------------

    $display("---- STEP 1: WRITE PEConfig");    
    rva_in_addr = 24'h4000_10;
    rva_in_rw = 1;
    cfg = 64'h0000_0101_0000_0001; rva_in_data = 0;
    rva_in_data[63:0] = cfg;
    rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_in);
    start_data_transfer_counter();
    ocl_rva_wr32(rva_in);
    stop_data_transfer_counter();
    // Read back to verify
    rva_in_rw = 0;
    rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_out);
    ocl_rva_r32(error_cnt, rva_in_data, rva_out);

    // ---------------------------
    // 2) WRITE WEIGHT SRAM (region 0x5, addr i<<4 for i in [0..15])
    // Align with step 1: format RVA message and poke via OCL
    // ---------------------------
    $display("---- STEP 2: WRITE WEIGHT SRAM");
    for (int i = 0; i < kNumVectorLanes; i++) begin
      rva_in_rw   = 1'b1;
      randomize(weight[i]);
      rva_in_data = weight[i];
      rva_in_addr = {4'h5, 20'h0} + (i << 4);
      rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_in);
      start_data_transfer_counter();
      ocl_rva_wr32(rva_in);
      stop_data_transfer_counter();
      // Read back to verify
      rva_in_rw = 0;
      rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_out);
      ocl_rva_r32(error_cnt, rva_in_data, rva_out);
    end

    // ---------------------------
    // 3) WRITE INPUT SRAM (region 0x6, addr 0x0020)
    // Align with step 1
    // ---------------------------
    $display("---- STEP 3: WRITE INPUT SRAM");
    rva_in_rw   = 1'b1;
    randomize(input_written);
    rva_in_data = input_written;
    rva_in_addr = {24'h6000_00};
    rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_in);
    start_data_transfer_counter();
    ocl_rva_wr32(rva_in);
    stop_data_transfer_counter();
    // Read back to verify
    rva_in_rw = 0;
    rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_out);
    ocl_rva_r32(error_cnt, rva_in_data, rva_out);

    // ---------------------------
    // 4) WRITE Manager1 config (region 0x4, local_index = 0x0004)
    // Align with step 1
    // ---------------------------
    $display("---- STEP 4: WRITE Manager1 config");
    rva_in_rw   = 1'b1;
    cfg = 64'h0000_0000_0000_0100; rva_in_data = 0;
    rva_in_data[63:0] = cfg;
    rva_in_addr = {24'h4000_20};
    rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_in);
    start_data_transfer_counter();
    ocl_rva_wr32(rva_in);
    stop_data_transfer_counter();
    // Read back to verify
    rva_in_rw = 0;
    rva_format(rva_in_rw, rva_in_addr, rva_in_data, rva_out);
    ocl_rva_r32(error_cnt, rva_in_data, rva_out);

    // Stop data transfer counter
    stop_data_transfer_counter();

    // ---------------------------
    // 5) start
    // ---------------------------
    $display("---- STEP 5: START");
    addr_w = ADDR_START_CFG;
    input_word = 32'h1;
    ocl_wr32(addr_w, input_word);

    // ---------------------------
    // 5) stop
    // ---------------------------
    addr_w = ADDR_START_CFG;
    input_word = 32'h0;
    ocl_wr32(addr_w, input_word);
    #500ns;

    // ---------------------------
    // 5) Read Output Act
    // ---------------------------
    start_data_transfer_counter();
    $display("---- STEP 6: READ OUTPUT ACT");
    for (int i = 0; i < LOOP_ACT_PORT; i++)
    begin
      addr_w = ADDR_ACT_PORT_START + i*4;
      ocl_rd32(addr_w, rd);
      output_obtained_flat[i*WIDTH_DATA_AXI +: WIDTH_DATA_AXI] = rd;
      #1ns;
    end
    stop_data_transfer_counter();

    for (int i = 0; i < kNumVectorLanes; i++)
    begin
      output_accum = 0;
      for (int j = 0; j < kVectorSize; j++)
      begin
        weight_byte = (weight[i] >> kIntWordWidth*j) & ((1 << kIntWordWidth) - 1);
        input_byte = (input_written >> kIntWordWidth*j) & ((1 << kIntWordWidth) - 1);
        output_accum += weight_byte * input_byte;
      end
      scaled = output_accum / 12.25;
      if (scaled > kActWordMax)
        scaled = kActWordMax;
      else if (scaled < kActWordMin)
        scaled = kActWordMin;
      output_act[i] = round(scaled);
      output_obtained[i] = output_obtained_flat[i*kActWordWidth +: kActWordWidth];
    end

    compare_act_vectors(output_obtained, output_act, error_cnt);

    get_data_transfer_cycles();
    get_compute_cycles();
    
 
    // Power down & report
    #500ns;
    tb.power_down();
    if (error_cnt == 0)
      report_pass_fail_status(1);
    else
      report_pass_fail_status(0);
    
    $finish;
  end
endmodule
