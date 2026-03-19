// SCVerify DUT wrapper used for SystemC > HDL interface bindings


module ccs_wrapper (
  clk, rst, start_vld, start_rdy, start_dat, input_port_vld, input_port_rdy, input_port_dat, rva_in_vld, rva_in_rdy,
      rva_in_dat, rva_out_vld, rva_out_rdy, rva_out_dat, act_port_vld, act_port_rdy, act_port_dat, SC_SRAM_CONFIG
);
  input clk;
  input rst;
  input start_vld;
  output start_rdy;
  input start_dat;
  input input_port_vld;
  output input_port_rdy;
  input [137:0] input_port_dat;
  input rva_in_vld;
  output rva_in_rdy;
  input [168:0] rva_in_dat;
  output rva_out_vld;
  input rva_out_rdy;
  output [127:0] rva_out_dat;
  output act_port_vld;
  input act_port_rdy;
  output [511:0] act_port_dat;
  input [31:0] SC_SRAM_CONFIG;


  PECore dut_inst (
    .clk(clk),
    .rst(rst),
    .start_vld(start_vld),
    .start_rdy(start_rdy),
    .start_dat(start_dat),
    .input_port_vld(input_port_vld),
    .input_port_rdy(input_port_rdy),
    .input_port_dat(input_port_dat),
    .rva_in_vld(rva_in_vld),
    .rva_in_rdy(rva_in_rdy),
    .rva_in_dat(rva_in_dat),
    .rva_out_vld(rva_out_vld),
    .rva_out_rdy(rva_out_rdy),
    .rva_out_dat(rva_out_dat),
    .act_port_vld(act_port_vld),
    .act_port_rdy(act_port_rdy),
    .act_port_dat(act_port_dat),
    .SC_SRAM_CONFIG(SC_SRAM_CONFIG)
  );

endmodule

