// Created during runtime of your simulation.


//  Verilog wrapper module for SystemC instance 'sc_main.tb.comb_ba_0'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::Combinational<bool,...> ;


endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.dummyPortManager_0'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::Combinational<bool,...>::DummyPortManager ;


endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.comb_ba_1'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::Combinational<spec::StreamType,...> ;


endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.dummyPortManager_1'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::Combinational<spec::StreamType,...>::DummyPortManager ;


endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.comb_ba_2'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::Combinational<RVSink<spec::Axi::rvaCfg>::Write,...> ;


endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.dummyPortManager_2'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::Combinational<RVSink<spec::Axi::rvaCfg>::Write,...>::DummyPortManager ;


endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.comb_ba_3'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::Combinational<RVSink<spec::Axi::rvaCfg>::Read,...> ;


endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.dummyPortManager_3'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::Combinational<RVSink<spec::Axi::rvaCfg>::Read,...>::DummyPortManager ;


endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.comb_ba_4'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::Combinational<nvhls::nv_scvector<ac_int<32,...>,...>,...> ;


endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.dummyPortManager_4'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::Combinational<nvhls::nv_scvector<ac_int<32,...>,...>,...>::DummyPortManager ;


endmodule


//  Stop traversal at instance 'sc_main.tb.dut.ccs_rtl' with bf_type=BF_SC_DKI_MODULE_T

//  Verilog wrapper module for SystemC instance 'sc_main.tb.dut'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$CCS_RTL::sysc_sim_wrapper ;

  //  instance:  ccs_rtl (BF_SC_DKI_MODULE_T)
  // Instance type = Verilog
  reg \ccs_rtl_R_clk ;
  reg \ccs_rtl_R_rst ;
  reg \ccs_rtl_R_start_vld ;
  reg \ccs_rtl_R_start_dat ;
  reg \ccs_rtl_R_input_port_vld ;
  reg [137:0] \ccs_rtl_R_input_port_dat ;
  reg \ccs_rtl_R_rva_in_vld ;
  reg [168:0] \ccs_rtl_R_rva_in_dat ;
  reg \ccs_rtl_R_rva_out_rdy ;
  reg \ccs_rtl_R_act_port_rdy ;
  reg [31:0] \ccs_rtl_R_SC_SRAM_CONFIG ;

  ccs_wrapper  ccs_rtl (
     .clk(\ccs_rtl_R_clk ),
     .rst(\ccs_rtl_R_rst ),
     .start_vld(\ccs_rtl_R_start_vld ),
     .start_dat(\ccs_rtl_R_start_dat ),
     .input_port_vld(\ccs_rtl_R_input_port_vld ),
     .input_port_dat(\ccs_rtl_R_input_port_dat ),
     .rva_in_vld(\ccs_rtl_R_rva_in_vld ),
     .rva_in_dat(\ccs_rtl_R_rva_in_dat ),
     .rva_out_rdy(\ccs_rtl_R_rva_out_rdy ),
     .act_port_rdy(\ccs_rtl_R_act_port_rdy ),
     .SC_SRAM_CONFIG(\ccs_rtl_R_SC_SRAM_CONFIG )
  );


  //  signal:  ccs_rtl_SIG_clk
  //  signal:  ccs_rtl_SIG_rst
  //  signal:  ccs_rtl_SIG_start_vld
  //  signal:  ccs_rtl_SIG_start_rdy
  //  signal:  ccs_rtl_SIG_start_dat
  //  signal:  ccs_rtl_SIG_input_port_vld
  //  signal:  ccs_rtl_SIG_input_port_rdy
  //  signal:  ccs_rtl_SIG_input_port_dat
  //  signal:  ccs_rtl_SIG_rva_in_vld
  //  signal:  ccs_rtl_SIG_rva_in_rdy
  //  signal:  ccs_rtl_SIG_rva_in_dat
  //  signal:  ccs_rtl_SIG_rva_out_vld
  //  signal:  ccs_rtl_SIG_rva_out_rdy
  //  signal:  ccs_rtl_SIG_rva_out_dat
  //  signal:  ccs_rtl_SIG_act_port_vld
  //  signal:  ccs_rtl_SIG_act_port_rdy
  //  signal:  ccs_rtl_SIG_act_port_dat
  //  signal:  ccs_rtl_SIG_SC_SRAM_CONFIG
  //  signal:  SIG_SC_LOGIC_0
  //  signal:  SIG_SC_LOGIC_1
  //  port:  clk
  //  port:  rst
  //  port:  start_vld
  //  port:  start_rdy
  //  port:  start_dat
  //  port:  input_port_vld
  //  port:  input_port_rdy
  //  port:  input_port_dat
  //  port:  rva_in_vld
  //  port:  rva_in_rdy
  //  port:  rva_in_dat
  //  port:  rva_out_vld
  //  port:  rva_out_rdy
  //  port:  rva_out_dat
  //  port:  act_port_vld
  //  port:  act_port_rdy
  //  port:  act_port_dat
  //  port:  SC_SRAM_CONFIG
  //  process:  update_proc

endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.source'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module sysc$Source;

  //  port:  port_0
  //  port:  port_1
  //  port:  out_vld_0
  //  port:  out_rdy_0
  //  port:  out_dat_0
  //  port:  out_vld_1
  //  port:  out_rdy_1
  //  port:  out_dat_1
  //  port:  out_vld_2
  //  port:  out_rdy_2
  //  port:  out_dat_2
  //  process:  run

endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.dest'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module sysc$Dest;

  //  port:  port_0
  //  port:  port_1
  //  port:  in_vld_0
  //  port:  in_rdy_0
  //  port:  in_dat_0
  //  port:  in_vld_1
  //  port:  in_rdy_1
  //  port:  in_dat_1
  //  process:  run

endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.dynamic_d2mport_0'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::DirectToMarshalledInPort<bool> ;

  //  signal:  signal_0
  //  port:  port_0
  //  process:  do_direct2marshalled

endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.dynamic_d2mport_1'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::DirectToMarshalledInPort<spec::StreamType> ;

  //  signal:  signal_0
  //  port:  port_0
  //  process:  do_direct2marshalled

endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.dynamic_d2mport_2'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::DirectToMarshalledInPort<RVSink<spec::Axi::rvaCfg>::Write> ;

  //  signal:  signal_0
  //  port:  port_0
  //  process:  do_direct2marshalled

endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.dynamic_m2dport_0'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::MarshalledToDirectOutPort<RVSink<spec::Axi::rvaCfg>::Read> ;

  //  signal:  signal_0
  //  port:  port_0
  //  port:  port_1
  //  process:  do_marshalled2direct

endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb.dynamic_m2dport_1'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module \sysc$Connections::MarshalledToDirectOutPort<nvhls::nv_scvector<ac_int<32,...>,...>> ;

  //  signal:  signal_0
  //  port:  port_0
  //  port:  port_1
  //  process:  do_marshalled2direct

endmodule


//  Verilog wrapper module for SystemC instance 'sc_main.tb'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module sysc$testbench;

  //  instance:  comb_ba_0 (BF_MODULE_T)
  \sysc$Connections::Combinational<bool,...>  comb_ba_0();

  //  instance:  dummyPortManager_0 (BF_MODULE_T)
  \sysc$Connections::Combinational<bool,...>::DummyPortManager  dummyPortManager_0();

  //  instance:  comb_ba_1 (BF_MODULE_T)
  \sysc$Connections::Combinational<spec::StreamType,...>  comb_ba_1();

  //  instance:  dummyPortManager_1 (BF_MODULE_T)
  \sysc$Connections::Combinational<spec::StreamType,...>::DummyPortManager  dummyPortManager_1();

  //  instance:  comb_ba_2 (BF_MODULE_T)
  \sysc$Connections::Combinational<RVSink<spec::Axi::rvaCfg>::Write,...>  comb_ba_2();

  //  instance:  dummyPortManager_2 (BF_MODULE_T)
  \sysc$Connections::Combinational<RVSink<spec::Axi::rvaCfg>::Write,...>::DummyPortManager  dummyPortManager_2();

  //  instance:  comb_ba_3 (BF_MODULE_T)
  \sysc$Connections::Combinational<RVSink<spec::Axi::rvaCfg>::Read,...>  comb_ba_3();

  //  instance:  dummyPortManager_3 (BF_MODULE_T)
  \sysc$Connections::Combinational<RVSink<spec::Axi::rvaCfg>::Read,...>::DummyPortManager  dummyPortManager_3();

  //  instance:  comb_ba_4 (BF_MODULE_T)
  \sysc$Connections::Combinational<nvhls::nv_scvector<ac_int<32,...>,...>,...>  comb_ba_4();

  //  instance:  dummyPortManager_4 (BF_MODULE_T)
  \sysc$Connections::Combinational<nvhls::nv_scvector<ac_int<32,...>,...>,...>::DummyPortManager  dummyPortManager_4();

  //  instance:  dut (BF_MODULE_T)
  \sysc$CCS_RTL::sysc_sim_wrapper  dut();

  //  instance:  source (BF_MODULE_T)
  sysc$Source source();

  //  instance:  dest (BF_MODULE_T)
  sysc$Dest dest();

  //  instance:  dynamic_d2mport_0 (BF_MODULE_T)
  \sysc$Connections::DirectToMarshalledInPort<bool>  dynamic_d2mport_0();

  //  instance:  dynamic_d2mport_1 (BF_MODULE_T)
  \sysc$Connections::DirectToMarshalledInPort<spec::StreamType>  dynamic_d2mport_1();

  //  instance:  dynamic_d2mport_2 (BF_MODULE_T)
  \sysc$Connections::DirectToMarshalledInPort<RVSink<spec::Axi::rvaCfg>::Write>  dynamic_d2mport_2();

  //  instance:  dynamic_m2dport_0 (BF_MODULE_T)
  \sysc$Connections::MarshalledToDirectOutPort<RVSink<spec::Axi::rvaCfg>::Read>  dynamic_m2dport_0();

  //  instance:  dynamic_m2dport_1 (BF_MODULE_T)
  \sysc$Connections::MarshalledToDirectOutPort<nvhls::nv_scvector<ac_int<32,...>,...>>  dynamic_m2dport_1();


  //  signal:  clk
  //  signal:  rst
  //  signal:  comb_in_vld_0
  //  signal:  comb_in_rdy_0
  //  signal:  comb_out_vld_0
  //  signal:  comb_out_rdy_0
  //  signal:  comb_in_dat_0
  //  signal:  comb_out_dat_0
  //  signal:  comb_in_vld_1
  //  signal:  comb_in_rdy_1
  //  signal:  comb_out_vld_1
  //  signal:  comb_out_rdy_1
  //  signal:  comb_in_dat_1
  //  signal:  comb_out_dat_1
  //  signal:  comb_in_vld_2
  //  signal:  comb_in_rdy_2
  //  signal:  comb_out_vld_2
  //  signal:  comb_out_rdy_2
  //  signal:  comb_in_dat_2
  //  signal:  comb_out_dat_2
  //  signal:  comb_in_vld_3
  //  signal:  comb_in_rdy_3
  //  signal:  comb_out_vld_3
  //  signal:  comb_out_rdy_3
  //  signal:  comb_in_dat_3
  //  signal:  comb_out_dat_3
  //  signal:  comb_in_vld_4
  //  signal:  comb_in_rdy_4
  //  signal:  comb_out_vld_4
  //  signal:  comb_out_rdy_4
  //  signal:  comb_in_dat_4
  //  signal:  comb_out_dat_4
  //  signal:  signal_0
  //  port:  sim_out_0_vld
  //  port:  sim_out_0_rdy
  //  port:  sim_out_0_dat
  //  port:  sim_in_0_vld
  //  port:  sim_in_0_rdy
  //  port:  sim_in_0_dat
  //  port:  sim_out_1_vld
  //  port:  sim_out_1_rdy
  //  port:  sim_out_1_dat
  //  port:  sim_in_1_vld
  //  port:  sim_in_1_rdy
  //  port:  sim_in_1_dat
  //  port:  sim_out_2_vld
  //  port:  sim_out_2_rdy
  //  port:  sim_out_2_dat
  //  port:  sim_in_2_vld
  //  port:  sim_in_2_rdy
  //  port:  sim_in_2_dat
  //  port:  sim_out_3_vld
  //  port:  sim_out_3_rdy
  //  port:  sim_out_3_dat
  //  port:  sim_in_3_vld
  //  port:  sim_in_3_rdy
  //  port:  sim_in_3_dat
  //  port:  sim_out_4_vld
  //  port:  sim_out_4_rdy
  //  port:  sim_out_4_dat
  //  port:  sim_in_4_vld
  //  port:  sim_in_4_rdy
  //  port:  sim_in_4_dat
  //  process:  method_p_0
  //  process:  method_p_1
  //  process:  method_p_2
  //  process:  method_p_3
  //  process:  method_p_4
  //  process:  run

endmodule


//  Verilog wrapper module for SystemC instance 'sc_main'
`noinline
(*vcs_systemc_2,vcs_systemc_skel*) module sc_main;

  //  instance:  tb (BF_MODULE_T)
  sysc$testbench tb();



endmodule


// VCS:  sysc_uni_top_module_name=sc_main
// use sc_main.tb.comb_ba_0 \sysc$Connections::Combinational<bool,...> 
// use sc_main.tb.dummyPortManager_0 \sysc$Connections::Combinational<bool,...>::DummyPortManager 
// use sc_main.tb.comb_ba_1 \sysc$Connections::Combinational<spec::StreamType,...> 
// use sc_main.tb.dummyPortManager_1 \sysc$Connections::Combinational<spec::StreamType,...>::DummyPortManager 
// use sc_main.tb.comb_ba_2 \sysc$Connections::Combinational<RVSink<spec::Axi::rvaCfg>::Write,...> 
// use sc_main.tb.dummyPortManager_2 \sysc$Connections::Combinational<RVSink<spec::Axi::rvaCfg>::Write,...>::DummyPortManager 
// use sc_main.tb.comb_ba_3 \sysc$Connections::Combinational<RVSink<spec::Axi::rvaCfg>::Read,...> 
// use sc_main.tb.dummyPortManager_3 \sysc$Connections::Combinational<RVSink<spec::Axi::rvaCfg>::Read,...>::DummyPortManager 
// use sc_main.tb.comb_ba_4 \sysc$Connections::Combinational<nvhls::nv_scvector<ac_int<32,...>,...>,...> 
// use sc_main.tb.dummyPortManager_4 \sysc$Connections::Combinational<nvhls::nv_scvector<ac_int<32,...>,...>,...>::DummyPortManager 
// use sc_main.tb.dut \sysc$CCS_RTL::sysc_sim_wrapper 
// use sc_main.tb.source sysc$Source
// use sc_main.tb.dest sysc$Dest
// use sc_main.tb.dynamic_d2mport_0 \sysc$Connections::DirectToMarshalledInPort<bool> 
// use sc_main.tb.dynamic_d2mport_1 \sysc$Connections::DirectToMarshalledInPort<spec::StreamType> 
// use sc_main.tb.dynamic_d2mport_2 \sysc$Connections::DirectToMarshalledInPort<RVSink<spec::Axi::rvaCfg>::Write> 
// use sc_main.tb.dynamic_m2dport_0 \sysc$Connections::MarshalledToDirectOutPort<RVSink<spec::Axi::rvaCfg>::Read> 
// use sc_main.tb.dynamic_m2dport_1 \sysc$Connections::MarshalledToDirectOutPort<nvhls::nv_scvector<ac_int<32,...>,...>> 
// use sc_main.tb sysc$testbench
// use sc_main sc_main

