// mc_dut_wrapper.h
#ifndef INCLUDED_CCS_DUT_WRAPPER_H
#define INCLUDED_CCS_DUT_WRAPPER_H

#ifndef SC_USE_STD_STRING
#define SC_USE_STD_STRING
#endif

#include <systemc.h>
#include <mc_simulator_extensions.h>

#ifdef CCS_SYSC
namespace HDL {
#endif

// Create a foreign module wrapper around the HDL
#ifdef VCS_SYSTEMC
// VCS support - ccs_DUT_wrapper is derived from VCS-generated SystemC wrapper around HDL code
class ccs_DUT_wrapper : public ccs_wrapper
{
public:
  ccs_DUT_wrapper(const sc_module_name& nm, const char *hdl_name)
  : ccs_wrapper(nm)
  {}
  ~ccs_DUT_wrapper() {}
};

#else
// non VCS simulators - ccs_DUT_wrapper is derived from mc_foreign_module (adding 2nd ctor arg)
class ccs_DUT_wrapper: public mc_foreign_module
{
public:
  // Interface Ports
  sc_in<bool> clk;
  sc_in<sc_logic> rst;
  sc_in<sc_logic> start_vld;
  sc_out<sc_logic> start_rdy;
  sc_in<sc_logic> start_dat;
  sc_in<sc_logic> input_port_vld;
  sc_out<sc_logic> input_port_rdy;
  sc_in<sc_lv<138> > input_port_dat;
  sc_in<sc_logic> rva_in_vld;
  sc_out<sc_logic> rva_in_rdy;
  sc_in<sc_lv<169> > rva_in_dat;
  sc_out<sc_logic> rva_out_vld;
  sc_in<sc_logic> rva_out_rdy;
  sc_out<sc_lv<128> > rva_out_dat;
  sc_out<sc_logic> act_port_vld;
  sc_in<sc_logic> act_port_rdy;
  sc_out<sc_lv<512> > act_port_dat;
  sc_in<sc_lv<32> > SC_SRAM_CONFIG;
public:
  ccs_DUT_wrapper(const sc_module_name& nm, const char *hdl_name)
  :
    mc_foreign_module(nm, hdl_name), 
    clk("clk"), 
    rst("rst"), 
    start_vld("start_vld"), 
    start_rdy("start_rdy"), 
    start_dat("start_dat"), 
    input_port_vld("input_port_vld"), 
    input_port_rdy("input_port_rdy"), 
    input_port_dat("input_port_dat"), 
    rva_in_vld("rva_in_vld"), 
    rva_in_rdy("rva_in_rdy"), 
    rva_in_dat("rva_in_dat"), 
    rva_out_vld("rva_out_vld"), 
    rva_out_rdy("rva_out_rdy"), 
    rva_out_dat("rva_out_dat"), 
    act_port_vld("act_port_vld"), 
    act_port_rdy("act_port_rdy"), 
    act_port_dat("act_port_dat"), 
    SC_SRAM_CONFIG("SC_SRAM_CONFIG")
  {
    elaborate_foreign_module(hdl_name);
  }

  ~ccs_DUT_wrapper() {}
};
#endif // VCS_SYSTEMC

#ifdef CCS_SYSC
} // namespace HDL
#endif
#endif // INCLUDED_CCS_DUT_WRAPPER_H


