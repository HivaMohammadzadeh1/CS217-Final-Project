# written for flow package Vivado 
set sdc_version 1.7 

create_clock -name clk -period 6.0 -waveform { 0.0 3.0 } [get_ports {clk}]
set_clock_uncertainty 0.0 [get_clocks {clk}]

create_clock -name virtual_io_clk -period 6.0
## IO TIMING CONSTRAINTS
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {start_vld}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {start_rdy}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {start_dat}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {input_port_vld}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {input_port_rdy}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {input_port_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {rva_in_vld}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {rva_in_rdy}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {rva_in_dat[*]}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {rva_out_vld}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {rva_out_rdy}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {rva_out_dat[*]}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {act_port_vld}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {act_port_rdy}]
set_output_delay -clock [get_clocks {clk}] 0.0 [get_ports {act_port_dat[*]}]
set_input_delay -clock [get_clocks {clk}] 0.0 [get_ports {SC_SRAM_CONFIG[*]}]

