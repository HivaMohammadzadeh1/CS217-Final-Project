open_project mx_datapath_hls
set_top mx_datapath_top

add_files mx_datapath_hls.cpp
add_files -tb testbench_hls.cpp

open_solution "solution1" -flow_target vivado
set_part {xcvu9p-flgb2104-2-i}
create_clock -period 4

csim_design
csynth_design

exit

