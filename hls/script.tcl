############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project embedded
set_top simple_fcn_accel
add_files ../../../../Downloads/newhls/newhls/top.cpp
add_files -tb ../../../../Downloads/newhls/newhls/testbench.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution2" -flow_target vivado
set_part {xc7z020-clg400-1}
create_clock -period 10 -name default
source "./embedded/solution2/directives.tcl"
csim_design -clean
csynth_design
cosim_design
export_design -format ip_catalog
