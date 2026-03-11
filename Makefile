PYTHON := $(shell \
	if [ -x .venv/bin/python3 ]; then echo .venv/bin/python3; \
	elif [ -x venv/bin/python3 ]; then echo venv/bin/python3; \
	else echo python3; fi)

.PHONY: help fpga-doctor fpga-ref-sim fpga-systemc-sim fpga-hls-sim fpga-hw-sim fpga-build fpga-program fpga-test py-tests

help:
	@printf '%s\n' \
	"Targets:" \
	"  make fpga-doctor      # show resolved FPGA build environment" \
	"  make fpga-ref-sim     # run portable MX reference simulation" \
	"  make fpga-systemc-sim # run PECore SystemC sim on Stanford build machine" \
	"  make fpga-hls-sim     # run Catapult HLS + RTL sim on Stanford build machine" \
	"  make fpga-hw-sim      # run design_top hardware sim" \
	"  make fpga-build       # build FPGA checkpoint / AFI input" \
	"  make fpga-program     # load image on F2 runtime host" \
	"  make fpga-test        # run runtime binary on F2 runtime host" \
	"  make py-tests         # run repo-local Python tests"

fpga-doctor:
	$(PYTHON) fpga/run_fpga_flow.py doctor

fpga-ref-sim:
	$(PYTHON) fpga/run_fpga_flow.py reference-sim

fpga-systemc-sim:
	$(PYTHON) fpga/run_fpga_flow.py systemc-sim

fpga-hls-sim:
	$(PYTHON) fpga/run_fpga_flow.py hls-sim

fpga-hw-sim:
	$(PYTHON) fpga/run_fpga_flow.py hw-sim

fpga-build:
	$(PYTHON) fpga/run_fpga_flow.py fpga-build

fpga-program:
	$(PYTHON) fpga/run_fpga_flow.py program-fpga

fpga-test:
	$(PYTHON) fpga/run_fpga_flow.py run-fpga-test --slot-id 0

py-tests:
	$(PYTHON) -m unittest integration.test_mx_offload_integration fpga.test_run_fpga_flow -v
