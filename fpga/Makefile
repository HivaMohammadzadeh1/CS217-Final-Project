DESIGN_NAME ?= PECore
AWS_DESIGN_NAME ?= design_top

REPO_TOP 	?= $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/)
SRC_HOME 	?= $(REPO_TOP)/src
HLS_HOME 	?= $(SRC_HOME)/hls
AWS_HOME	?= ./$(AWS_DESIGN_NAME)

systemc_sim:
	cd $(SRC_HOME)/$(DESIGN_NAME) && make sim_test && make run

hls_sim:
	cd $(HLS_HOME)/$(DESIGN_NAME) && make hls && make vcs_sim
	cp $(HLS_HOME)/$(DESIGN_NAME)/Catapult/$(DESIGN_NAME).v1/concat_$(DESIGN_NAME).v $(AWS_HOME)/design/

hls_sim_debug:
	cd $(HLS_HOME)/$(DESIGN_NAME) && make vcs_debug
	verdi -ssf $(HLS_HOME)/$(DESIGN_NAME)/default.fsdb -dbdir $(HLS_HOME)/$(DESIGN_NAME)/Catapult/$(DESIGN_NAME).v1/scverify/concat_sim_$(DESIGN_NAME)_v_vcs/sc_main.daidir/

clean:
	cd $(SRC_HOME)/$(DESIGN_NAME) && make clean
	cd $(SRC_HOME)/$(DESIGN_NAME)/Datapath && make clean
	cd $(HLS_HOME)/$(DESIGN_NAME) && make clean
	rm -rf design_top/build/checkpoints/ \
	design_top/build/constraints/generated_cl_clocks_aws.xdc  \
	design_top/build/reports/ \
	design_top/build/src_post_encryption \
	design_top/build/scripts/hd_visual/ \
	design_top/build/scripts/.Xil/ \
	design_top/build/scripts/*.jou \
	design_top/build/scripts/*.vivado.log \
	design_top/build/scripts/*.txt \
	design_top/verif/sim/ \
	design_top/software/runtime/design_top \
	rm -rf ./*~
	rm -rf ./*.key
	rm -rf ./core*
	rm -rf ./Catapult*
	rm -rf ./catapult*
	rm -rf ./*.log
	rm -rf ./design_checker_*.tcl
	rm -rf ./DVE*
	rm -rf ./verdi*
	rm -rf ./slec*	
	rm -rf ./novas*	
	rm -rf ./*.fsdb
	rm -rf ./*.saif*
	rm -rf ./*.vpd 

submission:
	zip -r lab1-submission.zip src/PECore/Datapath/Datapath.h \
	src/PECore/PECore.h \
	src/include/Spec.h \
	design_top/software/src/design_top.c \
	design_top/design/concat_PECore/* \
	logs/area.log.txt \
	logs/rtl_sim.log.txt \
	logs/systemc_sim.log.txt \
	design_top/logs/f2_hw_sim.log.txt \
	design_top/logs/fpga_test.log.txt




