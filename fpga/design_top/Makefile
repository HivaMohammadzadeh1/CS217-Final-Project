CL_DIR 			?= $(CL_DIR)
CL_DESIGN_NAME 	?= $(CL_DESIGN_NAME)
LOGS_DIR 		?= $(CL_DIR)/logs

hw_sim:
	cd $(CL_DIR)/verif/scripts && make ${CL_DESIGN_NAME}_base_test

fpga_build:
	cp $(CL_DIR)/design/concat_PECore/kIntWordWidth_8_kVectorSize_16_kNumVectorLanes_16/concat_PECore.v $(CL_DIR)/design/concat_PECore.v 
	cd $(CL_DIR)/design/ && sed -i '\|// synopsys translate|d' concat_PECore.v 
	cd $(CL_DIR)/build/scripts && ./aws_build_dcp_from_cl.py -c $(CL_DESIGN_NAME)

generate_afi:
	set -e; . ./setup.sh; \
	generate_afi

check_afi_available:
	set -e; . ./generated_afid.sh; \
	aws ec2 describe-fpga-images --fpga-image-ids $$AFI --region $(REGION)

program_fpga:
	sudo fpga-clear-local-image  -S 0
	set -e; . ./generated_afid.sh; \
    sudo fpga-load-local-image -S 0 -I $$AGFI

run_fpga_test:
	cd $(CL_DIR)/software/runtime && make clean && make $(CL_DESIGN_NAME) && sudo ./$(CL_DESIGN_NAME) 0 | tee $(LOGS_DIR)/fpga_test.log.txt 

clean:
	cd $(CL_DIR)/verif/scripts && make clean
	cd $(CL_DIR)/build/ && rm -rf checkpoints/ reports/ src_post_encryption/
	cd $(CL_DIR)/build/ && rm -f constraints/generated_cl_clocks_aws.xdc
	cd $(CL_DIR)/build/scripts && rm -rf hd_visual/ .Xil/
	cd $(CL_DIR)/build/scripts && rm -f *.jou *.vivado.log *.txt
	cd $(CL_DIR)/software/runtime && make clean