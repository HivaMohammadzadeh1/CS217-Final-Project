ACTIVE = @

-include /home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/scverify/concat_sim_PECore_v_vcs/sysc/subMakefile

BUILD_SYSC::
-include /home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/scverify/concat_sim_PECore_v_vcs/sysc/info.sysc.make.t

-include /home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/scverify/concat_sim_PECore_v_vcs/sysc/info.sysc.make.shared

-include /home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/scverify/concat_sim_PECore_v_vcs/sysc/modellist

-include /home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/scverify/concat_sim_PECore_v_vcs/sysc/stublist

#FORCE TARGET
FORCE:
SYSC_FILES += $(addprefix /home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/scverify/concat_sim_PECore_v_vcs/sysc/,$(MODELS))

SYSC_STUBS += $(addprefix /home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/scverify/concat_sim_PECore_v_vcs/sysc/,$(STUBLIB_concat_sim_PECore_v_vcs))
SYSC_STUB_LIB := $(SYSC_STUBS)

SYSC_OBJS := $(SYSC_OBJS) $(addsuffix .o,$(SYSC_FILES))
SYSC_LIBS := $(SYSC_LIBS) $(SYSC_STUB_LIB)
sysc_globals_cpp = /home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/scverify/concat_sim_PECore_v_vcs/sysc/sysc_globals.cpp
sysc_globals_o = /home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/scverify/concat_sim_PECore_v_vcs/sysc/sysc_globals.o
$(sysc_globals_o):	$(sysc_globals_cpp)
	@g++ -o $@ -c $^  -m64 -fPIC   -pipe -O -I/cad/synopsys/vcs/U-2023.03/include  
$(SYSC_STUB_LIB):
	$(MAKE) -C /home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/scverify/concat_sim_PECore_v_vcs/sysc -f Makefile_shared stubs

systemc_stubs_c = /home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/scverify/concat_sim_PECore_v_vcs/sysc/systemc_stub.c
systemc_stubs_o = /home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/scverify/concat_sim_PECore_v_vcs/sysc/systemc_stub.o
$(systemc_stubs_o):	$(systemc_stubs_c)
	@$(VCS_CC)  -o $@ -c $^  -m64 -fPIC 
