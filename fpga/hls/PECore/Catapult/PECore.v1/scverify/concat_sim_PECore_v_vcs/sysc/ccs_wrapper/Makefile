SYSC_MODULE = ccs_wrapper
V = @
CXX = g++
CXXFLAGS =  -DVCSSYSTEMC=1  -MMD -m64 -fPIC
INCLUDES = \
	-I/cad/synopsys/vcs/U-2023.03/include/systemc233 \
	-I/cad/synopsys/vcs/U-2023.03/include/systemc233 \
	-I$(VCS_HOME)/include \
	-I$(VCS_HOME)/lib \
	-I/home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1

OBJS = ccs_wrapper.o
all:	$(OBJS)
ccs_wrapper.o:	ccs_wrapper.cpp
	$(V) $(CXX) -c $< -o $@ $(INCLUDES) $(CXXFLAGS)

-include ccs_wrapper.d

