CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic

TARGET := mx_datapath_tb
SRC := testbench.cpp

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC) mx_types.h group_scaler.h mxfp_pe.h mx_datapath.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

