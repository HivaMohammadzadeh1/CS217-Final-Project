
import subprocess
import os
import datetime
import shutil
import re

SPEC_PATH = "design/design_top_defines.vh"
FILELIST_PATH = "verif/scripts/top.xsim.f"

def update_spec(width, vec_size, lanes):

    file_path = SPEC_PATH

    with open(file_path, 'r') as f:
        content = f.read()

    content = re.sub(r'(localparam int kIntWordWidth = )\d+;', rf'\g<1>{width};', content)
    content = re.sub(r'(localparam int kVectorSize = )\d+;', rf'\g<1>{vec_size};', content)
    content = re.sub(r'(localparam int kNumVectorLanes = )\d+;', rf'\g<1>{lanes};', content)

    with open(file_path, 'w') as f:
        f.write(content)
    
    file_path = FILELIST_PATH

    src_path = f"design/concat_PECore/kIntWordWidth_{width}_kVectorSize_{vec_size}_kNumVectorLanes_{lanes}/concat_PECore.v"
    dest_path = f"design/concat_PECore.v"
    shutil.copy(src_path, dest_path)

kIntWordWidth = [8, 16]
kVectorSize = [8, 16]
kNumVectorLanes = [8, 16]

default_kIntWordWidth = 8
default_kVectorSize = 16
default_kNumVectorLanes = 16

log = "Start of the hardware simulation log"
cmd = " cd $CL_DIR/verif/scripts; make ${CL_DESIGN_NAME}_base_test; cd $CL_DIR"
summary_table = []

for i in kIntWordWidth:
    for j in kVectorSize:
        for k in kNumVectorLanes:
            update_spec(i, j, k)
            log_info = f"Running kIntWordWidth: {i}, kVectorSize: {j}, kNumVectorLanes: {k}"
            log = log + "\n" + log_info
            print(log_info)
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
            
            passed = False
            data_cycles = "N/A"
            compute_cycles = "N/A"
            diff_percentage = "N/A"
            
            # Filter and print only the relevant lines from the output
            for line in result.stdout.splitlines():
                if (line.startswith("Dest: Difference observed") or
                    line.startswith("Data transfer cycles:") or
                    line.startswith("Compute cycles:") or
                    "TEST PASSED" in line):
                    print(line)
                
                if "TEST PASSED" in line:
                    passed = True
                if line.startswith("Data transfer cycles:"):
                    data_cycles = line.split(":")[1].strip()
                if line.startswith("Compute cycles:"):
                    compute_cycles = line.split(":")[1].strip()
                if line.startswith("Dest: Difference observed"):
                    diff_percentage = line.split()[-1]
            
            log = log + result.stdout
            summary_table.append((i, j, k, "PASSED" if passed else "FAILED", data_cycles, compute_cycles, diff_percentage))

summary_str = "\nSummary Table:\n"
summary_str += f"{'Width':<10} {'Vec Size':<10} {'Lanes':<10} {'Status':<10} {'Data Cycles':<15} {'Comp Cycles':<15} {'Diff %':<10}\n"
summary_str += "-" * 90 + "\n"
for row in summary_table:
    summary_str += f"{row[0]:<10} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<15} {row[5]:<15} {row[6]:<10}\n"

print(summary_str)
log += summary_str

LOGS_FOLDER = "logs"
if not os.path.exists(LOGS_FOLDER):
    os.makedirs(LOGS_FOLDER)

log_filename = "logs/f2_hw_sim.log.txt"

with open(log_filename, "w") as log_file:
    log_file.write(log)

update_spec(default_kIntWordWidth, default_kVectorSize, default_kNumVectorLanes)