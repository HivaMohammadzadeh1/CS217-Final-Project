import subprocess
import os
import datetime
import shutil
import re
import argparse

UNIT_NAME = "PECore"
SPEC_PATH = "src/include/Spec.h"
SRC_PATH = "src/PECore/"
HLS_PATH = "hls/PECore/"
AWS_PATH = os.getenv("AWS_HOME")

RTL_SIM_LOG = "logs/rtl_sim.log.txt"
SYSTEMC_SIM_LOG = "logs/systemc_sim.log.txt"
AREA_LOG = "logs/area.log.txt"

systemc_sim_cmd = "make clean; make sim_test; make run"
gen_rtl_cmd = "make clean; make hls"
rtl_sim_cmd = "make vcs_sim"

kIntWordWidth = [8, 16]
kVectorSize = [8, 16]
kNumVectorLanes = [8, 16]

default_kIntWordWidth = 8
default_kVectorSize = 16
default_kNumVectorLanes = 16

def get_area():
    area_table = [] 
    for i in kIntWordWidth:
        for j in kVectorSize:
            for k in kNumVectorLanes:
                folder_name = f"design/concat_{UNIT_NAME}/kIntWordWidth_{i}_kVectorSize_{j}_kNumVectorLanes_{k}/{UNIT_NAME}.rpt"
                path = os.path.join(AWS_PATH, folder_name)
                if not os.path.exists(path):
                    print(f"Warning: Report file not found for {path}. Please generate RTL first.")
                    return
                area_score = None
                try:
                    with open(path, 'r') as f:
                        content = f.read()
                        match = re.search(r"Total Area Score:\s+[\d.]+\s+[\d.]+\s+([\d.]+)", content)
                        if match:
                            area_score = float(match.group(1))
                except (IOError, ValueError) as e:
                    print(f"Warning: Could not parse area score for {folder_name}. Reason: {e}")
                area_table.append((i, j, k, area_score))
    area_summary = "\n\n" + "="*70 + "\n"
    area_summary += " " * 25 + "RTL AREA SUMMARY" + " " * 25 + "\n"
    area_summary += "="*70 + "\n"
    area_summary += f"{'kIntWordWidth':<15} | {'kVectorSize':<12} | {'kNumVectorLanes':<15} | {'Area Score':<15}\n"
    area_summary += "-"*80 + "\n"

    for i_val, j_val, k_val, area in area_table:
        area_str = f"{area:.2f}" if area is not None else "N/A"
        row = f"{i_val:<15} | {j_val:<12} | {k_val:<15} | {area_str:<15}\n"
        area_summary += row
    
    area_summary += "="*70 + "\n"
    print(area_summary)
    with open(AREA_LOG, "w") as f:
        f.write(area_summary)
    
                

def copy_concat_rtl(kIntWordWidth, kVectorSize, kNumVectorLanes):
    src_rtl_path = os.path.join(os.getcwd(), "hls/PECore/Catapult/PECore.v1/concat_PECore.v")
    src_rpt_path = os.path.join(os.getcwd(), "hls/PECore/Catapult/PECore.v1/PECore.rpt")
    src_log_path = os.path.join(os.getcwd(), "hls/PECore/catapult.log")

    base_dest_path = os.path.abspath(os.path.join(os.getcwd(), "design_top/design/concat_PECore/"))
    dest_folder = f"kIntWordWidth_{kIntWordWidth}_kVectorSize_{kVectorSize}_kNumVectorLanes_{kNumVectorLanes}"
    full_dest_path = os.path.join(base_dest_path, dest_folder)

    if not os.path.exists(full_dest_path):
        os.makedirs(full_dest_path)
    
    shutil.copy(src_rtl_path, full_dest_path)
    shutil.copy(src_rpt_path, full_dest_path)
    shutil.copy(src_log_path, full_dest_path)
   


def update_spec(width, vec_size, lanes):

    file_path = SPEC_PATH

    with open(file_path, 'r') as f:
        content = f.read()

    content = re.sub(r'(const int kIntWordWidth = )\d+;', rf'\g<1>{width};', content)
    content = re.sub(r'(const int kVectorSize = )\d+;', rf'\g<1>{vec_size};', content)
    content = re.sub(r'(const int kNumVectorLanes = )\d+;', rf'\g<1>{lanes};', content)

    with open(file_path, 'w') as f:
        f.write(content)

def sim(systemc_sim, hls_sim):

    error = 0
    info = []
    log = "Starting System C Simulation Test Suite\n"

    for i in kIntWordWidth:
        for j in kVectorSize:
            for k in kNumVectorLanes:
                update_spec(i, j, k)

                if systemc_sim:
                    cmd = systemc_sim_cmd
                    path = SRC_PATH
                    LOG = SYSTEMC_SIM_LOG
                elif hls_sim:
                    cmd = rtl_sim_cmd
                    path = HLS_PATH
                    LOG = RTL_SIM_LOG

                txt_log = "\nRunning test with kIntWordWidth = " + str(i) + ", kVectorSize = " + str(j) + ", kNumVectorLanes = " + str(k)
                log = log + "\n" + txt_log
                print(txt_log)
                if hls_sim:
                    result = subprocess.run(
                                gen_rtl_cmd,
                                shell=True,
                                capture_output=True,
                                text=True,
                                check=False,
                                cwd=path 
                            )
                result = subprocess.run(
                                cmd,
                                shell=True,
                                capture_output=True,
                                text=True,
                                check=False,
                                cwd=path 
                            )
                if systemc_sim:
                    print(result.stdout)

                if result.returncode:
                    print("ERROR")
                    exit
                else:
                    diff_percentage = "N/A"
                    # Find the difference percentage in the output
                    match = re.search(r"Difference observed in compute Act and expected value (.*)%", result.stdout)
                    if match:
                        diff_percentage = match.group(1)

                    if "TESTBENCH FAIL" in result.stdout:
                        info.append([i, j, k, "FAILED", diff_percentage])
                        print("FAILED: PE output and expected output difference = " + diff_percentage + "%")
                        error = error + 1
                        exit
                    elif "TESTBENCH PASS" in result.stdout:
                        if float(diff_percentage) > 0.2:
                            info.append([i, j, k, "FAILED", diff_percentage])
                            print("FAILED: PE output and expected output difference = " + diff_percentage + "%")
                            error = error + 1
                            exit
                        else: 
                            info.append([i, j, k, "PASSED", diff_percentage])
                            print("PASSED: PE output and expected output difference = " + diff_percentage + "%")
                
                if hls_sim:
                    copy_concat_rtl(i, j, k)

                log = log + result.stdout

    # --- Create and print summary table ---
    summary_header = "\n\n" + "="*70 + "\n"
    summary_header += " " * 25 + "TEST SUITE SUMMARY" + " " * 25 + "\n"
    summary_header += "="*70 + "\n"
    summary_header += f"{'kIntWordWidth':<15} | {'kVectorSize':<12} | {'kNumVectorLanes':<15} | {'Status':<8} | {'Difference (%)':<15}\n"
    summary_header += "-"*80 + "\n"

    log += summary_header
    print(summary_header.strip())

    summary_body = ""
    for i_val, j_val, k_val, status, diff in info:
        row = f"{i_val:<15} | {j_val:<12} | {k_val:<15} | {status:<8} | {diff:<15}\n"
        summary_body += row

    log += summary_body
    print(summary_body.strip())

    if error != 0:
        txt_log = "\nTest Failed with " + str(error) + " errors\n"
        log = log + txt_log
        print(txt_log)
    else:
        txt_log = "\nTest Passed\n"
        log = log + txt_log
        print(txt_log)

    with open(LOG, "w") as f:
        f.write(log)
    
    update_spec(default_kIntWordWidth, default_kVectorSize, default_kNumVectorLanes)

def cleanup():
    """
    Cleans up generated files from simulations and synthesis.
    """
    print("\n--- Cleaning up generated files ---")
    
    # The top-level Makefile's 'clean' target handles cleaning subdirectories.
    # We run it from the script's directory to ensure correct relative paths.
    repo_top = os.path.dirname(os.path.abspath(__file__))
    print(f"Running 'make clean' in {repo_top}")
    subprocess.run(["make", "clean"], check=False, cwd=repo_top)

def main():
    """
    Main function to parse arguments and run the specified simulation.
    """
    parser = argparse.ArgumentParser(description="Run SystemC or HLS simulation for the PE Core.")
    parser.add_argument(
        '--action',
        type=str,
        choices=['systemc_sim', 'rtl_sim', 'clean', 'get_rtl_area'],
        default='systemc_sim',
        help='Specify the action to perform: "systemc_sim", "rtl_sim", "get_rtl_area", or "clean". Default is "systemc_sim".'
    )
    args = parser.parse_args()

    LOGS_FOLDER = "logs"
    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)

    if args.action == 'clean':
        cleanup()
    elif args.action == 'rtl_sim':
        hls_sim = True
        systemc_sim = False
        print("--- Running HLS Simulation ---")
        sim(systemc_sim, hls_sim)
        get_area()
    elif args.action == 'systemc_sim':
        hls_sim = False
        systemc_sim = True
        print("--- Running SystemC Simulation ---")
        sim(systemc_sim, hls_sim)
    elif args.action == 'get_rtl_area':
        get_area()

if __name__ == "__main__":
    main()