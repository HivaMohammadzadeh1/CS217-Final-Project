# FPGA Rebuild Guide — Fix MXFP8 Timing Violation

## Background

We built an FPGA design that supports 3 precision modes: INT8, MXFP8, MXFP4. INT8 and MXFP4 pass hardware validation, but MXFP8 fails intermittently (~3/4 runs) due to a timing violation. The fix is rebuilding with a slower clock (A0 instead of A1).

---

## Step 1: Launch an F2 instance

- AMI: Search for "FPGA Developer AMI" in the AWS Marketplace
- Instance type: `f2.2xlarge` (or `f2.4xlarge` for faster build)
- Storage: at least 100GB
- Make sure your security group allows SSH (port 22)
- Launch and SSH in:

```bash
ssh -i your-key.pem ubuntu@<instance-ip>
```

## Step 2: Set up the AWS FPGA SDK

```bash
cd ~
git clone https://github.com/aws/aws-fpga.git
cd aws-fpga
source hdk_setup.sh
```

This takes ~5-10 minutes. Wait until it says "hdk_setup.sh PASSED".

## Step 3: Clone our repo

```bash
cd ~
git clone https://github.com/HivaMohammadzadeh1/CS217-Final-Project.git
cd CS217-Final-Project
git checkout grant-branch-interval4
git pull origin grant-branch-interval4
```

## Step 4: Prepare the RTL

```bash
cd ~/CS217-Final-Project/fpga/design_top
cp design/concat_PECore/kIntWordWidth_8_kVectorSize_16_kNumVectorLanes_16/concat_PECore.v design/concat_PECore.v
cd design/ && sed -i '/\/\/ synopsys translate/d' concat_PECore.v && cd ..
```

## Step 5: Set environment variable

```bash
export AWS_FPGA_REPO_DIR=/home/ubuntu/aws-fpga
```

## Step 6: Start the FPGA build in tmux

Start it in tmux so it survives SSH disconnect:

```bash
tmux new -s fpga_build
cd ~/CS217-Final-Project/fpga/design_top/build/scripts
./aws_build_dcp_from_cl.py -c design_top --clock_recipe_a A0
```

This will take **3-4 hours**.

- To detach from tmux (leave it running): press `Ctrl+B` then `D`
- To reattach later: `tmux attach -t fpga_build`

## Step 7: Check the build output

When the build finishes, look at the end of the output for:

- **GOOD:** "Finished building design checkpoints" with NO "timing failure" warning
- **BAD:** "Detected a post-route DCP with timing failure" — this means it still didn't meet timing (unlikely with A0 but possible)

Check if a tarball was created:

```bash
ls ~/CS217-Final-Project/fpga/design_top/build/checkpoints/*.Developer_CL.tar
```

If you see a `.tar` file, the build fully completed. **Skip to Step 8.**

### If the build crashed with a release_version.txt error

If you see this error:
```
FileNotFoundError: [Errno 2] No such file or directory: '.../aws-fpga/release_version.txt'
```

Fix the env var:
```bash
export AWS_FPGA_REPO_DIR=/home/ubuntu/aws-fpga
```

Then find the build tag by looking at the checkpoint files:
```bash
ls ~/CS217-Final-Project/fpga/design_top/build/checkpoints/*.post_route*
```

You'll see a filename like `design_top.YYYY_MM_DD-HHMMSS.post_route.dcp` (or `.post_route.VIOLATED.dcp` if timing failed). Note the timestamp part (e.g. `2026_03_20-123456`).

Create the tarball manually (replace `YYYY_MM_DD-HHMMSS` with your actual timestamp):

```bash
BUILD_TAG=YYYY_MM_DD-HHMMSS
CHECKPOINTS=~/CS217-Final-Project/fpga/design_top/build/checkpoints
mkdir -p ${CHECKPOINTS}/to_aws
cp ${CHECKPOINTS}/design_top.${BUILD_TAG}.post_route.dcp ${CHECKPOINTS}/to_aws/${BUILD_TAG}.SH_CL_routed.dcp
cp ${CHECKPOINTS}/${BUILD_TAG}.debug_probes.ltx ${CHECKPOINTS}/to_aws/${BUILD_TAG}.debug_probes.ltx
DCP_HASH=$(sha256sum ${CHECKPOINTS}/to_aws/${BUILD_TAG}.SH_CL_routed.dcp | cut -d' ' -f1)
printf "pci_device_id=0xF010\npci_vendor_id=0x1D0F\npci_subsystem_id=0x1D51\npci_subsystem_vendor_id=0xFEDC\nmanifest_format_version=2\ndcp_hash=${DCP_HASH}\nshell_version=0x10212415\nhdk_version=2.2.2\ntool_version=v2024.1\ndate=${BUILD_TAG}\nclock_recipe_a=A0\nclock_recipe_b=B2\nclock_recipe_c=C0\nclock_recipe_hbm=H2\ndcp_file_name=${BUILD_TAG}.SH_CL_routed.dcp\n" > ${CHECKPOINTS}/to_aws/${BUILD_TAG}.manifest.txt
cd ${CHECKPOINTS} && tar cf ${BUILD_TAG}.Developer_CL.tar -C ${CHECKPOINTS} to_aws/${BUILD_TAG}.SH_CL_routed.dcp to_aws/${BUILD_TAG}.debug_probes.ltx to_aws/${BUILD_TAG}.manifest.txt
```

Verify the tarball has 3 files:
```bash
tar tf ${CHECKPOINTS}/${BUILD_TAG}.Developer_CL.tar
```

Should show:
```
to_aws/YYYY_MM_DD-HHMMSS.SH_CL_routed.dcp
to_aws/YYYY_MM_DD-HHMMSS.debug_probes.ltx
to_aws/YYYY_MM_DD-HHMMSS.manifest.txt
```

## Step 8: Generate the AFI

```bash
cd ~/CS217-Final-Project/fpga/design_top
source setup.sh
make generate_afi
```

This takes ~30 minutes. Check status:

```bash
make check_afi_available
```

Keep running this every few minutes until the output shows `"State": { "Code": "available" }`.

## Step 9: Program the FPGA

```bash
make program_fpga
```

You should see `loaded` and `ok` in the output.

## Step 10: Validate all 3 modes

Run each test one at a time:

```bash
make run_fpga_test
```
Should say TEST PASSED (this is INT8).

```bash
make run_fpga_test FPGA_TEST_ARGS="MXFP8 8"
```
Should say TEST PASSED. **This is the critical one — it was failing on the old build.**

```bash
make run_fpga_test FPGA_TEST_ARGS="MXFP4 8"
```
Should say TEST PASSED.

Now run MXFP8 **4 more times** to confirm it's stable:

```bash
make run_fpga_test FPGA_TEST_ARGS="MXFP8 8"
make run_fpga_test FPGA_TEST_ARGS="MXFP8 8"
make run_fpga_test FPGA_TEST_ARGS="MXFP8 8"
make run_fpga_test FPGA_TEST_ARGS="MXFP8 8"
```

All should say TEST PASSED.

## Step 11: Report back to Grant

Send Grant:

1. Whether all tests passed (especially MXFP8 all 4 runs)
2. Whether the build had any timing violation warnings
3. The AFI ID — run this and send the output:

```bash
cat ~/CS217-Final-Project/fpga/design_top/generated_afid.sh
```

**Important note:** The AFI is tied to your AWS account, so Grant can't load it on his instance. The MXFP8 experiments will need to run on YOUR instance. Grant will send you the commands to run.

---

## Troubleshooting

- If `source hdk_setup.sh` fails: make sure you're in the `~/aws-fpga` directory
- If `make program_fpga` fails: make sure you ran `source setup.sh` first
- If `make run_fpga_test` says "command not found": run `cd ~/CS217-Final-Project/fpga/design_top` first
- If anything crashes with permissions: try prefixing with `sudo`
- If the build takes longer than 5 hours: that's normal for larger instances, just let it finish
- If you get disconnected from SSH: the build continues in tmux, just SSH back in and run `tmux attach -t fpga_build`
