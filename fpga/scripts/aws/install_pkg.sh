# This script clones the AWS FPGA repository from GitHub and setups the environment for F2 FPGA development.

# Clone the AWS FPGA repository
cd ~/
git clone https://github.com/aws/aws-fpga.git

# Download git-lfs if not already installed
sudo apt-get install git-lfs

# Install jq if not already installed
sudo apt  install jq

# Source Hardware Development Kit (HDK) environment
cd ~/aws-fpga
source hdk_setup.sh
echo ""
echo "Download any other dependencies if you see any errors after sourcing the hdk_setup.sh script."
echo ""

