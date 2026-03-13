setenv REPO_TOP `realpath ./`
setenv SRC_HOME `realpath ./src/`
setenv HLS_HOME `realpath ./hls/`
setenv AWS_HOME `realpath ./design_top/`

module load $REPO_TOP/scripts/hls/catapult.module
setenv VG_GNU_PACKAGE /cad/synopsys/vcs_gnu_package/S-2021.09/gnu_9/linux
source /cad/synopsys/vcs_gnu_package/S-2021.09/gnu_9/linux/source_me.csh
