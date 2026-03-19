// ------------------------------------------------------------------
// SYNOPSYS CONFIDENTIAL                                             
//                                                                   
// This is an unpublished, proprietary work of Synopsys, Inc., and   
// is fully protected under copyright and trade secret laws. You may 
// not view, use, disclose, copy, or distribute this file or any     
// information contained herein except pursuant to a valid written   
// license from Synopsys.                                            
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// This file is machine generated.                                   
// Any change made to this file will be lost after next elaboration. 
// ------------------------------------------------------------------
#include <iostream>
#include <string.h>
#include <string>
#ifdef __cplusplus
extern "C" {
#endif

  int sysc_rel = 2;
  int sysc_timeres_set = 1;
  int sysc_xlrm_time = 0;
  int sysc_q_flag = 0;
  int sysc_on_top = 1;
  int sysc_adjust_time = 0;
  int sysc_use_hdl_timeres = 0;
  int sysc_diag = 0;
  int snps_systemc_version = 233;
  int sysc_donut = 0;
  int sysc_only = 0;
  int sysc_new_sync = 0;
  int sysc_skeleton = 1;
  int sysc_show_sc_main = 1;
  int is_mx_design = 0;
  int sysc_vhdl_on_top = 0;
  int sysc_slave = 0;
  int sysc_virtio = 0;
  int sysc_innovator = 0;
  int sysc_debug_mode = 0;
  int sysc_multi_start = 1;
  int sysc_inout_opt_if = 0;
  int sysc_uni_test = 0;
  int sysc_link_shared = 0;
  int sysc_uni_phase = 3;
  int sysc_purify = 0;
  int sysc_tli = 1;
  int sysc_multi_top = 0;
  int sysc_dlp_active = 0;
  int sysc_has_pli = 1;
  int sysc_has_dpi = 0;
  int sysc_has_opt = 0;
  bool sysc_multi_top_ovrd = 0;
  const char *sysc_compiler_prefix = "gcc9";
  const char *sysc_mlib = "";
  const char *sysc_mdir = "scverify/concat_sim_PECore_v_vcs";
  const char *sysc_top_module_name = "sc_main";
  const char *linklibs = "";
  const char *sysc_tool_home = "/cad/synopsys/vcs/U-2023.03/linux64";
  const char *sysc_sdf_annotate = " ";
  const char *sysc_vcsi_home_internal = "";
  const char *sysc_vcs_cosim_output_directory = "scverify/concat_sim_PECore_v_vcs";
  const char *sysc_make_dir = "/home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/scverify/concat_sim_PECore_v_vcs";
  const char *sysc_top = "scverify/concat_sim_PECore_v_vcs/sysc/scverifyconcat_sim_PECore_v_vcssc_mainpwr";
  const char *sysc_pwd = "/home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1";
  const char *_sysc_cpp = "/cad/synopsys/vcs_gnu_package/S-2021.09/gnu_9/linux/gcc-9.2.0_64-shared/bin/g++";
  const char *_sysc_as = "/cad/synopsys/vcs_gnu_package/S-2021.09/gnu_9/linux/binutils-2.33.1_64/bin/as";
  const char *sysc_md5_checksum = "17f74a9c5cc0d6fd94611e9524400996";
  const char *sysc_arch = "linux64";
  const char *sysc_output = "/home/users/hiva/CS217-Final-Project/fpga/hls/PECore/Catapult/PECore.v1/scverify/concat_sim_PECore_v_vcs/sc_mainpwr";
  const char *wreal_res_func = "single";


  extern int SimvMain(int, char **);
  extern int bf_main(int, char **);


  extern int __attribute__((weak)) VcsSysCMain(int a, char **b);
  int VcsSysCMain(int a, char **b) {
    return -1;
  }

  int SyscSimvMain(int a, char **b) {
    return SimvMain(a, b);
  }

  extern void save_restore_check(int, char **);
  int __attribute__((weak)) main(int argc, char **argv) {
    save_restore_check(argc, argv);
    return bf_main(argc, argv);
  }

  char *mhdlGetName(void *) { return 0; }
  void *mhdlGetChildIterator(void *) { return 0; }
  void *mhdlScan(void *, void *) { return 0; }

int bfvForceInit(int, char **) { return 0; }

/***********************************************
 ***********************************************/

  void BF_postponeRegistration(const char *insttype, const char *instname);
  void BF_addPortMap(void *, int, const std::string &, int);
  void *createHsOptNode(char *insttype, char *instname) {
    char *stype = strrchr(insttype, '.');
    if (!stype) stype = insttype; else stype++;
  std::cout << "Sorry not found\n";
  return 0;
  }

  int sysc_opt_if = 0;
      void *cbug_wrap_get_sysc_object_by_name(const char *);
const char *cbug_wrap_get_sysc_active_scope(void *, int);
      void *cbug_wrap_get_sysc_object(const char *);
const char *cbug_wrap_get_sysc_object_type_by_name(const char *);
const char *cbug_wrap_get_sysc_whatis(const char *, const char *);
int _warp() {
  cbug_wrap_get_sysc_object_by_name("forty two");
  cbug_wrap_get_sysc_active_scope(0, 0);
  cbug_wrap_get_sysc_object("foo");
  cbug_wrap_get_sysc_object_type_by_name("bar");
  cbug_wrap_get_sysc_whatis("foo", "bar");
  return 0;
}

#ifdef __cplusplus
}
#endif

