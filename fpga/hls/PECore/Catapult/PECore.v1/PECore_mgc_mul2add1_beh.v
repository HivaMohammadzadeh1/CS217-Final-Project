//mul2add1
module PECore_mgc_mul2add1(a,b,b2,c,d,d2,cst,z);
  parameter gentype = 0;
  parameter width_a = 0;
  parameter signd_a = 0;
  parameter width_b = 0;
  parameter signd_b = 0;
  parameter width_b2 = 0;
  parameter signd_b2 = 0;
  parameter width_c = 0;
  parameter signd_c = 0;
  parameter width_d = 0;
  parameter signd_d = 0;
  parameter width_d2 = 0;
  parameter signd_d2 = 0;
  parameter width_e = 0;
  parameter signd_e = 0;
  parameter width_z = 0;
  parameter isadd = 1;
  parameter add_b2 = 1;
  parameter add_d2 = 1;
  parameter use_const = 1;
  input  [width_a-1:0] a;
  input  [width_b-1:0] b;
  input  [width_b2-1:0] b2; // spyglass disable SYNTH_5121,W240
  input  [width_c-1:0] c;
  input  [width_d-1:0] d;
  input  [width_d2-1:0] d2; // spyglass disable SYNTH_5121,W240
  input  [width_e-1:0] cst; // spyglass disable SYNTH_5121,W240
  output [width_z-1:0] z;

  function integer MAX;
    input integer LEFT, RIGHT;
  begin
    if (LEFT > RIGHT) MAX = LEFT;
    else              MAX = RIGHT;
  end endfunction
  function integer ZLEN;
    input integer a_len, b_len, c_len, d_len, e_len;
  begin
    ZLEN = MAX(a_len+b_len, MAX(c_len+d_len,e_len)) + 2;
  end endfunction
  function integer PREADDLEN;
    input integer b_len, d_len, width_d;
  begin
    if(width_d) PREADDLEN = MAX(b_len,d_len) + 1;
    else        PREADDLEN = b_len;
  end endfunction
  function integer PREADDMULLEN;
    input integer a_len, b_len, d_len, width_d;
  begin
    PREADDMULLEN = a_len + PREADDLEN(b_len,d_len,width_d);
  end endfunction

  localparam a_len    = width_a-signd_a+1;
  localparam b_len    = width_b-signd_b+1;
  localparam b2_len   = width_b2-signd_b2+1;
  localparam c_len    = width_c-signd_c+1;
  localparam d_len    = width_d-signd_d+1;
  localparam d2_len   = width_d2-signd_d2+1;
  localparam e_len    = width_e-signd_e+1;
  localparam bpb2_len = PREADDLEN(b_len, b2_len, width_b2);
  localparam dpd2_len = PREADDLEN(d_len, d2_len, width_d2);
  localparam axb_len  = PREADDMULLEN(a_len, b_len, b2_len, width_b2);
  localparam cxd_len  = PREADDMULLEN(c_len, d_len, d2_len, width_d2);
  localparam z_len    = ZLEN(a_len, bpb2_len, c_len, dpd2_len, e_len);

  reg [a_len-1:0]   aa;
  reg [b_len-1:0]   bb;
  reg [b2_len-1:0]  bb2;
  reg [c_len-1:0]   cc;
  reg [d_len-1:0]   dd;
  reg [d2_len-1:0]  dd2;
  reg [e_len-1:0]   ee;
  reg [bpb2_len-1:0]  b_bb2;
  reg [dpd2_len-1:0]  d_dd2;
  reg [axb_len-1:0] axb;
  reg [cxd_len-1:0] cxd;
  reg [z_len-1:0]   zz;

  // make all inputs signed
  always @(*) aa = signd_a ? a : {1'b0, a};
  always @(*) bb = signd_b ? b : {1'b0, b};
  generate if (width_b2) begin
    (* keep ="true" *) reg [b2_len-1:0]  bb2_keep;
    always @(*) bb2_keep = signd_b2 ? b2 : {1'b0, b2};
    always @(*) bb2 = bb2_keep;
  end endgenerate
  always @(*) cc = signd_c ? c : {1'b0, c};
  always @(*) dd = signd_d ? d : {1'b0, d};
  generate if (width_d2) begin
    (* keep ="true" *) reg [d2_len-1:0]  dd2_keep;
    always @(*) dd2_keep = signd_d2 ? d2 : {1'b0, d2};
    always @(*) dd2 = dd2_keep;
  end endgenerate
  always @(*) ee = signd_e ? cst : {1'b0, cst};

  // perform preadd1
  generate
    if (width_b2) begin
      if (add_b2) begin always @(*)  b_bb2 = $signed(bb) + $signed(bb2); end
      else        begin always @(*)  b_bb2 = $signed(bb) - $signed(bb2); end
    end else      begin always @(*)  b_bb2 = $signed(bb); end
  endgenerate

  // perform preadd2
  generate
    if (width_d2) begin
      if (add_d2) begin always @(*)  d_dd2 = $signed(dd) + $signed(dd2); end
      else        begin always @(*)  d_dd2 = $signed(dd) - $signed(dd2); end
    end else      begin always @(*)  d_dd2 = $signed(dd); end
  endgenerate

  // perform muladd1
  always @(*) axb = $signed(aa) * $signed(b_bb2);
  always @(*) cxd = $signed(cc) * $signed(d_dd2);
  generate
    if (use_const>0) begin
      if ( isadd) begin always @(*) zz = $signed(axb) + $signed(cxd) + $signed(ee); end else
                  begin always @(*) zz = $signed(axb) - $signed(cxd) + $signed(ee); end
    end else begin
      if ( isadd) begin always @(*) zz = $signed(axb) + $signed(cxd); end else
                  begin always @(*) zz = $signed(axb) - $signed(cxd); end
    end
  endgenerate

  // adjust output
  assign z = zz;

endmodule // mgc_mul2add1
