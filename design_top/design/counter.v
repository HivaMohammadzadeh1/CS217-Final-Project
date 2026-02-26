module counter #(parameter WIDTH = 32) (
    input  wire              clk,     // clock
    input  wire              rst_n,   // async reset, active low
    input  wire              en,      // enable: count only when high
    output reg  [WIDTH-1:0]  q        // counter output
);

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      q <= {WIDTH{1'b0}}; 
    else if (en)
      q <= q + 1'b1;
  end

endmodule
