// ============================================================================
// 稀疏激活判定: 动态阈值 vs k-WTA  —— 可综合 RTL
//
// 两个独立模块, 单独例化即可:
//   dyn_thr_activate : 流式动态阈值 θ = μ + β·MAD, 无乘法器, 无需缓存
//   kwta_activate    : 二分搜索阈值, 选出最大的 K 个
//
// 仿真:  iverilog -g2012 -o sim tb_sparse_activate.v sparse_activate.v && vvp sim
// 综合:  用 Vivado / Quartus 直接加这个文件, 顶层分别选两个模块
// ============================================================================
`timescale 1ns/1ps

// ----------------------------------------------------------------------------
// A. 动态阈值激活判定 (流式, 免乘法器)
//    θ = μ + β·MAD
//    μ、MAD 由两个泄漏累加器在线维护, 不需要缓存任何神经元的值,
//    每个样本进来当场判定, 单趟完成。
//    注意: 乘 β 是"乘常数", 综合成移位加法, 不是乘法器;
//          求平均绝对偏差只用减法+取绝对值, 同样没有乘法器。
//
//    位宽: 累加器里存的是 μ·2^SHIFT 与 MAD·2^SHIFT (Q(SHIFT) 定标),
//          保留小数位, 否则 >>> SHIFT 会把小信号直接截断成 0, 累加器永不收敛。
//          上界: |mean_acc| ≤ 32767·2^SHIFT, |mad_acc| ≤ 65535·2^SHIFT,
//          SHIFT=6 时约 2^21 / 2^22, AW=32 有数百倍余量, 不会回绕。
// ----------------------------------------------------------------------------
module dyn_thr_activate #(
    parameter DW    = 16,        // 数据位宽
    parameter AW    = 32,        // 累加器位宽
    parameter SHIFT = 6,         // EMA 时间常数 = 2^SHIFT
    parameter BNUM  = 33,        // β ≈ 33/16 = 2.0625
    parameter BSH   = 4
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 in_valid,
    input  wire signed [DW-1:0] s_in,
    output reg                  act_out
);

    reg signed [AW-1:0] mean_acc;
    reg signed [AW-1:0] mad_acc;

    wire signed [AW-1:0] s_ext = {{(AW-DW){s_in[DW-1]}}, s_in};
    wire signed [AW-1:0] mu    = mean_acc >>> SHIFT;
    wire signed [AW-1:0] mad   = mad_acc  >>> SHIFT;
    wire signed [AW-1:0] diff  = s_ext - mu;
    wire signed [AW-1:0] dev   = diff[AW-1] ? (~diff + 1'b1) : diff;   // |s-μ|
    wire signed [AW-1:0] thr   = mu + ((mad * BNUM) >>> BSH);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mean_acc <= {AW{1'b0}};
            mad_acc  <= {AW{1'b0}};
            act_out  <= 1'b0;
        end else if (in_valid) begin
            // 泄漏积分: 全精度加, 只对衰减项移位
            mean_acc <= mean_acc - (mean_acc >>> SHIFT) + s_ext;
            mad_acc  <= mad_acc  - (mad_acc  >>> SHIFT) + dev;
            act_out  <= (s_ext >= thr);
        end
    end

endmodule


// ----------------------------------------------------------------------------
// B. k-WTA 激活判定 (二分搜索阈值, 数字实现里较省的一种)
//    必须先把全部 N 个神经元的值存进寄存器, 再串行 BW 轮二分搜索,
//    每轮要 N 个比较器 + 一棵归约加法树。
//
//    时序 (与 Python 模型逐拍一致):
//      start 当拍  : lo=VMIN, hi=VMAX, mid=(VMIN+VMAX)/2 = -1
//      之后 BW 拍  : 数 ge = #{s >= mid}; 收缩边界; mid = 新边界中点
//                   最后第 BW 拍的 mid 只用于输出位图, 不再计数
//    因此比较器一共工作 BW 次, 输出对应的是第 BW+1 个 mid。
//
//    收敛到 θ = max{ x : #{s >= x} >= K }, 即第 K 大的值:
//      * 数据无并列值时, 恰好激活 K 个;
//      * 阈值上有并列值时, 多激活的几个也是并列 (至少 K 个, 不会少);
//      * 判据用 ge >= K (不是 > K), 否则会收敛到 θ-1, 恒定多激活 1 个。
// ----------------------------------------------------------------------------
module kwta_activate #(
    parameter N  = 2000,
    parameter DW = 16,
    parameter K  = 100,
    parameter BW = 16            // 二分搜索轮数 (= 数据位宽时可达 1 LSB 精度)
)(
    input  wire            clk,
    input  wire            rst_n,
    input  wire            start,
    input  wire [N*DW-1:0] s_flat,
    output wire [N-1:0]    act,
    output reg             done
);

    localparam signed [DW-1:0] VMAX = {1'b0, {(DW-1){1'b1}}};
    localparam signed [DW-1:0] VMIN = {1'b1, {(DW-1){1'b0}}};
    localparam                 CW   = (N <= 1) ? 1 : $clog2(N + 1);   // 数得下 0..N

    reg signed [DW-1:0] lo, hi, mid;
    reg [4:0]           iter;                         // 支持 BW <= 32
    reg                 busy;
    integer             i;
    reg [CW-1:0]        ge_cnt;                       // >= mid 的个数

    // ---- N 路比较 + 归约计数 ----
    always @* begin
        ge_cnt = {CW{1'b0}};
        for (i = 0; i < N; i = i + 1)
            if ($signed(s_flat[i*DW +: DW]) >= mid)
                ge_cnt = ge_cnt + 1'b1;
    end

    // ---- 下一拍边界: 先组合算出, 再求中点 ----
    //      1) 用组合的 lo_nxt/hi_nxt, 避免非阻塞赋值读到旧值导致 mid 滞后一拍
    //      2) 用 DW+1 位求和, 避免 (lo+hi) 在 16 位下溢出
    wire                 ge_ge_k = (ge_cnt >= K);      // >= K: 收敛到第 K 大的值
    wire signed [DW-1:0] lo_nxt  = ge_ge_k ? mid : lo;
    wire signed [DW-1:0] hi_nxt  = ge_ge_k ? hi  : mid;
    wire signed [DW:0]   sum_nxt = $signed({lo_nxt[DW-1], lo_nxt})
                                 + $signed({hi_nxt[DW-1], hi_nxt});

    // ---- 搜索状态机 ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lo   <= VMIN;
            hi   <= VMAX;
            mid  <= VMAX;                 // 空闲时 act 全 0
            iter <= 5'd0;
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                lo   <= VMIN;
                hi   <= VMAX;
                mid  <= -1;               // (VMIN+VMAX)/2, 见文件头时序说明
                iter <= 5'd0;
                busy <= 1'b1;
            end else if (busy) begin
                // 若 >= mid 的个数不少于 K, 说明第 K 大的值还在 [mid, hi] 里
                lo   <= lo_nxt;
                hi   <= hi_nxt;
                mid  <= sum_nxt[DW:1];
                iter <= iter + 5'd1;
                if (iter == BW - 1) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end

    // ---- 用最终阈值输出激活位图 (组合, 单一驱动源) ----
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : g_act
            assign act[g] = ($signed(s_flat[g*DW +: DW]) >= mid);
        end
    endgenerate

endmodule
