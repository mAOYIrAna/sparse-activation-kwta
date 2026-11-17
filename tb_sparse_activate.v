// ============================================================================
// tb_sparse_activate —— 与 Python 参考模型联仿的测试台
//
// 输入 (由 verify_rtl.py 生成, 十六进制, 每行一个 16-bit 补码值):
//   kvec.hex : N*NF 个数, k-WTA 的 NF 帧数据 (每帧 N 个)
//   svec.hex : M    个数, 动态阈值模块的流式输入
// 输出:
//   kact.txt : NF 行, 每行 N bit 的激活位图
//   dact.txt : M  行, 每行 0/1 (第 i 行 = 第 i 个样本的 act_out)
//
// 用法:
//   iverilog -g2012 -o sim tb_sparse_activate.v sparse_activate.v && vvp sim
//   参数 (N/K/NF/M) 可用 -P 覆盖, 见 verify_rtl.py
// ============================================================================
`timescale 1ns/1ps

module tb_sparse_activate;

    parameter N  = 64;
    parameter DW = 16;
    parameter K  = 4;
    parameter BW = 16;
    parameter NF = 4;            // k-WTA 帧数
    parameter M  = 512;          // 动态阈值流式样本数

    reg clk = 0;
    always #5 clk = ~clk;

    reg rst_n = 0;

    // ---------------- k-WTA ----------------
    reg  [N*DW-1:0] flat = 0;
    reg             start = 0;
    wire [N-1:0]    kact;
    wire            kdone;

    kwta_activate #(.N(N), .DW(DW), .K(K), .BW(BW)) u_kwta (
        .clk(clk), .rst_n(rst_n), .start(start),
        .s_flat(flat), .act(kact), .done(kdone)
    );

    // ---------------- 动态阈值 ----------------
    reg  signed [DW-1:0] s = 0;
    reg                  dv = 0;
    wire                 dact;

    dyn_thr_activate #(.DW(DW)) u_dyn (
        .clk(clk), .rst_n(rst_n), .in_valid(dv), .s_in(s), .act_out(dact)
    );

    // ---------------- 激励 ----------------
    reg signed [DW-1:0] kmem [0:N*NF-1];
    reg signed [DW-1:0] smem [0:M-1];

    integer f, i, fk, fd, fh;

    task wait_done;
        integer t;
        begin
            t = 0;
            while (!kdone && t < 4*BW + 32) begin
                @(posedge clk);
                t = t + 1;
            end
            if (!kdone) begin
                $display("FATAL: k-WTA 超时, 没有等到 done");
                $finish;
            end
        end
    endtask

    initial begin
        // 有向量文件就按文件跑 (verify_rtl.py 的逐位比对), 没有就自己造数据独立跑
        fh = $fopen("kvec.hex", "r");
        if (fh != 0) begin
            $fclose(fh);
            $readmemh("kvec.hex", kmem);
            $readmemh("svec.hex", smem);
            $display("tb_sparse_activate: 读入 kvec.hex / svec.hex");
        end else begin
            for (i = 0; i < N*NF; i = i + 1)
                kmem[i] = ($random % 800) + ($random % 800) + ($random % 800);
            for (i = 0; i < M; i = i + 1)
                smem[i] = ($random % 800) + ($random % 800) + ($random % 800);
            $display("tb_sparse_activate: 未找到向量文件, 使用内置随机数据 (近似正态)");
        end
        fk = $fopen("kact.txt", "w");
        fd = $fopen("dact.txt", "w");

        repeat (4) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        // ---- k-WTA: 逐帧跑 ----
        for (f = 0; f < NF; f = f + 1) begin
            @(negedge clk);
            for (i = 0; i < N; i = i + 1)
                flat[i*DW +: DW] = kmem[f*N + i];
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            wait_done;
            @(negedge clk);
            // 按索引顺序输出 (第 i 位 = 第 i 个神经元), 避免 %b 的 MSB-first 歧义
            for (i = 0; i < N; i = i + 1)
                $fwrite(fk, "%0d", kact[i]);
            $fwrite(fk, "\n");
        end

        // ---- 动态阈值: 复位后流式输入 ----
        @(negedge clk);
        rst_n = 1'b0;
        repeat (2) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        for (i = 0; i < M; i = i + 1) begin
            s  = smem[i];
            dv = 1'b1;
            @(posedge clk);
            #1;
            $fwrite(fd, "%0d\n", dact);
            @(negedge clk);
        end

        $fclose(fk);
        $fclose(fd);
        $display("tb_sparse_activate: N=%0d K=%0d NF=%0d M=%0d -> kact.txt / dact.txt",
                 N, K, NF, M);
        $finish;
    end

endmodule
