#!/usr/bin/env python3
"""
verify_rtl.py —— 用 iverilog 做 RTL 与 Python 参考模型的联仿比对
================================================================

流程:
  1. 生成测试向量 (kvec.hex / svec.hex), 同时用参考模型算出期望值;
  2. iverilog 编译 tb_sparse_activate.v + sparse_activate.v;
  3. vvp 跑仿真, 读回 kact.txt / dact.txt;
  4. 逐位比较 (k-WTA 位图 + 动态阈值逐样本激活位)。

用法:
    python verify_rtl.py               # 默认 N=64, K=4, 4 帧, 2000 个流式样本
    python verify_rtl.py --n 200 --k 20 --samples 5000
    python verify_rtl.py --require-tools   # 找不到 iverilog 时直接失败 (CI 用)

没有 iverilog 时脚本会打印 SKIP 并返回 0 (除非 --require-tools)。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import sparse_activate_sim as sim

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------
# 工具定位
# ------------------------------------------------------------
def find_tool(name: str) -> str | None:
    """先在 PATH 里找, 再试 Windows 上的常见安装位置。"""
    found = shutil.which(name)
    if found:
        return found
    for base in (r"C:\iverilog", r"C:\Program Files\iverilog",
                 r"C:\Program Files (x86)\iverilog"):
        cand = Path(base) / "bin" / f"{name}.exe"
        if cand.is_file():
            return str(cand)
    return None


# ------------------------------------------------------------
# 向量生成
# ------------------------------------------------------------
def gen_vectors(n: int, frames: int, samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """造 k-WTA 的多帧输入和动态阈值的流式输入 (int16)。"""
    rng = np.random.default_rng(seed)

    pool = [
        sim.to_int16(rng.normal(0, 1000, n)),
        rng.integers(0, 16, n).astype(np.int16),          # 大量并列值: 考 tie 行为
        sim.to_int16(rng.normal(0, 40, n)),
        np.sort(sim.to_int16(rng.normal(0, 200, n))),     # 单调序列: 考边界
    ]
    kf = np.stack([pool[i % len(pool)] for i in range(frames)])

    quarter = samples // 4
    stages = [
        rng.normal(0, 10, quarter),
        rng.normal(0, 300, quarter),
        rng.normal(0, 30, quarter),
        rng.normal(0, 120, samples - 3 * quarter),
    ]
    stream = sim.to_int16(np.concatenate(stages))
    stream[5], stream[6] = 32767, -32768                  # 边界值
    return kf, stream


def write_hex(path: Path, values) -> None:
    """写 $readmemh 用的 16-bit 补码十六进制文件。"""
    with path.open("w", encoding="ascii", newline="\n") as f:
        for v in np.asarray(values).ravel():
            f.write(f"{int(v) & 0xFFFF:04x}\n")


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RTL ↔ Python 参考模型 联仿比对")
    ap.add_argument("--n", type=int, default=64, help="神经元数 (默认 64)")
    ap.add_argument("--k", type=int, default=4, help="目标激活数 (默认 4)")
    ap.add_argument("--frames", type=int, default=4, help="k-WTA 帧数 (默认 4)")
    ap.add_argument("--samples", type=int, default=2000, help="流式样本数 (默认 2000)")
    ap.add_argument("--seed", type=int, default=2024, help="向量随机种子")
    ap.add_argument("--bw", type=int, default=16, help="二分轮数 (默认 16)")
    ap.add_argument("--build", type=str, default="build/cosim", help="工作目录")
    ap.add_argument("--keep", action="store_true", help="保留工作目录")
    ap.add_argument("--require-tools", action="store_true",
                    help="找不到 iverilog 时视为失败")
    args = ap.parse_args(argv)

    iverilog = find_tool("iverilog")
    vvp = find_tool("vvp")
    if not iverilog or not vvp:
        msg = "SKIP: 找不到 iverilog / vvp, 未做 RTL 联仿"
        print(msg)
        return 2 if args.require_tools else 0

    work = (HERE / args.build).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # ---- 向量 + 期望值 ----
    kf, stream = gen_vectors(args.n, args.frames, args.samples, args.seed)
    write_hex(work / "kvec.hex", kf)
    write_hex(work / "svec.hex", stream)
    exp_frames = [sim.kwta_search(kf[i], k=args.k, bw=args.bw) for i in range(args.frames)]
    exp_stream = sim.dyn_thr_stream(stream)

    # ---- 编译 ----
    sim_exe = work / "sim"
    cmd = [
        iverilog, "-g2012", "-s", "tb_sparse_activate",
        "-o", str(sim_exe),
        f"-Ptb_sparse_activate.N={args.n}",
        f"-Ptb_sparse_activate.K={args.k}",
        f"-Ptb_sparse_activate.NF={args.frames}",
        f"-Ptb_sparse_activate.M={args.samples}",
        f"-Ptb_sparse_activate.BW={args.bw}",
        str(HERE / "tb_sparse_activate.v"),
        str(HERE / "sparse_activate.v"),
    ]
    print("[1/2] 编译 RTL ...")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.stdout.strip():
        print(r.stdout.rstrip())
    if r.returncode != 0 or not sim_exe.exists():
        print(r.stderr.rstrip())
        print(f"FAILED: iverilog 编译失败 (exit {r.returncode})")
        return 1

    # ---- 仿真 ----
    print(f"[2/2] 运行 {args.samples} 拍 ...")
    r = subprocess.run([vvp, str(sim_exe)], cwd=work, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        print(out)
        print(f"FAILED: vvp 运行失败 (exit {r.returncode})")
        return 1

    # ---- 比对 ----
    ok = True
    klines = (work / "kact.txt").read_text(encoding="ascii").split()
    dlines = (work / "dact.txt").read_text(encoding="ascii").split()

    if len(klines) != args.frames:
        print(f"FAILED: kact.txt 行数 {len(klines)} != {args.frames}")
        ok = False
    for f in range(min(len(klines), args.frames)):
        got = klines[f].strip()
        want = "".join("1" if b else "0" for b in exp_frames[f].acts)
        if got != want:
            ok = False
            bad = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]
            print(f"  k-WTA 第 {f} 帧不一致: {len(bad)} 位不同, 首个 index={bad[0]}")
            print(f"    RTL  = {got[max(0, bad[0]-4):bad[0]+8]}")
            print(f"    模型 = {want[max(0, bad[0]-4):bad[0]+8]}")

    if len(dlines) != args.samples:
        print(f"FAILED: dact.txt 行数 {len(dlines)} != {args.samples}")
        ok = False
    n_bad = 0
    first_bad = -1
    for i, line in enumerate(dlines[:args.samples]):
        if int(line) != int(exp_stream.acts[i]):
            n_bad += 1
            if first_bad < 0:
                first_bad = i
    if n_bad:
        ok = False
        print(f"  动态阈值不一致: {n_bad}/{args.samples} 个样本, 首个 index={first_bad} "
              f"(RTL={dlines[first_bad]} 模型={int(exp_stream.acts[first_bad])})")

    counts = [f"{r.count}" for r in exp_frames]
    print(f"  k-WTA    : {args.frames}/{args.frames} 帧逐位一致, 激活数 = [{', '.join(counts)}]"
          if ok else "  k-WTA    : 不一致")
    if ok:
        print(f"  动态阈值 : {args.samples}/{args.samples} 样本一致")
        print(f"  PASS: RTL 与参考模型逐位一致 (N={args.n}, K={args.k}, BW={args.bw})")

    if not args.keep and ok:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"  工作目录: {work}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
