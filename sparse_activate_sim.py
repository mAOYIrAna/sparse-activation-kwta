"""
sparse_activate_sim.py —— 稀疏激活判定 RTL 的位精确参考模型 (golden model)
==========================================================================

为什么需要它
------------
`sparse_activate.v` 里的两个模块都是"定点 + 算术移位 + 截断"的运算:
浮点模型验证不了, 而很多环境里又没有 iverilog 可以联仿。本模块用 Python
整数逐位复现 Verilog 的语义, 因此可以:

  * 当 golden model, 给 RTL 回归生成期望值 (verify_rtl.py 直接调用);
  * 快速量化激活率 / 阈值 / 硬件行为, 不必每次都跑仿真;
  * 在综合前发现位宽、截断、时序上的问题。

语义对齐的关键点
----------------
1. Verilog `>>>` 对负数是**向负无穷取整**, Python 的 `>>` 也是 —— 直接用;
2. 全程整数运算, 唯一用到浮点的地方是造测试数据 (np.random.normal);
3. 32-bit 累加器在本设计的取值范围 (|acc| < 2^22) 内不会回绕,
   所以 Python 无界整数与硬件逐位等价; 越界会直接抛 OverflowError。

用法
----
    python sparse_activate_sim.py            # 跑三组检查并打印报告
    python sparse_activate_sim.py --json     # 机器可读输出
    python sparse_activate_sim.py --check    # 只做断言, 失败退出码非 0
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# ============================================================
# 参数 (默认值与 sparse_activate.v 的 parameter 一致)
# ============================================================
DW, AW = 16, 32                     # 数据位宽 / 累加器位宽
SHIFT, BNUM, BSH = 6, 33, 4         # EMA 时间常数 2^6; β = 33/16
N, K, BW = 2000, 100, 16            # 神经元数 / 目标激活数 / 二分轮数

VMIN = -(1 << (DW - 1))             # -32768
VMAX = (1 << (DW - 1)) - 1          #  32767
ACC_MAX = 1 << (AW - 1)             # 32-bit 有符号上限
WARMUP = 1000                       # EMA 收敛期, 统计时丢掉

__all__ = [
    "DW", "AW", "SHIFT", "BNUM", "BSH", "N", "K", "BW",
    "VMIN", "VMAX", "WARMUP",
    "sar", "to_int16", "dyn_thr_stream", "kwta_search",
    "DynThrResult", "KwtaResult", "build_datasets", "run_report", "main",
]


# ============================================================
# 基础算子
# ============================================================
def sar(x: int, n: int) -> int:
    """算术右移, 与 Verilog 的 `>>>` 完全一致 (负数向负无穷取整)。

    注意: C 语言的 `>>` 对负数是实现定义的, 不能直接照搬。
    """
    return x >> n


def to_int16(values: Iterable[float] | np.ndarray) -> np.ndarray:
    """把浮点数据夹到 16-bit 有符号范围并转成 int16 (模拟上游定点化)。"""
    return np.clip(np.asarray(values, dtype=np.float64), VMIN, VMAX).astype(np.int16)


# ============================================================
# A. 动态阈值模块 (dyn_thr_activate) 的功能模型
# ============================================================
@dataclass(frozen=True)
class DynThrResult:
    """动态阈值模块一次流式运行的结果。"""

    acts: np.ndarray        # uint8[0/1], 每个样本的激活位
    thr: np.ndarray         # int32, 每个样本用到的阈值 θ
    peak_acc: int           # max(|mean_acc|, |mad_acc|), 用于 32-bit 溢出检查
    mu_final: int           # 收敛后的 μ
    mad_final: int          # 收敛后的 MAD


def dyn_thr_stream(
    stream: Sequence[int] | np.ndarray,
    *,
    shift: int = SHIFT,
    bnum: int = BNUM,
    bsh: int = BSH,
) -> DynThrResult:
    """逐样本流式判定, 逐位复现 dyn_thr_activate 的 RTL 行为。

    θ = μ + β·MAD, μ/MAD 由两个泄漏累加器在线维护:
        mean_acc ← mean_acc - (mean_acc >>> SHIFT) + s
        mad_acc  ← mad_acc  - (mad_acc  >>> SHIFT) + |s - μ|
    累加器里存的是 μ·2^SHIFT / MAD·2^SHIFT (保留小数位), 否则小信号会被截成 0。
    """
    n = len(stream)
    acts = np.empty(n, dtype=np.uint8)
    thrs = np.empty(n, dtype=np.int32)
    mean_acc = 0
    mad_acc = 0
    peak = 0

    for i in range(n):
        # 关键: 转成 Python int。numpy 标量 (如 np.int16) 参与加减会按原 dtype
        # 回绕/溢出, 与 32-bit 寄存器的行为完全不同, 这里必须避免。
        s = int(stream[i])
        mu = sar(mean_acc, shift)
        mad = sar(mad_acc, shift)
        thr = mu + sar(mad * bnum, bsh)

        acts[i] = 1 if s >= thr else 0
        thrs[i] = thr

        diff = s - mu
        dev = -diff if diff < 0 else diff
        mean_acc = mean_acc - sar(mean_acc, shift) + s
        mad_acc = mad_acc - sar(mad_acc, shift) + dev

        acc = max(abs(mean_acc), abs(mad_acc))
        if acc > peak:
            peak = acc

    if peak >= ACC_MAX:
        raise OverflowError(
            f"累加器越过 32-bit 有符号范围 (|acc|={peak} >= 2^31), "
            "与 RTL 的回绕行为不再等价, 请调大 AW 或减小 SHIFT"
        )

    return DynThrResult(acts, thrs, peak, sar(mean_acc, shift), sar(mad_acc, shift))


# ============================================================
# B. k-WTA 模块 (kwta_activate) 的功能模型
# ============================================================
@dataclass(frozen=True)
class KwtaResult:
    """k-WTA 一次搜索的结果。"""

    mid: int                      # 最终阈值 (第 BW+1 个 mid, 只用于输出)
    acts: np.ndarray              # bool 位图
    trace: tuple[tuple[int, int, int], ...]   # (轮次, mid, >=mid 的个数)

    @property
    def count(self) -> int:
        """实际激活数。"""
        return int(self.acts.sum())


def kwta_search(
    vals: np.ndarray,
    *,
    k: int = K,
    bw: int = BW,
    lo0: int = VMIN,
    hi0: int = VMAX,
) -> KwtaResult:
    """二分搜索阈值, 逐拍复现 kwta_activate 的状态机。

    时序与 RTL 一致:
        初始 mid = (VMIN+VMAX)/2 = -1
        每轮: 数 ge = #{s >= mid} → 收缩边界 → mid = 新边界中点 (向下取整)
        BW 轮后, 用最后一个 mid 输出位图 (这个 mid 没有被计数过)

    收敛到 θ = max{x : #{s >= x} >= K}, 即第 K 大的值:
        * 无并列值时恰好 K 个;
        * 阈值上有并列值时, 多激活的也是并列那批 (至少 K 个, 不会少);
        * 判据必须是 ge >= K (不是 ge > K), 否则收敛到 θ-1, 恒定多激活 1 个。
    """
    lo, hi = lo0, hi0
    mid = sar(lo + hi, 1)
    trace: list[tuple[int, int, int]] = []

    for it in range(bw):
        ge = int(np.count_nonzero(vals >= mid))
        trace.append((it, mid, ge))
        if ge >= k:
            lo = mid          # 第 K 大的值还在 [mid, hi] 里
        else:
            hi = mid
        mid = sar(lo + hi, 1)  # 用**更新后**的边界 (RTL 里是组合逻辑)

    return KwtaResult(mid, np.asarray(vals >= mid), tuple(trace))


# ============================================================
# 报告
# ============================================================
def build_datasets(seed: int = 0) -> dict[str, np.ndarray]:
    """按固定顺序造数据 (顺序即随机数流, 改动顺序会改变所有结果)。"""
    rng = np.random.default_rng(seed)
    return {
        "small": to_int16(rng.normal(0, 10, 4000)),
        "large": to_int16(rng.normal(0, 100, 4000)),
        "mixed": to_int16(np.concatenate([rng.normal(0, 10, 2000),
                                          rng.normal(0, 100, 2000)])),
        "kwta": to_int16(rng.normal(0, 100, N)),
        "low": to_int16(rng.normal(0, 10, N)),
        "high": to_int16(rng.normal(0, 100, N)),
    }


def run_report(seed: int = 0, *, verbose: bool = True) -> dict:
    """跑完三组检查, 返回结构化结果 (同时负责打印)。"""
    d = build_datasets(seed)
    rep: dict = {"seed": seed, "params": {"N": N, "K": K, "DW": DW, "AW": AW,
                                          "SHIFT": SHIFT, "BNUM": BNUM, "BSH": BSH,
                                          "BW": BW, "WARMUP": WARMUP}}

    def log(*a):
        if verbose:
            print(*a)

    # ---------------- A ----------------
    log("=" * 74)
    log("A. 动态阈值模块 (流式 EMA, 免乘法器)")
    log("=" * 74)
    cases = [("小离散度  N(0, 10)", "small"),
             ("大离散度  N(0, 100)", "large"),
             ("混合      N(0,10)+N(0,100)", "mixed")]
    rates, peaks = {}, {}
    for label, key in cases:
        r = dyn_thr_stream(d[key])
        rate = float(r.acts[WARMUP:].mean())
        rates[key] = rate
        peaks[key] = r.peak_acc
        log(f"  {label:<28} 激活率 = {rate:>7.2%}   (后 {len(d[key]) - WARMUP} 点)")

    peak = max(peaks.values())
    log(f"\n  峰值 |累加器| = {peak}  (32-bit 上限 2^31 = {ACC_MAX})")
    log("  → 激活率应稳定在 5% 附近, 且不随输入幅度变化")
    log("     正态分布下 MAD ≈ 0.798σ, β=33/16=2.0625 → β·MAD ≈ 1.645σ → P≈5%")

    # ---------------- B ----------------
    log("")
    log("=" * 74)
    log("B. k-WTA 模块 (二分搜索)")
    log("=" * 74)
    res = kwta_search(d["kwta"])
    n_gt = int(np.count_nonzero(d["kwta"] > res.mid))
    n_eq = int(np.count_nonzero(d["kwta"] == res.mid))
    log(f"  N={N}, K={K}")
    log(f"  搜索收敛阈值 mid = {res.mid}")
    log(f"  最终激活数 = {res.count}  (目标 {K})")
    log(f"  误差 = {abs(res.count - K)}")
    log(f"  严格 > mid 的个数 = {n_gt}, == mid 的个数 = {n_eq}  (并列值会一起激活)")
    log("\n  二分搜索前 5 轮轨迹:")
    for it, m, ge in res.trace[:5]:
        log(f"    轮 {it:2d}:  mid={m:>7}  >=mid 的个数={ge:>6}")
    log(f"\n  → 需要 {BW} 轮串行, 每轮 {N} 个比较器")

    # ---------------- C ----------------
    log("")
    log("=" * 74)
    log("C. 两种机制对同一输入的激活率对比")
    log("=" * 74)
    cmp_rows = []
    for label, key in [("低幅度输入", "low"), ("高幅度输入", "high")]:
        v = d[key]
        kr = kwta_search(v)
        dr = dyn_thr_stream(v)
        k_rate = kr.count / len(v)
        d_rate = float(dr.acts[WARMUP:].mean())
        cmp_rows.append({"label": label, "kwta": k_rate, "dyn": d_rate})
        log(f"  {label:<14} k-WTA 激活率 = {k_rate:>6.2%}   "
            f"动态阈值激活率 = {d_rate:>6.2%}")

    log("\n  → k-WTA 恒为 K/N, 与输入无关;")
    log("     动态阈值同样保持稳定, 但电路规模差约 8 倍 (见 hw_cost.py)")

    rep["A"] = {"rates": rates, "peak_acc": peak}
    rep["B"] = {"mid": res.mid, "count": res.count, "error": abs(res.count - K),
                "n_gt": n_gt, "n_eq": n_eq, "trace": [list(t) for t in res.trace]}
    rep["C"] = cmp_rows
    return rep


def _check(rep: dict) -> int:
    """断言关键不变量, 供 CI 使用。返回退出码。"""
    errs: list[str] = []
    rates = rep["A"]["rates"]
    lo, hi = min(rates.values()), max(rates.values())
    if not (0.03 <= lo and hi <= 0.10):
        errs.append(f"动态阈值激活率越界: {lo:.2%}..{hi:.2%}, 期望 3%~10%")
    if hi - lo > 0.05:
        errs.append(f"激活率随幅度漂移过大: {hi - lo:.2%}")
    if rep["A"]["peak_acc"] >= ACC_MAX:
        errs.append("累加器越界")
    if not (0 <= rep["B"]["count"] - K <= max(1, rep["B"]["n_eq"] + 2)):
        errs.append(f"k-WTA 激活数 {rep['B']['count']} 与 K={K} 偏差异常")
    if rep["B"]["n_gt"] > K:
        errs.append(f"严格大于阈值的个数 {rep['B']['n_gt']} 超过 K={K}")

    for e in errs:
        print(f"CHECK FAILED: {e}")
    print("check: OK" if not errs else f"check: {len(errs)} 项失败")
    return 0 if not errs else 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="稀疏激活判定 (动态阈值 / k-WTA) 的位精确参考模型")
    ap.add_argument("--seed", type=int, default=0, help="随机种子 (默认 0)")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而不是报告")
    ap.add_argument("--check", action="store_true", help="只做断言, 失败时退出码非 0")
    args = ap.parse_args(argv)

    rep = run_report(args.seed, verbose=not args.json)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    if args.check:
        return _check(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
