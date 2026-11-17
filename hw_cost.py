"""
稀疏激活判定机制的硬件代价模型
================================
对比两种实现:
  A. k-WTA       选出激活值最大的 K 个神经元 (需要选择/仲裁)
  B. 动态阈值    θ = μ + β·MAD, 每个神经元独立比较

代价口径说明:
  算子代价以"16 位加法器"为 1.0 归一化, 取自公开的 28nm 标准单元面积量级。
  这是设计前期的估算模型, 不是综合结果。真实面积/功耗需 Vivado/Quartus 综合。

用法:
    python hw_cost.py
    python hw_cost.py --n 4096 --k 205 --bits 16
"""
from __future__ import annotations

import argparse

# ---------------- 默认参数 ----------------
W = 16          # 数据位宽
N = 2000        # 神经元数量
K = 100         # 目标激活数 (5%)
B = 16          # k-WTA 二分搜索的迭代次数 (= 位宽)

# ---------------- 算子代价表 (归一化到 16 位加法器) ----------------
COST = {
    "add":   1.00,      # 16 位加法器
    "sub":   1.00,
    "cmp":   0.90,      # 16 位比较器
    "mul":  12.00,      # 16x16 乘法器
    "abs":   0.30,      # 取绝对值
    "mux":   0.50,      # 16 位 2:1 选择器
    "reg":   0.60,      # 16 位寄存器
    "cnt":   0.80,      # 计数器 (log2(N)+1 位)
    "sqrt": 45.00,      # 16 位开方 (可用比较平方替代, 见变体)
}


def tree_levels(n: int) -> int:
    """归约树深度。"""
    lv, m = 0, n
    while m > 1:
        m = (m + 1) // 2
        lv += 1
    return lv


def structure_kwta(n: int = N, b: int = B) -> tuple[dict, float, int, int]:
    """k-WTA: 二分搜索阈值 (数字实现里较省的一种)。"""
    ops = {}
    levels = tree_levels(n)

    # b 轮: 每轮 n 个比较器 + 计数器累加 + 归约树
    ops["cmp"] = b * n
    ops["cnt"] = b * n * 0.1              # 计数器按位宽折算
    ops["add"] = b * (n - 1)              # 每轮数有多少个 >= 阈值

    # 必须缓存全部 n 个神经元的值 (两阶段: 先收集, 再选)
    ops["reg"] = n

    area = (ops["cmp"] * COST["cmp"] + ops["add"] * COST["add"]
            + ops["cnt"] * COST["cnt"] + ops["reg"] * COST["reg"])

    # 关键路径: b 轮串行, 每轮 = 1 比较 + 归约树
    path = b * (1 + levels)
    return ops, area, path, levels


def structure_dyn_var(n: int = N) -> tuple[dict, float, int, int]:
    """动态阈值 θ = μ + βσ (含开方)。"""
    ops = {}
    levels = tree_levels(n)

    ops["add"] = (n - 1) * 2              # Σs 和 Σs²
    ops["mul"] = n                        # 每个 s 求平方
    ops["sub"] = n                        # s - μ
    ops["mul"] += n                       # (s-μ)²
    ops["cmp"] = n                        # 与阈值比较
    ops["sqrt"] = 1
    ops["reg"] = 2                        # 流式: 只需两个泄漏累加器

    area = (ops["add"] * COST["add"] + ops["mul"] * COST["mul"]
            + ops["sub"] * COST["sub"] + ops["cmp"] * COST["cmp"]
            + ops["sqrt"] * COST["sqrt"] + ops["reg"] * COST["reg"])
    path = levels + 4 + 1
    return ops, area, path, levels


def structure_dyn_mad(n: int = N) -> tuple[dict, float, int, int]:
    """动态阈值, 用平均绝对偏差替代标准差 (不需要乘法器)。"""
    ops = {}
    levels = tree_levels(n)

    ops["add"] = (n - 1) + n              # Σs, Σ|s-μ|
    ops["sub"] = n
    ops["abs"] = n
    ops["cmp"] = n
    ops["reg"] = 2

    area = (ops["add"] * COST["add"] + ops["sub"] * COST["sub"]
            + ops["abs"] * COST["abs"] + ops["cmp"] * COST["cmp"]
            + ops["reg"] * COST["reg"])
    path = levels + 3 + 1
    return ops, area, path, levels


def report(n: int = N, k: int = K, w: int = W, b: int = B) -> dict:
    res = {
        "k-WTA (二分搜索选阈值)": structure_kwta(n, b),
        "动态阈值 μ+βσ (含开方)": structure_dyn_var(n),
        "动态阈值 (平均绝对偏差)": structure_dyn_mad(n),
    }
    base = res["k-WTA (二分搜索选阈值)"][1]

    print("=" * 82)
    print(f"硬件代价模型  N={n} 神经元, K={k}, 位宽={w}")
    print("=" * 82)
    print(f"{'实现':<26}{'面积(归一)':>12}{'相对k-WTA':>11}{'关键路径':>10}"
          f"{'需缓存N个值':>13}")
    print("-" * 82)
    for name, (ops, area, path, lv) in res.items():
        buf = "是 (N个寄存器)" if ops.get("reg", 0) >= n else "否 (2个累加器)"
        print(f"{name:<26}{area:>12,.0f}{area / base:>10.2f}x{path:>10}{buf:>15}")

    print("\n" + "=" * 82)
    print("算子数量明细")
    print("=" * 82)
    keys = ["add", "sub", "mul", "abs", "cmp", "cnt", "sqrt", "reg"]
    print(f"{'实现':<26}" + "".join(f"{key:>9}" for key in keys))
    print("-" * 82)
    for name, (ops, area, path, lv) in res.items():
        print(f"{name:<26}" + "".join(f"{ops.get(key, 0):>9,.0f}" for key in keys))

    ratio = res["动态阈值 (平均绝对偏差)"][1] / base
    print("\n" + "=" * 82)
    print("结论")
    print("=" * 82)
    print("  1. k-WTA 的大头是 B×N 个比较器和 N 个寄存器 —— 两阶段结构,")
    print(f"     必须先缓存全部 {n} 个神经元的值, 再串行 {b} 轮搜阈值。")
    print("  2. 动态阈值如果老老实实算 σ(含 N 个乘法器), 面积并不占优 ——")
    print("     16x16 乘法器约等于 12 个加法器, N 个乘法器把优势吃光了。")
    print("  3. 动态阈值真正的优势来自两点:")
    print(f"     · 用免乘法器的离散度度量(平均绝对偏差)后, 面积降到 k-WTA 的"
          f" {ratio:.2f}x (约 {1 / ratio:.1f} 倍)")
    print("     · 流式结构只需 2 个泄漏累加器, 不需要缓存 N 个值")
    print("  4. 时延确定性: 动态阈值 = log2(N)+常数, 与 K 无关;")
    print("     k-WTA 的串行轮数与位宽成正比, 且控制逻辑数据相关, 难深流水。")
    print()
    print("  写专利时该强调的是【免乘法器 + 流式无缓存 + 时延确定】,")
    print("  而不是【无需全局排序】—— 后者在算子数上并不成立。")
    return {name: {"area": area, "ratio": area / base, "path": path}
            for name, (ops, area, path, lv) in res.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="稀疏激活判定的硬件代价估算")
    ap.add_argument("--n", type=int, default=N, help="神经元数 (默认 2000)")
    ap.add_argument("--k", type=int, default=K, help="目标激活数 (默认 100)")
    ap.add_argument("--bits", type=int, default=W, help="数据位宽 (默认 16)")
    args = ap.parse_args(argv)
    report(args.n, args.k, args.bits, args.bits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
