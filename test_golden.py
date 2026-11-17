#!/usr/bin/env python3
"""
test_golden.py —— 纯 Python 回归测试 (不需要 iverilog)
====================================================

    python -m unittest -v test_golden

覆盖:
  * `>>>` 的算术语义 (负数向负无穷取整);
  * β 的推导是否真的落在 1.645σ;
  * 累加器不溢出、numpy 标量不悄悄回绕;
  * k-WTA: 无并列时严格 K 个 / 有并列时至少 K 个 / 退化输入;
  * 旧判据 (ge > K) 恒定多激活 1 个 —— 回归保护。
"""
import math
import unittest

import numpy as np

import sparse_activate_sim as sim


class TestSar(unittest.TestCase):
    def test_floor_on_negative(self):
        self.assertEqual(sim.sar(-1, 1), -1)
        self.assertEqual(sim.sar(-3, 1), -2)
        self.assertEqual(sim.sar(-32768, 6), -512)
        self.assertEqual(sim.sar(32767, 6), 511)

    def test_matches_floor_division(self):
        for x in range(-300, 300):
            for n in (0, 1, 4, 6):
                self.assertEqual(sim.sar(x, n), x // (1 << n))


class TestDynThreshold(unittest.TestCase):
    def test_beta_lands_on_1p645_sigma(self):
        """β·E|X-μ| 应该 ≈ 1.645σ, 这样正态输入才有 5% 激活。"""
        beta = sim.BNUM / (1 << sim.BSH)
        z = beta * math.sqrt(2 / math.pi)      # E|X-μ|/σ = sqrt(2/pi)
        self.assertAlmostEqual(beta, 2.0625, places=6)
        self.assertAlmostEqual(z, 1.645, delta=0.01)

    def test_constant_input(self):
        """恒定输入: MAD→0, 阈值→μ, 收敛后每个样本都激活 (需跑够 ~15τ)。"""
        r = sim.dyn_thr_stream([1000] * 3000)
        self.assertEqual(int(r.acts[-100:].sum()), 100)
        self.assertLess(abs(r.mu_final - 1000), 2)
        self.assertLess(r.mad_final, 2)

    def test_numpy_scalar_does_not_overflow(self):
        """int16 数组若不过 int(), 加减会按 int16 回绕, 必须和 list 结果一致。"""
        vals = np.array([32767, -32768] * 60, dtype=np.int16)
        a = sim.dyn_thr_stream(vals).acts
        b = sim.dyn_thr_stream([int(v) for v in vals]).acts
        np.testing.assert_array_equal(a, b)

    def test_accumulator_stays_in_range(self):
        r = sim.dyn_thr_stream(np.array([32767, -32768] * 200, np.int16))
        self.assertLess(r.peak_acc, sim.ACC_MAX)
        self.assertLessEqual(r.peak_acc, 1 << 22)   # 设计上界 ~65535·2^SHIFT

    def test_rate_is_amplitude_independent(self):
        d = sim.build_datasets(0)
        rates = [float(sim.dyn_thr_stream(d[k]).acts[sim.WARMUP:].mean())
                 for k in ("small", "large", "mixed")]
        self.assertLess(max(rates) - min(rates), 0.05)
        for r in rates:
            self.assertLess(r, 0.10)


class TestKwta(unittest.TestCase):
    def test_exact_k_without_ties(self):
        vals = np.arange(-50, 50, dtype=np.int16)          # 100 个互不相同的值
        r = sim.kwta_search(vals, k=7)
        self.assertEqual(r.count, 7)
        self.assertEqual(r.mid, int(np.sort(vals)[::-1][6]))   # 第 K 大的值

    def test_ties_activate_whole_tie_group(self):
        """3 个并列第一, K=2: 阈值落在 5, 激活 3 个 (至少 K, 不会少)。"""
        r = sim.kwta_search(np.array([5, 5, 5, 2, 1], dtype=np.int16), k=2)
        self.assertEqual(r.mid, 5)
        self.assertEqual(r.count, 3)

    def test_all_equal(self):
        r = sim.kwta_search(np.full(20, 7, dtype=np.int16), k=3)
        self.assertEqual((r.mid, r.count), (7, 20))

    def test_k_equals_n(self):
        vals = np.arange(10, dtype=np.int16)
        self.assertEqual(sim.kwta_search(vals, k=10).count, 10)

    def test_old_rule_was_off_by_one(self):
        """旧判据 ge > K 会收敛到 θ-1, 恒定多激活 1 个 —— 回归保护。"""
        vals = np.arange(64, dtype=np.int16)

        lo, hi = sim.VMIN, sim.VMAX
        mid = sim.sar(lo + hi, 1)
        for _ in range(sim.BW):
            ge = int(np.count_nonzero(vals >= mid))
            if ge > 4:                                  # 旧判据: ge > K
                lo = mid
            else:
                hi = mid
            mid = sim.sar(lo + hi, 1)
        old_count = int(np.count_nonzero(vals >= mid))

        self.assertEqual(old_count, 5)                     # 旧的: K+1
        self.assertEqual(sim.kwta_search(vals, k=4).count, 4)   # 新的: K


class TestData(unittest.TestCase):
    def test_clip_to_int16(self):
        v = sim.to_int16([-1e9, 1e9, 0])
        self.assertEqual(v.dtype, np.int16)
        self.assertEqual(v.tolist(), [sim.VMIN, sim.VMAX, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
