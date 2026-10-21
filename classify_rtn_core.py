# -*- coding: utf-8 -*-
"""
classify_rtn_core.py
---------------------------------
RTN分類 + RTNデータの summary 自動出力

正式版（今回の方針）:
  - find_peaks の height 条件なし
  - prominence の基準を smax（最大ピーク）から外し、percentile（代表的高さ）にする
      prom_th = max(PEAK_PROM_REL * percentile(smooth), PEAK_PROM_MIN)
  - 入力CSVの値をそのまま使用（掛け算なし）
  - ドリフト判定/除外なし
  - tau_e, tau_c は「時間」ではなく平均回数（平均連続カウント）を出力
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ===== 定数・パラメータ =====================================

# ヒストグラム bins
BINS_MIN = 100
BINS_MAX = 1000
BINS_SCALE = 2.0

# スムージング
SMOOTH_WINDOW = 17
SMOOTH_PAD = 14
SMOOTH_REPEATS = 1

# ---- prominence 基準を percentile にする ----
PROM_REF_PERCENTILE = 90  # smoothの上位○%点を基準にする（巨大ピークに引きずられにくい）

# ---- ピーク検出（ゆるめ設定） ----
PEAK_DISTANCE = 10          # 近いピークも許容
PEAK_PROM_REL = 0.05        # 相対prominence（基準は sref に対して）
PEAK_PROM_MIN = 1.5         # 絶対prominenceの下限
PEAK_WIDTH_MIN = 2          # 細いピークも許容

# ---- 2峰性の追加チェック（ゆるめ設定） ----
SEP_REL_MIN = 0.03          # ピーク間隔の下限を緩める
SEP_BIN_MIN = 1             # bin間隔条件を緩める
DIP_RATIO_MAX = 0.85        # 谷が浅くても2峰OK（非対称RTN対策）

# ===========================================================


def moving_average(x, window):
    if window <= 1 or len(x) < window:
        return x.astype(float)
    kernel = np.ones(window) / window
    return np.convolve(x.astype(float), kernel, mode="same")


def smooth_hist(freq, repeats=SMOOTH_REPEATS, pad=SMOOTH_PAD, window=SMOOTH_WINDOW):
    y = freq.astype(float).copy()
    for _ in range(repeats):
        left = np.full(pad, y[0])
        right = np.full(pad, y[-1])
        y = moving_average(np.concatenate([left, y, right]), window=window)[pad:-pad]
    return y


def build_hist(values):
    vmin, vmax = float(np.min(values)), float(np.max(values))
    span = max(vmax - vmin, 1e-9)
    bins = int(max(BINS_MIN, min(span * BINS_SCALE, BINS_MAX)))
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return hist, centers, vmin, vmax


def _top_two_by_prominence(peaks_idx, prom):
    if len(peaks_idx) <= 2:
        return peaks_idx
    order = np.argsort(prom)[::-1]
    return peaks_idx[order[:2]]


def _passes_strict_2peak_checks(centers, smooth, p1, p2, vmin, vmax):
    """
    2峰性が「それっぽい」かの追加チェック（緩め設定）。
    """
    if p1 > p2:
        p1, p2 = p2, p1

    bin_width = centers[1] - centers[0] if len(centers) > 1 else 1.0
    x_sep = abs(centers[p2] - centers[p1])
    sep_ok = (x_sep >= max(SEP_REL_MIN * (vmax - vmin), SEP_BIN_MIN * bin_width))
    if not sep_ok:
        return False

    valley_rel = np.argmin(smooth[p1:p2 + 1])
    valley_idx = p1 + valley_rel
    valley_h = smooth[valley_idx]
    dip_ratio = valley_h / max(min(smooth[p1], smooth[p2]), 1e-9)
    if dip_ratio > DIP_RATIO_MAX:
        return False

    return True


# ===== ピーク解析（peak_num, level, th） ====================
def analyze_peaks(values):
    """
    各データ列のヒストグラム解析を行い、
    peak数, level(二峰間距離), th(二峰中間値) を返す。

    変更点:
      - prominence閾値の基準を smax ではなく percentile(smooth) にする
      - height 条件なし
    """
    hist, centers, vmin, vmax = build_hist(values)
    smooth = smooth_hist(hist)

    if len(smooth) == 0:
        return 0, np.nan, np.nan

    smax = float(np.max(smooth))
    if smax <= 0:
        return 0, np.nan, np.nan

    # ★ percentile 基準（巨大ピークに引きずられない）
    sref = float(np.percentile(smooth, PROM_REF_PERCENTILE))
    if sref <= 0:
        # 念のためフォールバック（全て0に近いケース）
        sref = smax

    prom_th = max(PEAK_PROM_REL * sref, PEAK_PROM_MIN)

    peaks, props = find_peaks(
        smooth,
        distance=PEAK_DISTANCE,
        prominence=prom_th,
        width=PEAK_WIDTH_MIN
    )

    peak_num = len(peaks)
    level = np.nan
    th = np.nan

    if peak_num >= 2:
        prominences = props.get("prominences", np.ones_like(peaks))
        sel = _top_two_by_prominence(peaks, prominences)
        if len(sel) >= 2:
            p1, p2 = int(sel[0]), int(sel[1])
            if _passes_strict_2peak_checks(centers, smooth, p1, p2, vmin, vmax):
                level = abs(centers[p2] - centers[p1])
                th = (centers[p1] + centers[p2]) / 2.0

    return peak_num, level, th


def classify_peak_count(values):
    peak_num, level, th = analyze_peaks(values)
    if (peak_num >= 2) and (not np.isnan(level)) and (not np.isnan(th)):
        return "RTN", peak_num, level, th
    return "nonRTN", peak_num, np.nan, np.nan


# ===== τe / τc の計算（平均回数） ============================
def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """
    True/False の一次元配列に対して、True が連続する区間長の配列を返す。
    例: [F,T,T,F,T] -> [2,1]
    """
    if mask.size == 0:
        return np.array([], dtype=int)

    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1

    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]

    return (ends - starts).astype(int)


def compute_tau_count(values: np.ndarray, th: float, mode: str = "e") -> float:
    """
    τ（時間）ではなく平均回数/平均連続カウントを出力:
      mode="e": values >= th の連続カウント平均（回）
      mode="c": values <  th の連続カウント平均（回）
    """
    if np.isnan(th):
        return np.nan

    mask = (values >= th) if mode == "e" else (values < th)
    runs = _run_lengths(mask)
    if runs.size == 0:
        return np.nan

    return float(runs.mean())


# ===== メイン：CSV分類 =======================================
def classify_csv(filepath, output_dir="."):
    """
    CSV内の全データ列を RTN / nonRTN に分類して出力。
    さらに RTN 列のみについて、peak_num/level/th/tau_e/tau_c を
    summary CSV（*_RTN_summary.csv）として同時出力。

    注意:
      - 入力CSVの値をそのまま使用（掛け算なし）
      - ドリフト判定/除外なし
      - tau_e, tau_c は「平均回数（平均連続カウント）」を出力
    """
    df = pd.read_csv(filepath, header=0)
    if df.shape[1] < 2:
        raise ValueError("Input CSV must have at least 2 columns (time + data columns).")

    time = df.iloc[:, 0].values
    data_cols = df.columns[1:]

    groups = {"RTN": {}, "nonRTN": {}}
    summary_records = []

    for col in data_cols:
        series = df[col].dropna().to_numpy(dtype=float)

        label, peak_num, level, th = classify_peak_count(series)

        if label == "RTN":
            groups["RTN"][col] = df[col].values

            tau_e = compute_tau_count(series, th, mode="e")
            tau_c = compute_tau_count(series, th, mode="c")

            summary_records.append({
                "X": col,
                "peak_num": int(peak_num),
                "level": level,
                "th": th,
                "tau_e": tau_e,  # 平均回数（回）
                "tau_c": tau_c,  # 平均回数（回）
            })
        else:
            groups["nonRTN"][col] = df[col].values

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(filepath))[0]

    # --- RTN のみ出力（値は元のまま） ---
    if groups["RTN"]:
        out_df = pd.concat(
            [pd.Series(time, name=df.columns[0])] +
            [pd.Series(v, name=k) for k, v in groups["RTN"].items()],
            axis=1
        )
        out_path = os.path.join(output_dir, f"{base}_RTN.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path} ({len(groups['RTN'])} columns)")
    else:
        print(f"No RTN detected in {base} → Skipped.")

    # --- summary 出力（RTN列のみ） ---
    if summary_records:
        summary_df = pd.DataFrame(
            summary_records,
            columns=["X", "peak_num", "tau_c", "tau_e", "level", "th"]
        )
        out_summary = os.path.join(output_dir, f"{base}_RTN_summary.csv")
        summary_df.to_csv(out_summary, index=False)
        print(f"Saved summary: {out_summary}")

    print("Classification complete.\n")
