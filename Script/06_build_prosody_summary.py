# 对提取出来的 音频特征 进行 格式化处理（将抽象的数字转化为可理解的文本）
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# =========================
# 配置区（可直接修改路径）
# =========================
INPUT_CSV = Path(r"E:\Project\Python\dataProcess\dataset\math\T01\MATH_T01_L01\anno\prosody_compact.csv")
OUT_JSONL = Path(r"E:\Project\Python\dataProcess\dataset\math\T01\MATH_T01_L01\anno\prosody_summary.jsonl")
OUT_CSV   = Path(r"E:\Project\Python\dataProcess\dataset\math\T01\MATH_T01_L01\anno\prosody_summary.csv")


# =========================
# 工具函数
# =========================
def compute_tertile_thresholds(series: pd.Series) -> Tuple[float, float]:
    """计算三分位数阈值（33% / 67%）"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    q1 = s.quantile(1 / 3)
    q2 = s.quantile(2 / 3)
    return float(q1), float(q2)


def bucket_3(x: float, q1: float, q2: float, labels=("low", "medium", "high")) -> str:
    """三分位分级"""
    if pd.isna(x):
        return "unknown"
    if x <= q1:
        return labels[0]
    elif x <= q2:
        return labels[1]
    else:
        return labels[2]


def make_intonation_label(rising: float, falling: float, rise_q: float, fall_q: float) -> str:
    """生成语调趋势标签"""
    if pd.isna(rising) or pd.isna(falling):
        return "unknown"

    strong_rise = rising > rise_q
    strong_fall = falling > fall_q

    if strong_rise and not strong_fall:
        return "rising_dominant"
    elif strong_fall and not strong_rise:
        return "falling_dominant"
    elif strong_rise and strong_fall:
        return "mixed"
    else:
        return "flat"


def make_pause_style(mean_unvoiced: float, std_unvoiced: float, q_mean: Tuple[float, float], q_std: Tuple[float, float]) -> str:
    """生成停顿风格标签"""
    if pd.isna(mean_unvoiced) or pd.isna(std_unvoiced):
        return "unknown"

    mean_label = bucket_3(mean_unvoiced, q_mean[0], q_mean[1], labels=("few_pauses", "moderate_pauses", "frequent_pauses"))
    std_label = bucket_3(std_unvoiced, q_std[0], q_std[1], labels=("stable_pauses", "moderate_pauses", "variable_pauses"))

    # 主参考平均无声时长
    if mean_label == "few_pauses":
        return "few_pauses"
    elif mean_label == "frequent_pauses":
        return "frequent_pauses"
    else:
        return "moderate_pauses"


def make_voicing_continuity(voiced_per_sec: float, mean_voiced_len: float, mean_unvoiced_len: float,
                            q_vps: Tuple[float, float], q_mvl: Tuple[float, float], q_mul: Tuple[float, float]) -> str:
    """生成发声连贯性标签"""
    if pd.isna(voiced_per_sec) or pd.isna(mean_voiced_len) or pd.isna(mean_unvoiced_len):
        return "unknown"

    vps_label = bucket_3(voiced_per_sec, q_vps[0], q_vps[1], labels=("sparse", "moderate", "dense"))
    mvl_label = bucket_3(mean_voiced_len, q_mvl[0], q_mvl[1], labels=("short", "medium", "long"))
    mul_label = bucket_3(mean_unvoiced_len, q_mul[0], q_mul[1], labels=("short_gap", "medium_gap", "long_gap"))

    if mvl_label == "long" and mul_label == "short_gap":
        return "continuous"
    elif mvl_label == "short" and mul_label == "long_gap":
        return "fragmented"
    else:
        return "moderate"


def build_prosody_note(row: Dict[str, str]) -> str:
    """生成自然语言韵律说明"""
    parts: List[str] = []

    sr = row["speech_rate_level"]
    ps = row["pause_style"]
    en = row["energy_level"]
    ev = row["energy_variation"]
    pl = row["pitch_level"]
    pr = row["pitch_range"]
    it = row["intonation_tendency"]

    if sr != "unknown":
        parts.append(f"语速{ {'low':'较慢','medium':'中等','high':'较快'}.get(sr, sr) }")
    if ps != "unknown":
        parts.append(f"停顿{ {'few_pauses':'较少','moderate_pauses':'适中','frequent_pauses':'较多'}.get(ps, ps) }")
    if en != "unknown":
        parts.append(f"响度{ {'low':'较低','medium':'中等','high':'较高'}.get(en, en) }")
    if ev != "unknown":
        parts.append(f"响度起伏{ {'stable':'较平稳','medium':'中等','dynamic':'较明显'}.get(ev, ev) }")
    if pl != "unknown":
        parts.append(f"音高{ {'low':'偏低','medium':'中等','high':'偏高'}.get(pl, pl) }")
    if pr != "unknown":
        parts.append(f"音高变化范围{ {'narrow':'较窄','medium':'中等','wide':'较宽'}.get(pr, pr) }")

    if it == "rising_dominant":
        parts.append("整体上扬趋势较明显")
    elif it == "falling_dominant":
        parts.append("整体下行趋势较明显")
    elif it == "mixed":
        parts.append("升降变化都较明显")
    elif it == "flat":
        parts.append("整体语调较平")

    return "，".join(parts) + "。" if parts else "韵律信息不足。"


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"找不到输入文件: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # 全局分位数计算
    q_pitch_level = compute_tertile_thresholds(df["F0semitoneFrom27.5Hz_sma3nz_percentile50.0"])
    q_pitch_range_1 = compute_tertile_thresholds(df["F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2"])
    q_pitch_range_2 = compute_tertile_thresholds(df["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"])

    q_energy_level_1 = compute_tertile_thresholds(df["equivalentSoundLevel_dBp"])
    q_energy_level_2 = compute_tertile_thresholds(df["loudness_sma3_amean"])

    q_energy_var_1 = compute_tertile_thresholds(df["loudness_sma3_stddevNorm"])
    q_energy_var_2 = compute_tertile_thresholds(df["loudness_sma3_pctlrange0-2"])
    q_energy_var_3 = compute_tertile_thresholds(df["loudnessPeaksPerSec"])
    q_energy_var_4 = compute_tertile_thresholds(df["spectralFlux_sma3_stddevNorm"])

    q_speech_rate = compute_tertile_thresholds(df["VoicedSegmentsPerSec"])
    q_pause_mean = compute_tertile_thresholds(df["MeanUnvoicedSegmentLength"])
    q_pause_std = compute_tertile_thresholds(df["StddevUnvoicedSegmentLength"])

    q_voiced_per_sec = compute_tertile_thresholds(df["VoicedSegmentsPerSec"])
    q_mean_voiced = compute_tertile_thresholds(df["MeanVoicedSegmentLengthSec"])
    q_mean_unvoiced = compute_tertile_thresholds(df["MeanUnvoicedSegmentLength"])

    rise_q = float(df["F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope"].quantile(0.67))
    fall_q = float(df["F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope"].quantile(0.67))

    outputs = []

    # 逐行处理
    for _, r in df.iterrows():
        seg_id = r["seg_id"]

        # 1. 语速等级
        speech_rate_level = bucket_3(
            r["VoicedSegmentsPerSec"],
            q_speech_rate[0],
            q_speech_rate[1],
            labels=("slow", "medium", "fast"),
        )

        # 2. 停顿风格
        pause_style = make_pause_style(
            r["MeanUnvoicedSegmentLength"],
            r["StddevUnvoicedSegmentLength"],
            q_pause_mean,
            q_pause_std,
        )

        # 3. 响度等级（双特征综合）
        e1 = bucket_3(r["equivalentSoundLevel_dBp"], q_energy_level_1[0], q_energy_level_1[1])
        e2 = bucket_3(r["loudness_sma3_amean"], q_energy_level_2[0], q_energy_level_2[1])
        energy_level = e1 if e1 == e2 else "medium"

        # 4. 响度起伏（多特征投票）
        ev_scores = []
        for val, q in [
            (r["loudness_sma3_stddevNorm"], q_energy_var_1),
            (r["loudness_sma3_pctlrange0-2"], q_energy_var_2),
            (r["loudnessPeaksPerSec"], q_energy_var_3),
            (r["spectralFlux_sma3_stddevNorm"], q_energy_var_4),
        ]:
            label = bucket_3(val, q[0], q[1], labels=("stable", "medium", "dynamic"))
            ev_scores.append(label)

        if ev_scores.count("dynamic") >= 2:
            energy_variation = "dynamic"
        elif ev_scores.count("stable") >= 3:
            energy_variation = "stable"
        else:
            energy_variation = "medium"

        # 5. 音高水平
        pitch_level = bucket_3(
            r["F0semitoneFrom27.5Hz_sma3nz_percentile50.0"],
            q_pitch_level[0],
            q_pitch_level[1],
        )

        # 6. 音高范围（双特征综合）
        p1 = bucket_3(r["F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2"], q_pitch_range_1[0], q_pitch_range_1[1], labels=("narrow", "medium", "wide"))
        p2 = bucket_3(r["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"], q_pitch_range_2[0], q_pitch_range_2[1], labels=("narrow", "medium", "wide"))
        pitch_range = p1 if p1 == p2 else "medium"

        # 7. 语调趋势
        intonation_tendency = make_intonation_label(
            r["F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope"],
            r["F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope"],
            rise_q,
            fall_q,
        )

        # 8. 发声连贯性
        voicing_continuity = make_voicing_continuity(
            r["VoicedSegmentsPerSec"],
            r["MeanVoicedSegmentLengthSec"],
            r["MeanUnvoicedSegmentLength"],
            q_voiced_per_sec,
            q_mean_voiced,
            q_mean_unvoiced,
        )

        # 组装结果
        row = {
            "seg_id": seg_id,
            "speech_rate_level": speech_rate_level,
            "pause_style": pause_style,
            "energy_level": energy_level,
            "energy_variation": energy_variation,
            "pitch_level": pitch_level,
            "pitch_range": pitch_range,
            "intonation_tendency": intonation_tendency,
            "voicing_continuity": voicing_continuity,
        }
        row["prosody_note"] = build_prosody_note(row)
        outputs.append(row)

    # 输出文件
    out_df = pd.DataFrame(outputs)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[完成] CSV 输出: {OUT_CSV}")
    print(f"[完成] JSONL 输出: {OUT_JSONL}")
    print(f"[完成] 处理样本数: {len(outputs)}")


if __name__ == "__main__":
    main()