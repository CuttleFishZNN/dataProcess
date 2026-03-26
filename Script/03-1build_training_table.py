# 对教学视频的数据进行处理

import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# =========================================================
# 配置区：根据实际路径修改此处
# =========================================================
# 原始 prosody 全特征表
PROSODY_CSV = Path(r"E:\Project\Python\dataProcess\Script\data\math\llm_reseg_fuzzy\prosody_features.csv")

# 切分后的文本文件（优先用 jsonl）
SEGMENT_JSONL = Path(r"E:\Project\Python\dataProcess\Script\data\math\llm_reseg_fuzzy\llm_resegmented_fuzzy.jsonl")

# 如果你更想用 json，也可以填这个；不用则保持 None
SEGMENT_JSON = None
# 示例配置：
# SEGMENT_JSON = Path(r"E:\Project\Python\dataProcess\data\math\llm_reseg_fuzzy\llm_resegmented_fuzzy.json")

# 输出目录
OUT_DIR = Path(r"E:\Project\Python\dataProcess\Script\data\math\llm_reseg_fuzzy")

# =========================================================
# Step 1: 选取精简版 prosody 特征
# =========================================================
KEEP_PROSODY_COLS = [
    "seg_id",
    "audio_path",
    # ===== 音高（pitch）相关 =====
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
    "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
    "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
    # ===== 响度（loudness）相关 =====
    "loudness_sma3_amean",
    "loudness_sma3_stddevNorm",
    "loudness_sma3_percentile20.0",
    "loudness_sma3_percentile50.0",
    "loudness_sma3_percentile80.0",
    "loudness_sma3_pctlrange0-2",
    "loudness_sma3_meanRisingSlope",
    "loudness_sma3_meanFallingSlope",
    "equivalentSoundLevel_dBp",
    # ===== 节奏/说话组织（rhythm/speaking organization）相关 =====
    "loudnessPeaksPerSec",
    "VoicedSegmentsPerSec",
    "MeanVoicedSegmentLengthSec",
    "StddevVoicedSegmentLengthSec",
    "MeanUnvoicedSegmentLength",
    "StddevUnvoicedSegmentLength",
    # ===== 动态变化（dynamic change）相关 =====
    "spectralFlux_sma3_amean",
    "spectralFlux_sma3_stddevNorm",
]


# =========================================================
# 工具函数
# =========================================================
def load_segment_rows_from_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """从 jsonl 文件读取分段数据"""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_segment_rows_from_json(json_path: Path) -> List[Dict[str, Any]]:
    """从 json 文件读取分段数据"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "segments" in data and isinstance(data["segments"], list):
        return data["segments"]
    if isinstance(data, list):
        return data
    raise ValueError("无法从 JSON 中解析 segment 列表。")


def normalize_segment_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    统一整理 segment 元数据，并补充 duration_sec 字段
    """
    clean_rows = []
    for r in rows:
        seg_id = r.get("seg_id")
        if seg_id is None:
            continue

        start = r.get("start")
        end = r.get("end")
        duration = r.get("duration")

        # 若未直接提供时长，则通过 start/end 计算
        if duration is None and start is not None and end is not None:
            duration = float(end) - float(start)

        clean_rows.append({
            "seg_id": seg_id,
            "orig_segment_id": r.get("orig_segment_id"),
            "text": r.get("text"),
            "orig_text_span": r.get("orig_text_span"),
            "start": float(start) if start is not None else None,
            "end": float(end) if end is not None else None,
            "duration_sec": float(duration) if duration is not None else None,
        })

    df = pd.DataFrame(clean_rows)
    df = df.drop_duplicates(subset=["seg_id"]).reset_index(drop=True)
    return df


def save_jsonl(df: pd.DataFrame, out_path: Path) -> None:
    """将 DataFrame 保存为 jsonl 格式文件"""
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


# =========================================================
# 主流程
# =========================================================
def main():
    # 确保输出目录存在
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Step 1: 生成精简版 prosody 特征表
    # -------------------------
    print("Step 1/3: 读取并压缩 prosody 特征表 ...", flush=True)
    prosody_df = pd.read_csv(PROSODY_CSV)

    # 检查必要列是否存在
    missing_cols = [c for c in KEEP_PROSODY_COLS if c not in prosody_df.columns]
    if missing_cols:
        raise ValueError(f"prosody_features.csv 缺少这些列：{missing_cols}")

    # 筛选精简特征并去重
    prosody_compact = prosody_df[KEEP_PROSODY_COLS].copy()
    prosody_compact = prosody_compact.drop_duplicates(subset=["seg_id"]).reset_index(drop=True)

    # 保存精简版特征表
    prosody_compact_csv = OUT_DIR / "prosody_compact.csv"
    prosody_compact.to_csv(prosody_compact_csv, index=False, encoding="utf-8-sig")
    print(f"已保存精简版 prosody: {prosody_compact_csv}", flush=True)

    # -------------------------
    # Step 2: 读取文本分段并补充时长
    # -------------------------
    print("Step 2/3: 读取切分后的文本 JSON/JSONL，并补 duration ...", flush=True)
    if SEGMENT_JSONL is not None and SEGMENT_JSONL.exists():
        segment_rows = load_segment_rows_from_jsonl(SEGMENT_JSONL)
    elif SEGMENT_JSON is not None and Path(SEGMENT_JSON).exists():
        segment_rows = load_segment_rows_from_json(Path(SEGMENT_JSON))
    else:
        raise FileNotFoundError("没有找到 segment jsonl/json 文件，请检查路径配置。")

    # 整理元数据并补充时长
    segment_meta = normalize_segment_rows(segment_rows)

    # 保存分段元数据表
    segment_meta_csv = OUT_DIR / "segment_meta.csv"
    segment_meta.to_csv(segment_meta_csv, index=False, encoding="utf-8-sig")
    print(f"已保存 segment 元数据: {segment_meta_csv}", flush=True)

    # -------------------------
    # Step 3: 按 seg_id 合并所有数据
    # -------------------------
    print("Step 3/3: 按 seg_id merge 文本与 prosody ...", flush=True)
    merged = pd.merge(
        segment_meta,
        prosody_compact,
        on="seg_id",
        how="left"
    )

    # 调整列顺序，核心信息前置
    front_cols = [
        "seg_id", "orig_segment_id", "text", "orig_text_span",
        "start", "end", "duration_sec", "audio_path"
    ]
    rest_cols = [c for c in merged.columns if c not in front_cols]
    merged = merged[front_cols + rest_cols]

    # 保存最终总表（CSV + JSONL 双格式）
    training_csv = OUT_DIR / "training_table.csv"
    training_jsonl = OUT_DIR / "training_table.jsonl"
    merged.to_csv(training_csv, index=False, encoding="utf-8-sig")
    save_jsonl(merged, training_jsonl)

    print(f"已保存最终训练总表 CSV: {training_csv}", flush=True)
    print(f"已保存最终训练总表 JSONL: {training_jsonl}", flush=True)

    # -------------------------
    # 打印统计信息（验证数据完整性）
    # -------------------------
    print("\n===== 统计信息 =====")
    print(f"prosody 原始条数: {len(prosody_df)}")
    print(f"prosody 精简条数: {len(prosody_compact)}")
    print(f"segment 条数: {len(segment_meta)}")
    print(f"merge 后条数: {len(merged)}")

    matched = merged["audio_path"].notna().sum()
    print(f"成功匹配到 prosody 的条数: {matched}")
    print(f"未匹配到 prosody 的条数: {len(merged) - matched}")

    print("\n===== 前 3 行预览 =====")
    print(merged.head(3).to_string())


if __name__ == "__main__":
    main()