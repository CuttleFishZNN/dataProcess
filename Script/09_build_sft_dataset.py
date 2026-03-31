from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List


# =========================
# 配置区：直接改这里即可
# =========================
# 输入：推理结果总表
INPUT_JSONL = Path(r"E:\Project\Python\dataProcess\dataset\math\T01\MATH_T01_L01\Result\semantic_code_results_qwen3.5plus.jsonl")

# 输出：SFT数据集目录
OUT_DIR = Path(r"E:\Project\Python\dataProcess\dataset\math\T01\MATH_T01_L01\SFT")
OUT_TRAIN_JSON = OUT_DIR / "train_alpaca.json"
OUT_VAL_JSON = OUT_DIR / "val_alpaca.json"

# 可选：导出扁平化人工检查文件
OUT_DEBUG_JSONL = OUT_DIR / "debug_sft_samples.jsonl"

# 数据集拆分配置
VAL_RATIO = 0.1
RANDOM_SEED = 42

# 训练输入可选配置
KEEP_TIME_INFO = True       # 是否保留start/end/duration时间字段进训练输入
KEEP_PROSODY_NOTE = True   # 是否在输入里保留prosody_note韵律备注


# =========================
# 固定系统提示词
# =========================
SYSTEM_PROMPT = (
    "你是一个课堂教学语义码标注器。"
    "请根据教学文本和音频韵律摘要，输出固定 JSON。"
    "不要输出解释，不要输出额外文字，只输出 JSON。"
)

INSTRUCTION = (
    "请根据“教学文本”和“音频韵律摘要”推断教学语义码。"
    "输出 JSON，字段必须包括："
    "intent, primitive, rhythm, interaction_flag, reference_flag, "
    "reference_target, body_orientation, writing_state, sentence_independence。"
)


# =========================
# 基础工具函数
# =========================
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取jsonl格式的推理结果文件"""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    """写入json格式数据集"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """写入jsonl格式调试文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def deduplicate_keep_last_ok(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    同seg_id去重逻辑：
    1. 优先保留最后一条status=ok的记录
    2. 无ok记录时，保留最后一条记录
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        seg_id = row.get("seg_id")
        if not seg_id:
            continue
        grouped.setdefault(seg_id, []).append(row)

    final_rows: List[Dict[str, Any]] = []
    for seg_id, items in grouped.items():
        ok_items = [x for x in items if x.get("status") == "ok"]
        final_rows.append(ok_items[-1] if ok_items else items[-1])

    final_rows.sort(key=lambda x: x["seg_id"])
    return final_rows


def build_input_text(item: Dict[str, Any]) -> str:
    """构建训练输入文本：仅保留文本+音频+可选时间信息，绝对不含视频证据"""
    payload = item["input_payload"]
    text = payload.get("text", "")
    prosody = payload.get("audio_prosody", {})

    lines = []
    # 1. 核心教学文本
    lines.append(f"教学文本：{text}")

    # 2. 可选时间信息
    if KEEP_TIME_INFO:
        start = payload.get("start", None)
        end = payload.get("end", None)
        duration = payload.get("duration", None)
        lines.append("")
        lines.append("时间信息：")
        lines.append(f"- start: {start}")
        lines.append(f"- end: {end}")
        lines.append(f"- duration: {duration}")

    # 3. 音频韵律摘要
    lines.append("")
    lines.append("音频韵律摘要：")
    lines.append(f"- speech_rate_level: {prosody.get('speech_rate_level', 'unknown')}")
    lines.append(f"- pause_style: {prosody.get('pause_style', 'unknown')}")
    lines.append(f"- energy_level: {prosody.get('energy_level', 'unknown')}")
    lines.append(f"- energy_variation: {prosody.get('energy_variation', 'unknown')}")
    lines.append(f"- pitch_level: {prosody.get('pitch_level', 'unknown')}")
    lines.append(f"- pitch_range: {prosody.get('pitch_range', 'unknown')}")
    lines.append(f"- intonation_tendency: {prosody.get('intonation_tendency', 'unknown')}")
    lines.append(f"- voicing_continuity: {prosody.get('voicing_continuity', 'unknown')}")

    # 4. 可选韵律备注
    if KEEP_PROSODY_NOTE:
        lines.append(f"- prosody_note: {prosody.get('prosody_note', '')}")

    return "\n".join(lines)


def build_output_text(item: Dict[str, Any]) -> str:
    """构建训练目标：仅保留模型需要学会输出的核心字段，剔除置信度/推理说明"""
    sc = item["semantic_code"]
    target = {
        "intent": sc["intent"],
        "primitive": sc["primitive"],
        "rhythm": sc["rhythm"],
        "interaction_flag": sc["interaction_flag"],
        "reference_flag": sc["reference_flag"],
        "reference_target": sc["reference_target"],
        "body_orientation": sc["body_orientation"],
        "writing_state": sc["writing_state"],
        "sentence_independence": sc["sentence_independence"],
    }
    return json.dumps(target, ensure_ascii=False)


def convert_to_alpaca(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把推理结果转换成LLaMA-Factory支持的alpaca格式"""
    dataset: List[Dict[str, Any]] = []
    for row in rows:
        # 仅保留推理成功的样本
        if row.get("status") != "ok":
            continue
        if "semantic_code" not in row or "input_payload" not in row:
            continue

        # 标准alpaca格式
        sample = {
            "system": SYSTEM_PROMPT,
            "instruction": INSTRUCTION,
            "input": build_input_text(row),
            "output": build_output_text(row),
        }
        dataset.append(sample)
    return dataset


def build_debug_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """构建人工检查用的扁平化样本"""
    debug_rows = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        debug_rows.append({
            "seg_id": row.get("seg_id", ""),
            "text": row.get("text", ""),
            "input_text": build_input_text(row),
            "output_text": build_output_text(row),
        })
    return debug_rows


# =========================
# 主执行流程
# =========================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 读取并清洗数据
    raw_rows = read_jsonl(INPUT_JSONL)
    dedup_rows = deduplicate_keep_last_ok(raw_rows)
    dataset = convert_to_alpaca(dedup_rows)

    if not dataset:
        raise ValueError("没有可用的 status=ok 样本，无法构建训练集。")

    # 2. 拆分训练集/验证集
    random.seed(RANDOM_SEED)
    random.shuffle(dataset)
    n_total = len(dataset)
    n_val = max(1, int(n_total * VAL_RATIO))
    val_set = dataset[:n_val]
    train_set = dataset[n_val:]

    # 3. 写入文件
    write_json(OUT_TRAIN_JSON, train_set)
    write_json(OUT_VAL_JSON, val_set)
    debug_rows = build_debug_rows(dedup_rows)
    write_jsonl(OUT_DEBUG_JSONL, debug_rows)

    # 4. 输出统计信息
    print(f"[DONE] 原始记录数: {len(raw_rows)}")
    print(f"[DONE] 去重后记录数: {len(dedup_rows)}")
    print(f"[DONE] 可训练样本数: {n_total}")
    print(f"[DONE] 训练集: {len(train_set)} -> {OUT_TRAIN_JSON}")
    print(f"[DONE] 验证集: {len(val_set)} -> {OUT_VAL_JSON}")
    print(f"[DONE] 调试样本: {OUT_DEBUG_JSONL}")


if __name__ == "__main__":
    main()