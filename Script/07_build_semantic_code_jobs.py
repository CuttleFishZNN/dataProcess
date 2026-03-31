from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# =========================
# 配置区：直接改这里即可
# =========================
PROJECT_ROOT = Path(r"E:\Project\Python\dataProcess")

PROSODY_JSONL = PROJECT_ROOT / r"dataset\math\T01\MATH_T01_L01\anno\prosody_summary.jsonl"

ACTION_JSONL = PROJECT_ROOT / r"dataset\math\T01\MATH_T01_L01\anno\action_evidence_qwen25vl0000-qwen3.5-plus.jsonl"

RESEG_JSON = PROJECT_ROOT / r"dataset\math\T01\MATH_T01_L01\anno\llm_resegmented_fuzzy.json"

# 原始路径目录（用于补全母表里的音频/视频文件路径）
CLIPS_DIR = PROJECT_ROOT / r"dataset\math\T01\MATH_T01_L01\raw\clips"
AUDIO_DIR = PROJECT_ROOT / r"dataset\math\T01\MATH_T01_L01\raw\audio"

# 输出：总数据母表（可索引）
OUT_MASTER_JSONL = PROJECT_ROOT / r"dataset\math\T01\MATH_T01_L01\anno\semantic_code_master.jsonl"
OUT_MASTER_CSV = PROJECT_ROOT / r"dataset\math\T01\MATH_T01_L01\anno\semantic_code_master.csv"

# 输出：训练输入表（不带路径，但保留 start/end/duration）
OUT_TRAIN_JSONL = PROJECT_ROOT / r"dataset\math\T01\MATH_T01_L01\anno\semantic_code_train_input.jsonl"
OUT_TRAIN_CSV = PROJECT_ROOT / r"dataset\math\T01\MATH_T01_L01\anno\semantic_code_train_input.csv"

SUBJECT = "math"
TEACHER_ID = "T01"
LESSON_ID = "MATH_T01_L01"

# 只使用 action status = ok 的样本
ONLY_ACTION_OK = True


# =========================
# 基础工具
# =========================
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_reseg_json(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到重分段文件: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    segs = data.get("segments", [])
    out: Dict[str, Dict[str, Any]] = {}
    for seg in segs:
        seg_id = seg.get("seg_id")
        if seg_id:
            out[seg_id] = seg
    return out


def safe_get_clip_path(action_row: Dict[str, Any], seg_id: str) -> str:
    clip_path = str(action_row.get("clip_path", "")).strip()
    if clip_path:
        p = Path(clip_path)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        else:
            p = p.resolve()
        return str(p)
    # 兜底
    return str((CLIPS_DIR / f"{seg_id}.mp4").resolve())


def safe_get_audio_path(seg_id: str) -> str:
    return str((AUDIO_DIR / f"{seg_id}.wav").resolve())


def flatten_master_row(row: Dict[str, Any]) -> Dict[str, Any]:
    prosody = row.get("audio_prosody", {})
    action = row.get("video_action_evidence", {})

    flat = {
        "seg_id": row.get("seg_id", ""),
        "text": row.get("text", ""),
        "subject": row.get("subject", ""),
        "teacher_id": row.get("teacher_id", ""),
        "lesson_id": row.get("lesson_id", ""),
        "audio_path": row.get("audio_path", ""),
        "clip_path": row.get("clip_path", ""),
        "start": row.get("start", ""),
        "end": row.get("end", ""),
        "duration": row.get("duration", ""),

        "speech_rate_level": prosody.get("speech_rate_level", ""),
        "pause_style": prosody.get("pause_style", ""),
        "energy_level": prosody.get("energy_level", ""),
        "energy_variation": prosody.get("energy_variation", ""),
        "pitch_level": prosody.get("pitch_level", ""),
        "pitch_range": prosody.get("pitch_range", ""),
        "intonation_tendency": prosody.get("intonation_tendency", ""),
        "voicing_continuity": prosody.get("voicing_continuity", ""),
        "prosody_note": prosody.get("prosody_note", ""),

        "body_orientation": action.get("body_orientation", ""),
        "head_orientation": action.get("head_orientation", ""),
        "posture": action.get("posture", ""),
        "locomotion": action.get("locomotion", ""),
        "board_writing": action.get("board_interaction", {}).get("writing", ""),
        "board_erasing": action.get("board_interaction", {}).get("erasing", ""),
        "board_touching": action.get("board_interaction", {}).get("touching_board", ""),
        "facing_students": action.get("student_interaction_cues", {}).get("facing_students", ""),
        "addressing_students_visually": action.get("student_interaction_cues", {}).get("addressing_students_visually", ""),
        "inviting_attention": action.get("student_interaction_cues", {}).get("inviting_attention", ""),
        "upper_body_occluded": action.get("occlusion", {}).get("upper_body_occluded", ""),
        "lower_body_occluded": action.get("occlusion", {}).get("lower_body_occluded", ""),
        "desk_occlusion": action.get("occlusion", {}).get("desk_occlusion", ""),

        "visible_body_parts": json.dumps(action.get("visible_body_parts", []), ensure_ascii=False),
        "arm_actions": json.dumps(action.get("arm_actions", []), ensure_ascii=False),
        "hand_actions": json.dumps(action.get("hand_actions", []), ensure_ascii=False),
        "object_interaction": json.dumps(action.get("object_interaction", []), ensure_ascii=False),
        "evidence_sentences": json.dumps(action.get("evidence_sentences", []), ensure_ascii=False),
        "uncertain_parts": json.dumps(action.get("uncertain_parts", []), ensure_ascii=False),
    }
    return flat


def flatten_train_row(row: Dict[str, Any]) -> Dict[str, Any]:
    prosody = row.get("audio_prosody", {})
    action = row.get("video_action_evidence", {})

    flat = {
        "seg_id": row.get("seg_id", ""),
        "text": row.get("text", ""),
        "subject": row.get("subject", ""),
        "teacher_id": row.get("teacher_id", ""),
        "lesson_id": row.get("lesson_id", ""),
        "start": row.get("start", ""),
        "end": row.get("end", ""),
        "duration": row.get("duration", ""),

        "speech_rate_level": prosody.get("speech_rate_level", ""),
        "pause_style": prosody.get("pause_style", ""),
        "energy_level": prosody.get("energy_level", ""),
        "energy_variation": prosody.get("energy_variation", ""),
        "pitch_level": prosody.get("pitch_level", ""),
        "pitch_range": prosody.get("pitch_range", ""),
        "intonation_tendency": prosody.get("intonation_tendency", ""),
        "voicing_continuity": prosody.get("voicing_continuity", ""),
        "prosody_note": prosody.get("prosody_note", ""),

        "body_orientation": action.get("body_orientation", ""),
        "head_orientation": action.get("head_orientation", ""),
        "posture": action.get("posture", ""),
        "locomotion": action.get("locomotion", ""),
        "board_writing": action.get("board_interaction", {}).get("writing", ""),
        "board_erasing": action.get("board_interaction", {}).get("erasing", ""),
        "board_touching": action.get("board_interaction", {}).get("touching_board", ""),
        "facing_students": action.get("student_interaction_cues", {}).get("facing_students", ""),
        "addressing_students_visually": action.get("student_interaction_cues", {}).get("addressing_students_visually", ""),
        "inviting_attention": action.get("student_interaction_cues", {}).get("inviting_attention", ""),

        "visible_body_parts": json.dumps(action.get("visible_body_parts", []), ensure_ascii=False),
        "arm_actions": json.dumps(action.get("arm_actions", []), ensure_ascii=False),
        "hand_actions": json.dumps(action.get("hand_actions", []), ensure_ascii=False),
        "object_interaction": json.dumps(action.get("object_interaction", []), ensure_ascii=False),
        "evidence_sentences": json.dumps(action.get("evidence_sentences", []), ensure_ascii=False),
        "uncertain_parts": json.dumps(action.get("uncertain_parts", []), ensure_ascii=False),
    }
    return flat


# =========================
# 主流程
# =========================
def main() -> None:
    prosody_rows = read_jsonl(PROSODY_JSONL)
    action_rows = read_jsonl(ACTION_JSONL)
    reseg_map = read_reseg_json(RESEG_JSON)

    prosody_map: Dict[str, Dict[str, Any]] = {}
    for row in prosody_rows:
        seg_id = row.get("seg_id")
        if seg_id:
            prosody_map[seg_id] = row

    action_map: Dict[str, Dict[str, Any]] = {}
    for row in action_rows:
        seg_id = row.get("seg_id")
        if not seg_id:
            continue
        if ONLY_ACTION_OK and row.get("status") != "ok":
            continue
        action_map[seg_id] = row

    seg_ids = sorted(set(prosody_map.keys()) & set(action_map.keys()) & set(reseg_map.keys()))

    print(f"[INFO] prosody 条数: {len(prosody_map)}")
    print(f"[INFO] action(ok) 条数: {len(action_map)}")
    print(f"[INFO] reseg 条数: {len(reseg_map)}")
    print(f"[INFO] 可合并 seg_id 数: {len(seg_ids)}")

    master_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []

    master_flat_rows: List[Dict[str, Any]] = []
    train_flat_rows: List[Dict[str, Any]] = []

    for seg_id in seg_ids:
        p = prosody_map[seg_id]
        a = action_map[seg_id]
        r = reseg_map[seg_id]

        text = a.get("text", "") or r.get("text", "")
        clip_path = safe_get_clip_path(a, seg_id)
        audio_path = safe_get_audio_path(seg_id)
        action_evidence = a.get("action_evidence", {})

        audio_prosody = {
            "speech_rate_level": p.get("speech_rate_level", "unknown"),
            "pause_style": p.get("pause_style", "unknown"),
            "energy_level": p.get("energy_level", "unknown"),
            "energy_variation": p.get("energy_variation", "unknown"),
            "pitch_level": p.get("pitch_level", "unknown"),
            "pitch_range": p.get("pitch_range", "unknown"),
            "intonation_tendency": p.get("intonation_tendency", "unknown"),
            "voicing_continuity": p.get("voicing_continuity", "unknown"),
            "prosody_note": p.get("prosody_note", ""),
        }

        # 1) 总数据母表（可索引）
        master_row = {
            "seg_id": seg_id,
            "subject": SUBJECT,
            "teacher_id": TEACHER_ID,
            "lesson_id": LESSON_ID,

            "text": text,
            "audio_path": audio_path,
            "clip_path": clip_path,

            "start": r.get("start", None),
            "end": r.get("end", None),
            "duration": r.get("duration", None),

            "audio_prosody": audio_prosody,
            "video_action_evidence": action_evidence,
        }
        master_rows.append(master_row)
        master_flat_rows.append(flatten_master_row(master_row))

        # 2) 训练输入表（保留时间，不保留路径）
        train_row = {
            "seg_id": seg_id,
            "subject": SUBJECT,
            "teacher_id": TEACHER_ID,
            "lesson_id": LESSON_ID,

            "text": text,
            "start": r.get("start", None),
            "end": r.get("end", None),
            "duration": r.get("duration", None),

            "audio_prosody": audio_prosody,
            "video_action_evidence": action_evidence,

            # 后面给大模型推教学语义码时可以直接用这个字段
            "llm_inference_input": {
                "task": "infer_teaching_semantic_code",
                "text": text,
                "start": r.get("start", None),
                "end": r.get("end", None),
                "duration": r.get("duration", None),
                "audio_prosody": audio_prosody,
                "video_action_evidence": action_evidence,
            },
        }
        train_rows.append(train_row)
        train_flat_rows.append(flatten_train_row(train_row))

    # 写 master
    write_jsonl(OUT_MASTER_JSONL, master_rows)
    pd.DataFrame(master_flat_rows).to_csv(OUT_MASTER_CSV, index=False, encoding="utf-8-sig")

    # 写 train
    write_jsonl(OUT_TRAIN_JSONL, train_rows)
    pd.DataFrame(train_flat_rows).to_csv(OUT_TRAIN_CSV, index=False, encoding="utf-8-sig")

    print(f"[DONE] 总数据母表 JSONL: {OUT_MASTER_JSONL}")
    print(f"[DONE] 总数据母表 CSV:   {OUT_MASTER_CSV}")
    print(f"[DONE] 训练输入表 JSONL: {OUT_TRAIN_JSONL}")
    print(f"[DONE] 训练输入表 CSV:   {OUT_TRAIN_CSV}")
    print(f"[DONE] 样本数: {len(master_rows)}")


if __name__ == "__main__":
    main()