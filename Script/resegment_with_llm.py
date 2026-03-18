import json
import os
import re
import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# =========================
# 配置区：直接改这里即可
# =========================
JSON_PATH = Path("teacher_class_whisperx_full.json")
VIDEO_PATH = Path(r"E:\Project\Python\dataProcess\data\math\test_5min.mp4")
OUT_DIR = Path(r"data/math/llm_reseg_fuzzy")

MODEL_NAME = "qwen3.5-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 在脚本同目录创建 dashscope_key.txt，里面只放一行 API key
KEY_FILE = Path("dashscope_key.txt")

# 切片前后额外补一点点边界
PAD_BEFORE = 0.10
PAD_AFTER = 0.15

# 若 LLM 分段结果太怪，是否回退为原始整段
FALLBACK_TO_ORIGINAL_SEGMENT = True


# =========================
# 文本工具
# =========================
KEEP_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def normalize_text_for_match(text: str) -> str:
    """只保留中英文和数字，用于模糊对齐。"""
    if text is None:
        return ""
    return "".join(ch for ch in str(text) if KEEP_CHAR_RE.match(ch))


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except Exception:
            pass

    brace_match = re.search(r"(\{.*\})", text, re.S)
    if brace_match:
        return json.loads(brace_match.group(1))

    raise ValueError("无法从模型输出中解析 JSON。原始输出如下：\n" + text)


def load_api_key() -> str:
    if not KEY_FILE.exists():
        raise FileNotFoundError(
            f"找不到 {KEY_FILE.resolve()}。\n"
            f"请在脚本同目录新建 dashscope_key.txt，并在其中写入一行 API key。"
        )
    key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"{KEY_FILE} 为空，请写入有效 API key。")
    return key


# =========================
# 时间字符单元
# =========================
@dataclass
class CharUnit:
    char: str
    start: Optional[float]
    end: Optional[float]


def build_char_units_from_segment(seg: Dict[str, Any]) -> List[CharUnit]:
    """
    优先使用 seg["chars"]。
    如果没有 chars，则从 seg["words"] 展开。
    """
    units: List[CharUnit] = []

    if isinstance(seg.get("chars"), list) and seg["chars"]:
        for ch in seg["chars"]:
            if not isinstance(ch, dict):
                continue
            c = str(ch.get("char", ""))
            if c == "":
                continue
            units.append(
                CharUnit(
                    char=c,
                    start=float(ch["start"]) if ch.get("start") is not None else None,
                    end=float(ch["end"]) if ch.get("end") is not None else None,
                )
            )
        return units

    # 退回 words
    for w in seg.get("words", []):
        if not isinstance(w, dict):
            continue
        word = str(w.get("word", ""))
        if not word:
            continue

        start = float(w["start"]) if w.get("start") is not None else None
        end = float(w["end"]) if w.get("end") is not None else None

        # 如果是单字符，直接放
        if len(word) == 1:
            units.append(CharUnit(word, start, end))
        else:
            # 多字符词，若有时长则均分；没有则都置空
            if start is not None and end is not None and end >= start:
                total = len(word)
                dur = (end - start) / total if total > 0 else 0.0
                for i, ch in enumerate(word):
                    s = start + i * dur
                    e = start + (i + 1) * dur
                    units.append(CharUnit(ch, s, e))
            else:
                for ch in word:
                    units.append(CharUnit(ch, None, None))

    return units


def fill_missing_times(units: List[CharUnit], seg_start: float, seg_end: float) -> None:
    """
    给没有时间的字符补时间。
    主要应对：
    - 标点没有时间
    - 少数字没有时间
    - 个别错字没有时间
    """
    if not units:
        return

    timed_indices = [i for i, u in enumerate(units) if u.start is not None and u.end is not None]

    # 全都没有时间，整体均分
    if not timed_indices:
        total = len(units)
        dur = max(seg_end - seg_start, 1e-6) / total
        for i, u in enumerate(units):
            u.start = seg_start + i * dur
            u.end = seg_start + (i + 1) * dur
        return

    # 前缀缺失：贴到第一个有时间字符之前
    first_timed = timed_indices[0]
    for i in range(first_timed - 1, -1, -1):
        units[i].start = units[first_timed].start
        units[i].end = units[first_timed].start

    # 后缀缺失：贴到最后一个有时间字符之后
    last_timed = timed_indices[-1]
    for i in range(last_timed + 1, len(units)):
        units[i].start = units[last_timed].end
        units[i].end = units[last_timed].end

    # 中间缺失：在前后有时间的字符之间插值
    i = 0
    while i < len(units):
        if units[i].start is not None and units[i].end is not None:
            i += 1
            continue

        j = i
        while j < len(units) and (units[j].start is None or units[j].end is None):
            j += 1

        prev_idx = i - 1
        next_idx = j if j < len(units) else None

        prev_t = units[prev_idx].end if prev_idx >= 0 else seg_start
        next_t = units[next_idx].start if next_idx is not None else seg_end

        gap_count = j - i
        if gap_count <= 0:
            i = j
            continue

        if next_t < prev_t:
            next_t = prev_t

        dur = (next_t - prev_t) / gap_count if gap_count > 0 else 0.0
        for k in range(gap_count):
            units[i + k].start = prev_t + k * dur
            units[i + k].end = prev_t + (k + 1) * dur

        i = j


# =========================
# LLM：纠错 + 分段
# =========================
def build_messages(orig_text: str) -> List[Dict[str, str]]:
    system_prompt = """你是“课堂话语分割与纠错助手”。

你的任务：对一段课堂 ASR 转写文本进行轻度纠错与话语分割。

要求：
1. corrected_text：输出更自然、正确的文本，可以修正明显错字、补合理标点。
2. segments：把 corrected_text 切成更自然的“教学话语单元”。
3. 每个 segment 尽量是完整的话语单元，不要把固定搭配、完整问句切断。
4. 尽量注意这些前导提示词前后可能是边界：
   同学们、下面、那么、接下来、首先、请看、我们来看、你知道吗、大家想一想
5. 不要把这些完整表达切碎：
   教学展示、比赛视频、东京奥运赛场、世界跳水双冠王
6. 不要输出任何解释。
7. 只返回 JSON，格式必须严格为：
{
  "corrected_text": "......",
  "segments": [
    "第一段",
    "第二段"
  ]
}
"""

    user_prompt = f"请对下面这段课堂文本进行纠错和话语分割，只返回 JSON：\n\n{orig_text}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_qwen_correct_and_segment(
    client: OpenAI,
    model_name: str,
    orig_text: str,
) -> Tuple[str, List[str]]:
    messages = build_messages(orig_text)

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
        stream=False,
    )

    content = completion.choices[0].message.content
    data = safe_json_loads(content)

    corrected_text = str(data.get("corrected_text", "")).strip()
    segs = data.get("segments", [])

    if not corrected_text:
        raise ValueError("模型没有返回 corrected_text。")
    if not isinstance(segs, list) or not segs:
        raise ValueError("模型没有返回有效 segments。")

    segments = [str(x).strip() for x in segs if str(x).strip()]
    if not segments:
        raise ValueError("模型 segments 为空。")

    return corrected_text, segments


# =========================
# 模糊对齐
# =========================
def build_norm_string_and_map_from_units(units: List[CharUnit]) -> Tuple[str, List[int]]:
    """
    返回：
    - norm_text：归一化后的字符串
    - norm_to_unit_idx：归一化字符串中每个字符对应原始 unit 的索引
    """
    chars = []
    idx_map = []
    for i, u in enumerate(units):
        if KEEP_CHAR_RE.match(u.char):
            chars.append(u.char)
            idx_map.append(i)
    return "".join(chars), idx_map


def project_boundary_corr_to_orig(
    corr_boundary: int,
    matching_blocks: List[Any],
) -> int:
    """
    将“纠正文本归一化串”中的边界位置，映射到“原始归一化串”中的边界位置。
    边界位置是 [0, len(norm_str)] 之间的整数，表示字符之间的边界。
    """
    # matching_blocks 元素: Match(a, b, size)
    # a: orig 起点, b: corr 起点
    # 如果落在完全匹配块内，直接映射
    for block in matching_blocks:
        a0, b0, size = block.a, block.b, block.size
        if b0 <= corr_boundary <= b0 + size:
            return a0 + (corr_boundary - b0)

    prev_block = None
    next_block = None

    for block in matching_blocks:
        if block.b + block.size <= corr_boundary:
            prev_block = block
        elif block.b > corr_boundary:
            next_block = block
            break

    if prev_block is None and next_block is None:
        return 0

    if prev_block is None:
        return next_block.a

    if next_block is None:
        return prev_block.a + prev_block.size

    prev_b = prev_block.b + prev_block.size
    prev_a = prev_block.a + prev_block.size
    next_b = next_block.b
    next_a = next_block.a

    # 在两个块之间做线性插值
    if next_b == prev_b:
        return prev_a

    ratio = (corr_boundary - prev_b) / (next_b - prev_b)
    mapped = prev_a + ratio * (next_a - prev_a)
    return int(round(mapped))


def align_llm_segments_fuzzy(
    orig_units: List[CharUnit],
    corrected_text: str,
    llm_segments: List[str],
) -> List[Dict[str, Any]]:
    """
    用模糊对齐把 LLM 输出的 segments 映射回原始 unit 时间轴。
    """
    orig_norm, orig_norm_to_unit = build_norm_string_and_map_from_units(orig_units)

    # 用分段拼接后的文本做对齐主串
    corr_concat = "".join(llm_segments)
    corr_norm = normalize_text_for_match(corr_concat)

    if not corr_norm:
        raise ValueError("LLM 分段内容归一化后为空。")

    # 全局模糊对齐
    sm = SequenceMatcher(a=orig_norm, b=corr_norm, autojunk=False)
    matching_blocks = sm.get_matching_blocks()

    # 计算每个 segment 在 corr_norm 中的边界
    corr_seg_norm_lengths = [len(normalize_text_for_match(s)) for s in llm_segments]

    outputs = []
    cursor = 0
    for seg_text, seg_len in zip(llm_segments, corr_seg_norm_lengths):
        if seg_len == 0:
            continue

        corr_start = cursor
        corr_end = cursor + seg_len  # exclusive
        cursor = corr_end

        orig_start_norm = project_boundary_corr_to_orig(corr_start, matching_blocks)
        orig_end_norm = project_boundary_corr_to_orig(corr_end, matching_blocks)

        # 防止边界倒置
        if orig_end_norm <= orig_start_norm:
            orig_end_norm = min(len(orig_norm), orig_start_norm + 1)

        # 转回原始 unit 索引
        orig_start_norm = max(0, min(orig_start_norm, len(orig_norm) - 1))
        orig_end_norm = max(1, min(orig_end_norm, len(orig_norm)))

        start_unit_idx = orig_norm_to_unit[orig_start_norm]
        end_unit_idx = orig_norm_to_unit[orig_end_norm - 1]

        seg_units = orig_units[start_unit_idx:end_unit_idx + 1]
        start_time = seg_units[0].start
        end_time = seg_units[-1].end

        outputs.append({
            "text": seg_text,
            "start": round(float(start_time), 3),
            "end": round(float(end_time), 3),
            "duration": round(float(end_time - start_time), 3),
            "orig_text_span": "".join(u.char for u in seg_units),
        })

    return outputs


# =========================
# 切视频 / 音频
# =========================
def cut_video(input_video: Path, start: float, end: float, out_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(input_video),
        "-c:v", "libx264",
        "-c:a", "aac",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def cut_audio(input_video: Path, start: float, end: float, out_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(input_video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


# =========================
# 主流程
# =========================
def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"找不到 JSON 文件: {JSON_PATH.resolve()}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"找不到视频文件: {VIDEO_PATH.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips_dir = OUT_DIR / "clips"
    audio_dir = OUT_DIR / "audio"
    clips_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    api_key = load_api_key()
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    final_segments = []
    seg_counter = 1

    for orig_idx, seg in enumerate(data.get("segments", []), start=1):
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))

        orig_units = build_char_units_from_segment(seg)
        if not orig_units:
            continue

        fill_missing_times(orig_units, seg_start, seg_end)

        orig_text = "".join(u.char for u in orig_units).strip()
        if not orig_text:
            continue

        try:
            corrected_text, llm_segments = call_qwen_correct_and_segment(
                client=client,
                model_name=MODEL_NAME,
                orig_text=orig_text,
            )

            aligned_parts = align_llm_segments_fuzzy(
                orig_units=orig_units,
                corrected_text=corrected_text,
                llm_segments=llm_segments,
            )

            if not aligned_parts:
                raise ValueError("模糊对齐后没有得到任何子段。")

            for part in aligned_parts:
                row = {
                    "seg_id": f"seg_{seg_counter:04d}",
                    "orig_segment_id": orig_idx,
                    "text": part["text"],                # 纠错后的展示文本
                    "start": part["start"],
                    "end": part["end"],
                    "duration": part["duration"],
                    "orig_text_span": part["orig_text_span"],  # 原始时间轴对应的文本片段
                }
                final_segments.append(row)
                seg_counter += 1

        except Exception as e:
            print(f"[WARN] 原始段 {orig_idx} 处理失败: {e}")

            if FALLBACK_TO_ORIGINAL_SEGMENT:
                row = {
                    "seg_id": f"seg_{seg_counter:04d}",
                    "orig_segment_id": orig_idx,
                    "text": orig_text,
                    "start": round(seg_start, 3),
                    "end": round(seg_end, 3),
                    "duration": round(seg_end - seg_start, 3),
                    "orig_text_span": orig_text,
                }
                final_segments.append(row)
                seg_counter += 1

    out_json = OUT_DIR / "llm_resegmented_fuzzy.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"segments": final_segments}, f, ensure_ascii=False, indent=2)

    out_jsonl = OUT_DIR / "llm_resegmented_fuzzy.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in final_segments:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 切视频和音频
    for row in final_segments:
        start = max(0.0, row["start"] - PAD_BEFORE)
        end = row["end"] + PAD_AFTER

        clip_path = clips_dir / f"{row['seg_id']}.mp4"
        wav_path = audio_dir / f"{row['seg_id']}.wav"

        cut_video(VIDEO_PATH, start, end, clip_path)
        cut_audio(VIDEO_PATH, start, end, wav_path)

    print(f"完成。共生成 {len(final_segments)} 个更细的话语片段。")
    print(f"JSON:  {out_json.resolve()}")
    print(f"JSONL: {out_jsonl.resolve()}")
    print(f"视频切片目录: {clips_dir.resolve()}")
    print(f"音频切片目录: {audio_dir.resolve()}")


if __name__ == "__main__":
    main()