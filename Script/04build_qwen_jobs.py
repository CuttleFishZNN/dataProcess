# 调整LLM从视频中提取出来的Json数据的格式

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
AUDIO_EXTS = [".wav", ".mp3", ".m4a", ".flac", ".aac"]


def find_matching_file(folder: Path, stem: str, exts: list[str]) -> Optional[Path]:
    for ext in exts:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    matches = list(folder.glob(f"{stem}.*"))
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_json", required=True, help="anno/llm_resegmented_fuzzy.json")
    parser.add_argument("--clips_dir", required=True, help="raw/clips")
    parser.add_argument("--audio_dir", default="", help="raw/audio，可留空")
    parser.add_argument("--output_jsonl", required=True, help="输出任务 JSONL")
    parser.add_argument("--subject", default="math")
    parser.add_argument("--teacher_id", default="T01")
    parser.add_argument("--lesson_id", default="MATH_T01_L01")
    args = parser.parse_args()

    anno_json = Path(args.anno_json)
    clips_dir = Path(args.clips_dir)
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(anno_json.read_text(encoding="utf-8"))
    segments = data["segments"]

    total = 0
    missing_clips = 0

    with output_jsonl.open("w", encoding="utf-8") as fout:
        for seg in segments:
            seg_id = seg["seg_id"]

            clip_path = find_matching_file(clips_dir, seg_id, VIDEO_EXTS)
            audio_path = find_matching_file(audio_dir, seg_id, AUDIO_EXTS) if audio_dir else None

            if clip_path is None:
                print(f"[WARN] clip 不存在: {seg_id}")
                missing_clips += 1
                continue

            job = {
                "seg_id": seg_id,
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["duration"],
                "clip_path": str(clip_path).replace("\\", "/"),
                "audio_path": str(audio_path).replace("\\", "/") if audio_path else "",
                "subject": args.subject,
                "teacher_id": args.teacher_id,
                "lesson_id": args.lesson_id,
            }
            fout.write(json.dumps(job, ensure_ascii=False) + "\n")
            total += 1

    print(f"[DONE] 写入任务数: {total}")
    print(f"[DONE] 缺失 clip 数: {missing_clips}")
    print(f"[DONE] 输出文件: {output_jsonl}")


if __name__ == "__main__":
    main()