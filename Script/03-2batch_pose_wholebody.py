# 教学视频中教师的动作提取
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
from tqdm import tqdm

# -----------------------------
# 模型配置（已修复路径，无报错）
# -----------------------------
DET_CONFIG = "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
DET_WEIGHTS = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
    "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
)
POSE_CONFIG = (
    "configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/"
    "td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py"
)
POSE_WEIGHTS = (
    "https://download.openmmlab.com/mmpose/top_down/hrnet/"
    "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量处理视频片段，提取全身姿态")
    parser.add_argument("--lesson-dir", type=str, required=True, help="课次目录")
    parser.add_argument("--mmpose-dir", type=str, required=True, help="MMPose根目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="运行设备")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在结果")
    parser.add_argument("--limit", type=int, default=0, help="限制处理数量")
    parser.add_argument("--sleep", type=float, default=0.0, help="片段间暂停时间")
    parser.add_argument("--quiet-finish-log", action="store_true", help="静默模式，仅显示进度条")
    parser.add_argument("--refresh-sec", type=float, default=1.0, help="进度条刷新间隔")
    return parser.parse_args()

def ensure_exists(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{desc} 不存在: {path}")

def collect_clips(clips_dir: Path, limit: int) -> List[Path]:
    clips = sorted(clips_dir.glob("*.mp4"))
    return clips[:limit] if limit > 0 else clips

def already_done(output_dir: Path, seg_id: str) -> bool:
    pred_json = output_dir / f"results_{seg_id}.json"
    vis_video = output_dir / f"{seg_id}.mp4"
    return pred_json.exists() and vis_video.exists()

def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0

def get_video_info(video_path: Path) -> Tuple[int, float, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0.0, 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    duration_sec = frame_count / fps if (frame_count > 0 and fps > 0) else 0.0
    return frame_count, fps, duration_sec

def write_summary(summary_path: Path, rows: List[Dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "segment_id", "clip_path", "frame_count", "fps", "video_duration_sec",
        "output_dir", "json_path", "video_path", "log_path",
        "status", "return_code", "elapsed_sec", "cumulative_elapsed_sec",
        "avg_vsec_per_sec", "est_vsec_per_sec"
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    args = parse_args()
    lesson_dir = Path(args.lesson_dir).resolve()
    mmpose_dir = Path(args.mmpose_dir).resolve()
    clips_dir = lesson_dir / "raw" / "clips"
    outputs_root = lesson_dir / "pose" / "wholebody_hrnet" / "outputs"
    summary_path = lesson_dir / "pose" / "wholebody_hrnet" / "run_summary.csv"

    demo_script = mmpose_dir / "demo" / "topdown_demo_with_mmdet.py"
    det_config_path = mmpose_dir / DET_CONFIG
    pose_config_path = mmpose_dir / POSE_CONFIG

    # 校验文件
    ensure_exists(lesson_dir, "课次目录")
    ensure_exists(mmpose_dir, "MMPose目录")
    ensure_exists(clips_dir, "视频片段目录")
    ensure_exists(demo_script, "推理脚本")
    ensure_exists(det_config_path, "检测配置")
    ensure_exists(pose_config_path, "姿态配置")

    outputs_root.mkdir(parents=True, exist_ok=True)
    clips = collect_clips(clips_dir, args.limit)
    if not clips:
        print(f"[WARN] 未找到MP4文件: {clips_dir}")
        sys.exit(1)

    # 预加载视频信息
    clip_infos = {}
    total_video_sec_all = 0.0
    for clip in clips:
        frame_count, fps, duration_sec = get_video_info(clip)
        clip_infos[clip.stem] = {"frame_count": frame_count, "fps": fps, "duration_sec": duration_sec}
        total_video_sec_all += duration_sec

    print(f"[INFO] 总片段数: {len(clips)} | 总视频时长: {total_video_sec_all:.2f}s")
    print("-" * 100)

    summary_rows = []
    total_start = time.perf_counter()
    done_count = skipped_count = failed_count = 0
    processed_video_sec = 0.0

    pbar = tqdm(total=len(clips), desc="Pose extracting", unit="clip", dynamic_ncols=True, leave=True)

    try:
        for idx, clip_path in enumerate(clips, 1):
            seg_id = clip_path.stem
            info = clip_infos.get(seg_id, {})
            duration_sec = float(info.get("duration_sec", 0.0))
            output_dir = outputs_root / seg_id
            output_dir.mkdir(parents=True, exist_ok=True)
            pred_json = output_dir / f"results_{seg_id}.json"
            vis_video = output_dir / f"{seg_id}.mp4"
            log_path = output_dir / f"{seg_id}.log"

            # 跳过已处理
            if not args.overwrite and already_done(output_dir, seg_id):
                skipped_count += 1
                processed_video_sec += duration_sec
                cumulative_elapsed = time.perf_counter() - total_start
                avg_vsec = safe_div(processed_video_sec, cumulative_elapsed)
                summary_rows.append({
                    "segment_id": seg_id, "status": "skipped_exists", "elapsed_sec": 0.0,
                    "avg_vsec_per_sec": round(avg_vsec, 3), "est_vsec_per_sec": 0.0
                })
                write_summary(summary_path, summary_rows)
                if not args.quiet_finish_log:
                    tqdm.write(f"[{idx}/{len(clips)}] {seg_id} | 已跳过 | avg_vsec/s={avg_vsec:.2f}")
                pbar.set_postfix_str(f"seg={seg_id} | SKIP | avg_vsec/s={avg_vsec:.2f}")
                pbar.update(1)
                continue

            # 构建命令
            cmd = [
                sys.executable, str(demo_script), str(det_config_path), DET_WEIGHTS,
                str(pose_config_path), POSE_WEIGHTS, "--input", str(clip_path),
                "--output-root", str(output_dir), "--save-predictions", "--device", args.device
            ]

            clip_start = time.perf_counter()
            proc, result_code = None, -1
            try:
                with log_path.open("w", encoding="utf-8") as log_file:
                    proc = subprocess.Popen(cmd, cwd=str(mmpose_dir), stdout=log_file, stderr=subprocess.STDOUT, text=True)
                    # 实时刷新进度条
                    while True:
                        ret = proc.poll()
                        clip_elapsed_live = time.perf_counter() - clip_start
                        cumulative_elapsed_live = time.perf_counter() - total_start

                        est_vsec = safe_div(duration_sec, clip_elapsed_live)
                        avg_vsec = safe_div(processed_video_sec, cumulative_elapsed_live)
                        remaining = len(clips) - (done_count + skipped_count + failed_count)
                        avg_clip = safe_div(cumulative_elapsed_live, max(1, done_count+skipped_count+failed_count))
                        eta = remaining * avg_clip

                        pbar.set_postfix_str(
                            f"seg={seg_id} | est={est_vsec:.2f} | avg={avg_vsec:.2f} | eta={format_seconds(eta)}"
                        )
                        if ret is not None:
                            result_code = ret
                            break
                        time.sleep(args.refresh_sec)
            except KeyboardInterrupt:
                if proc: proc.kill()
                raise

            # 处理完成
            clip_elapsed = round(time.perf_counter() - clip_start, 3)
            cumulative_elapsed = time.perf_counter() - total_start
            status = "done"
            if result_code != 0:
                status = "failed"
                failed_count += 1
            elif not pred_json.exists() or not vis_video.exists():
                status = "missing_output"
                failed_count += 1
            else:
                done_count += 1
                processed_video_sec += duration_sec

            avg_vsec = safe_div(processed_video_sec, cumulative_elapsed)
            est_vsec = safe_div(duration_sec, clip_elapsed)
            summary_rows.append({
                "segment_id": seg_id, "status": status, "elapsed_sec": clip_elapsed,
                "avg_vsec_per_sec": round(avg_vsec, 3), "est_vsec_per_sec": round(est_vsec, 3)
            })
            write_summary(summary_path, summary_rows)

            if not args.quiet_finish_log:
                tqdm.write(f"[{idx}/{len(clips)}] {seg_id} | {status} | 耗时={clip_elapsed:.3f}s | avg={avg_vsec:.2f}")
            pbar.update(1)
            if args.sleep > 0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\n[WARN] 手动中断，结果已保存")
    finally:
        pbar.close()

    # 最终统计
    total_elapsed = time.perf_counter() - total_start
    final_avg = safe_div(processed_video_sec, total_elapsed)
    print("-" * 100)
    print(f"[INFO] 处理完成 | 总耗时: {format_seconds(total_elapsed)}")
    print(f"[INFO] 成功:{done_count} | 跳过:{skipped_count} | 失败:{failed_count}")
    print(f"[INFO] 平均吞吐量: {final_avg:.2f} vsec/s")
    print(f"[INFO] 日志文件: {summary_path}")

if __name__ == "__main__":
    main()