import json
from pathlib import Path
import whisperx

video_file = Path(r"../data/math/test_5min.mp4")
device = "cuda"
compute_type = "float16"
batch_size = 4
language = "zh"
model_name = "large-v2"   # 比 medium 更稳一些，RTX2080 可以试

if not video_file.exists():
    raise FileNotFoundError(f"找不到视频文件: {video_file.resolve()}")

# 1. 加载模型
model = whisperx.load_model(
    model_name,
    device,
    compute_type=compute_type,
    language=language,
    vad_method="silero",
)

# 2. 读取音频
audio = whisperx.load_audio(str(video_file))

# 3. 转写
result = model.transcribe(
    audio,
    batch_size=batch_size,
    language=language
)

# 4. 对齐，保留更细的字符级信息
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"],
    device=device
)

result = whisperx.align(
    result["segments"],
    model_a,
    metadata,
    audio,
    device,
    return_char_alignments=True
)

# 5. 保存完整底稿
out_file = Path("teacher_class_whisperx_full.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Done. Saved to {out_file.resolve()}")