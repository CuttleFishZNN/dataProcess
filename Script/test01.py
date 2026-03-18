import json
import whisperx

video_file = "data/math/hubei_qingjiao_man_01.mp4"
device = "cuda"
compute_type = "float16"
batch_size = 4

model = whisperx.load_model("medium", device, compute_type=compute_type)
audio = whisperx.load_audio(video_file)
result = model.transcribe(audio, batch_size=batch_size)

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
    return_char_alignments=False
)

with open("teacher_class_whisperx.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)