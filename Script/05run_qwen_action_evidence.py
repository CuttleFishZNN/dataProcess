from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from openai import OpenAI
from tqdm import tqdm


# =========================
# 配置区：直接改这里即可
# =========================
PROJECT_ROOT = Path(r"E:\Project\Python\dataProcess")

JOBS_JSONL = PROJECT_ROOT / r"dataset\math\T01\MATH_T01_L01\anno\qwen_action_jobs.jsonl"
OUT_JSONL = PROJECT_ROOT / r"dataset\math\T01\MATH_T01_L01\anno\action_evidence_qwen25vl0000-qwen3.5-plus.jsonl"

# 改成你在 DashScope 上实际可用的视觉模型名
# MODEL_NAME = "qwen-vl-max-latest"
# MODEL_NAME = "qwen3.5-35b-a3b"

MODEL_NAME = "qwen3.5-plus"

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 在脚本同目录创建 dashscope_key.txt，里面只放一行 API key
KEY_FILE = Path(__file__).resolve().parent / "dashscope_key.txt"

# 抽帧参数
NUM_FRAMES = 6
MAX_SIDE = 960

# 是否跳过已经成功完成的 seg_id
SKIP_DONE = True


PROMPT_TEMPLATE = """
你是一个“课堂教师动作证据提取器”。

你的唯一任务：
给定一个课堂视频片段的若干关键帧和对应话语文本，请你只提取“视觉上可直接观察到的动作证据”，并输出为 JSON。

你不是教学分析器，不是情感分析器，也不是意图识别器。
不要输出教学意图、节奏、力度、情绪、风格、学科解释等高层标签。
不要根据文本内容去猜动作，只能依据图像中实际可见内容作答。

====================
一、核心原则
====================

1. 只写“看见了什么”，不要写“这意味着什么”。
2. 文本只用于对齐样本，不能作为动作判断依据。
3. 任何不可见、被遮挡、模糊、角度不足的信息，都必须写 unknown / occluded / not_visible，或放入 uncertain_parts。
4. 如果不能确定，就不要猜；宁可保守，也不要脑补。
5. 输出必须是合法 JSON，不要输出任何额外解释、标题、注释或 markdown 代码块。
6. 所有字段值必须尽量使用下方“允许词表”；不要自造新标签。

====================
二、严格禁止
====================

1. 禁止把动作证据写成教学意图：
   不要输出“在强调”“在提问”“在引导”“在总结”“在鼓励”等解释性表述。

2. 禁止根据文本脑补动作：
   例如文本里出现“看黑板”，不代表画面里一定有“指向黑板”。

3. 禁止脑补不可见身体部位：
   尤其下半身若被讲台遮挡，不要猜测是否迈步、转身、走动。

4. 禁止随意识别小物体：
   对于 remote_control / clicker / microphone / marker / pen 等小物体，
   除非在多个关键帧中都非常清晰、非常确定，否则不要输出。
   不确定时，object_interaction 应为空列表 []，或在 uncertain_parts 中说明“手中可能有物体，但无法确认”。

5. 禁止输出组合标签：
   不能输出类似：
   - "toward_students / downward"
   - "upright / leaning_forward"
   - "clicker/remote"
   每个字段只能填一个允许值。
   如果状态确实混合，填 mixed；如果无法判断，填 unknown。

6. 禁止输出不在允许词表中的动作词：
   如 hanging、touching 等。
   如果没有合适词，请改成最接近的允许值；实在不确定则填 unknown。

====================
三、主状态规则
====================

1. 顶层字段 body_orientation / head_orientation / posture / locomotion 表示该片段的“主状态”。
2. 如果片段内不同帧之间状态有变化：
   - 若存在明显主状态，顶层字段填主状态；
   - 若没有明显主状态，填 mixed 或 unknown；
   - 并在 evidence_sentences 中写明“前几帧……后几帧……”。
3. 如果某动作只在少数帧短暂出现，但确实清楚可见，可以写入 arm_actions / hand_actions，并在 evidence_sentences 中说明。
4. 如果证据太弱，不要强行输出动作。

====================
四、置信度规则
====================

confidence 取值范围 0.0 ~ 1.0，建议如下：

- 0.90 ~ 1.00：多个帧中都非常清楚，几乎无歧义
- 0.75 ~ 0.89：较清楚，但仍有少量遮挡或角度限制
- 0.55 ~ 0.74：可以判断，但证据有限
- 0.40 ~ 0.54：较弱证据，谨慎使用
- < 0.40：通常不要输出该动作，改用 unknown / [] / uncertain_parts

如果你自己都不太确定，就不要为了填满 JSON 而写低质量动作。

====================
五、固定词表（严格使用）
====================

visible_body_parts 只允许：
- head
- torso
- left_arm
- right_arm
- left_hand
- right_hand
- lower_body

body_orientation 只允许：
- toward_students
- toward_blackboard
- toward_screen
- side_to_blackboard
- mixed
- unknown

head_orientation 只允许：
- toward_students
- toward_blackboard
- toward_screen
- downward
- side
- mixed
- unknown

posture 只允许：
- upright
- leaning_forward
- leaning_sideways
- unknown

locomotion 只允许：
- standing_still
- walking
- stepping
- unknown

arm_actions.action 只允许：
- raising
- pointing
- extending
- retracting
- resting
- unknown

hand_actions.action 只允许：
- open_palm
- pointing
- holding_object
- writing_like
- resting
- unknown

target 只允许：
- students
- blackboard
- screen
- desk
- self
- unknown

side 只允许：
- left
- right
- both
- unknown

====================
六、object_interaction 规则
====================

1. object_interaction 默认优先输出 []。
2. 只有当物体“大且清楚”或“接触关系非常明确”时才填写。
3. 优先考虑的大目标物体只有：
   - blackboard
   - screen
   - desk
   - book
   - paper
   - unknown
4. 除非极其清楚，否则不要识别遥控器、话筒、点击器、马克笔等小物体。
5. 如果只是“手靠近某区域”，不代表发生了 object_interaction。

====================
七、布尔字段严格判定
====================

board_interaction.writing = true 的条件：
- 必须清楚看到手/笔/接触黑板，并呈现明显书写样动作
- 不能因为“老师面朝黑板”就判 true

board_interaction.erasing = true 的条件：
- 必须清楚看到擦除动作
- 否则为 false

board_interaction.touching_board = true 的条件：
- 必须清楚看到手或工具与黑板接触
- 不能仅因靠近黑板就判 true

student_interaction_cues.facing_students = true 的条件：
- 身体或头部明显朝向学生区域

student_interaction_cues.addressing_students_visually = true 的条件：
- 能看出视线/头部/身体明显面向学生
- 不能仅因文本里出现“同学们”就判 true

student_interaction_cues.inviting_attention = true 的条件：
- 必须有可见动作支持，例如明显招呼式抬手、指向、开放手势等
- 不能根据说话内容脑补

====================
八、evidence_sentences 与 uncertain_parts
====================

1. evidence_sentences 必须写 2~4 句，不能为空。
2. 每句都必须是“可观察的事实描述”，不能写解释。
3. 推荐写法：
   - “教师身体朝向学生区域。”
   - “右臂抬起并向屏幕方向伸出。”
   - “下半身被讲台遮挡，无法判断步态。”
4. uncertain_parts 在存在遮挡、模糊、视角不足、物体不清楚时必须填写。
5. 如果确实没有明显不确定项，也可以输出空列表 []，但不要滥用。
6. 如果顶层字段填 mixed，evidence_sentences 里必须说明变化依据。

====================
九、输出格式
====================

请严格按照以下 schema 输出，不要新增无关字段，不要缺字段：

{
  "clip_id": "",
  "text": "",
  "visible_body_parts": [],
  "body_orientation": "",
  "head_orientation": "",
  "posture": "",
  "locomotion": "",
  "arm_actions": [
    {
      "side": "",
      "action": "",
      "target": "",
      "confidence": 0.0
    }
  ],
  "hand_actions": [
    {
      "side": "",
      "action": "",
      "target": "",
      "confidence": 0.0
    }
  ],
  "board_interaction": {
    "writing": false,
    "erasing": false,
    "touching_board": false
  },
  "object_interaction": [],
  "student_interaction_cues": {
    "facing_students": false,
    "addressing_students_visually": false,
    "inviting_attention": false
  },
  "occlusion": {
    "upper_body_occluded": false,
    "lower_body_occluded": false,
    "desk_occlusion": false
  },
  "evidence_sentences": [],
  "uncertain_parts": []
}

补充要求：
- 如果没有足够证据，不要为了“完整”而强行填动作
- arm_actions 和 hand_actions 可以为空列表 []
- object_interaction 默认优先输出 []
- 除非极其清楚，否则不要识别遥控器、话筒、点击器、马克笔等小物体
- 顶层字段必须是单值，不能写组合串
- 所有输出尽量克制、保守、可验证

当前样本：
clip_id: __SEG_ID__
text: __TEXT__
""".strip()


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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("status") == "ok" and obj.get("seg_id"):
                    done.add(obj["seg_id"])
            except Exception:
                pass
    return done


def resolve_clip_path(raw_path: str) -> Path:
    clip_path = Path(raw_path)
    if not clip_path.is_absolute():
        clip_path = PROJECT_ROOT / clip_path
    return clip_path.resolve()


def sample_frames_from_video(
    video_path: Path,
    num_frames: int = 6,
    max_side: int = 960,
) -> List[str]:
    video_path = video_path.resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"视频已打开，但帧数异常: {video_path}")

    idxs = np.linspace(0, frame_count - 1, num=min(num_frames, frame_count), dtype=int)
    frames_b64: List[str] = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        scale = min(max_side / max(h, w), 1.0)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frames_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))

    cap.release()

    if not frames_b64:
        raise RuntimeError(f"没有成功抽取关键帧: {video_path}")

    return frames_b64


def build_prompt(seg_id: str, text: str) -> str:
    prompt = PROMPT_TEMPLATE.replace("__SEG_ID__", seg_id)
    prompt = prompt.replace("__TEXT__", text)
    return prompt


def call_qwen_action_evidence(
    client: OpenAI,
    model_name: str,
    seg_id: str,
    text: str,
    frames_b64: List[str],
) -> Dict[str, Any]:
    prompt = build_prompt(seg_id=seg_id, text=text)

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b64 in frames_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        stream=False,
    )

    raw_text = completion.choices[0].message.content
    if not raw_text:
        raise ValueError("模型返回为空。")

    parsed = safe_json_loads(raw_text)
    return {
        "raw_text": raw_text,
        "parsed_json": parsed,
    }


def main() -> None:
    if not JOBS_JSONL.exists():
        raise FileNotFoundError(f"找不到任务文件: {JOBS_JSONL.resolve()}")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key()
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    jobs = read_jsonl(JOBS_JSONL)
    done_ids = load_done_ids(OUT_JSONL) if SKIP_DONE else set()

    print(f"[INFO] 任务总数: {len(jobs)}")
    print(f"[INFO] 已完成数: {len(done_ids)}")
    print(f"[INFO] 输出文件: {OUT_JSONL}")

    for job in tqdm(jobs, desc="Qwen 动作证据提取"):
        seg_id = str(job["seg_id"])
        if seg_id in done_ids:
            continue

        text = str(job["text"])
        clip_path = resolve_clip_path(job["clip_path"])

        try:
            frames_b64 = sample_frames_from_video(
                video_path=clip_path,
                num_frames=NUM_FRAMES,
                max_side=MAX_SIDE,
            )

            result = call_qwen_action_evidence(
                client=client,
                model_name=MODEL_NAME,
                seg_id=seg_id,
                text=text,
                frames_b64=frames_b64,
            )

            row = {
                "seg_id": seg_id,
                "text": text,
                "clip_path": str(clip_path),
                "status": "ok",
                "action_evidence": result["parsed_json"],
                "raw_response": result["raw_text"],
            }
            append_jsonl(OUT_JSONL, row)

        except Exception as e:
            print(f"[ERROR] seg_id={seg_id} | clip_path={clip_path} | error={e}")
            row = {
                "seg_id": seg_id,
                "text": text,
                "clip_path": str(clip_path),
                "status": "error",
                "error": str(e),
            }
            append_jsonl(OUT_JSONL, row)

    print("[DONE] 处理完成。")


if __name__ == "__main__":
    main()