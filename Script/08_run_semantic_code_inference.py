from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm


# =========================
# 配置区：直接改这里即可
# =========================
PROJECT_ROOT = Path(r"E:\Project\Python\dataProcess")

INPUT_JSONL = PROJECT_ROOT / r"dataset\math\T01\MATH_T01_L01\anno\semantic_code_train_input.jsonl"
OUT_JSONL = PROJECT_ROOT / r"dataset\math\T01\MATH_T01_L01\Result\semantic_code_results_qwen3.5plus.jsonl"

MODEL_NAME = "qwen3.5-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 在脚本同目录创建 dashscope_key.txt，里面只放一行 API key
KEY_FILE = Path(__file__).resolve().parent / "dashscope_key.txt"

# True: 已经成功的 seg_id 不再重复跑
SKIP_DONE = True

# =========================
# 只跑部分样本（灵活控制）
# =========================
START_INDEX = 32      # 从第几条开始（0 表示第一条）
MAX_SAMPLES = None     # 只跑多少条；设为 None 表示从 START_INDEX 一直跑到最后


# =========================
# 【冻结版】终版提示词
# =========================
PROMPT_TEMPLATE = """
你是一个“课堂教学语义码标注器”。

你的任务：
根据一条课堂话语的文本、音频韵律摘要、视频动作证据，
为该话语标注一个“教学语义码”。

====================
一、核心目标
====================

你要标注的是“这句话在课堂中的主教学功能”，
不是把所有伴随动作都堆在一起。

必须区分两个层面：
1. intent：这句话最主要的教学功能
2. reference_flag / writing_state / body_orientation / interaction_flag：伴随的辅助属性

很多句子会“一边解释，一边指屏幕”；
此时如果主要功能是解释，则：
- intent = EXPLAIN
- reference_flag = true
而不是直接把 intent 改成 REFERENCE。

====================
二、模态分工
====================

1. 文本：主依据
- 主要决定 intent
- 主要决定 primitive
- 主要决定 sentence_independence

2. 音频韵律摘要：辅助依据
- 主要决定 rhythm
- 辅助判断 QUESTION / EMPHASIZE 的语气强弱
- 不单独决定 intent，只辅助修正

3. 视频动作证据：辅助依据
- 主要决定 reference_flag
- 主要决定 reference_target
- 主要决定 writing_state
- 主要决定 body_orientation
- 辅助判断 interaction_flag
- 不要让视频中的一个弱动作证据推翻文本主教学功能

当三者不一致时：
- 优先信任文本确定主教学功能
- 音频主要修正 rhythm
- 视频主要修正辅助属性

====================
三、固定决策顺序（必须遵守）
====================

请严格按以下顺序思考并输出结果：

步骤 1：先只根据文本，判断这句话最可能的主教学功能 intent
步骤 2：再看音频，修正 rhythm，并辅助判断 QUESTION/EMPHASIZE 倾向
步骤 3：再看视频，补充或修正：
- reference_flag
- reference_target
- writing_state
- body_orientation
- interaction_flag
步骤 4：最后再确定 primitive，且 primitive 必须与主 intent 一致
步骤 5：最后再判断 sentence_independence 和 confidence

不要先被视频里的“指向动作”带偏，再倒推 intent。

====================
四、输出 JSON
====================

请输出且只输出以下 JSON 对象：

{
  "intent": "",
  "primitive": "",
  "rhythm": "",
  "interaction_flag": false,
  "reference_flag": false,
  "reference_target": "",
  "body_orientation": "",
  "writing_state": false,
  "sentence_independence": "",
  "confidence": 0.0,
  "reason_brief": ""
}

不要输出任何 JSON 之外的文字。
不要输出 markdown 代码块。
不要输出注释。
不要输出多余逗号。
不要新增字段。

====================
五、字段允许值
====================

intent 只允许以下 9 类之一：
- DEFINE
- EXPLAIN
- EMPHASIZE
- QUESTION
- ENUMERATE
- COMPARE
- REFERENCE
- TRANSITION
- EXAMPLE

primitive 只允许以下之一：
- point
- open_explain
- invite_response
- write_board
- summarize
- compare
- enumerate
- transition
- emphasize
- neutral

rhythm 只允许：
- slow
- medium
- fast

reference_target 只允许：
- blackboard
- screen
- students
- desk
- self
- none
- unknown

body_orientation 只允许：
- toward_students
- toward_blackboard
- toward_screen
- side_to_blackboard
- mixed
- unknown

sentence_independence 只允许：
- independent
- context_dependent

====================
六、intent 判定规则（最重要）
====================

1. QUESTION
当文本是明确问句时，通常优先判定为 QUESTION。
典型信号包括：
- 吗
- 呢
- 谁来说
- 你认为
- 请你回答
- 你知道吗
- 你看过吗
- 你还听说过吗
即使视频中有指向动作，也通常仍以 QUESTION 为主。
只有当整句话的核心功能明显不是提问，而是要求学生观察某对象时，才不判 QUESTION。

2. EXPLAIN
当文本主要是在展开说明、描述过程、介绍内容、补充解释时，判 EXPLAIN。
即使伴随朝向屏幕/黑板或轻微指向，也优先保持 EXPLAIN。

3. TRANSITION
当文本主要作用是进入下一步、切换环节、承上启下时，判 TRANSITION。
典型信号包括：
- 下面
- 接下来
- 那么今天我们来……
- 现在我们来看……
即使伴随自然手势或轻微指向，也不要轻易改成 REFERENCE。

4. EMPHASIZE
当文本主要作用是提醒重点、突出注意事项、加强语气时，判 EMPHASIZE。
音频中如果有明显强度、停顿、重音，可增强该判断。

5. REFERENCE
只有在以下情况下才把 REFERENCE 作为 intent：
- 该句最主要的教学功能就是把学生注意力明确引向某个对象/区域
- 文本本身主要是“看这里 / 看黑板 / 看图 / 看屏幕 / 请观察这里”
- 视频也支持明确、稳定的指向行为

如果一句话主要是在解释、提问、过渡、举例，只是伴随指向动作：
- intent 保持原主功能
- reference_flag = true
不要把 intent 直接改成 REFERENCE。

6. DEFINE / ENUMERATE / COMPARE / EXAMPLE
只有当文本主功能非常明确时才使用。
不要为了“分得更细”而过度使用这些类。

====================
七、primitive 判定规则
====================

primitive 必须与主 intent 一致，不要被局部动作带偏。

优先匹配规则：
- QUESTION -> invite_response
- EXPLAIN -> open_explain
- TRANSITION -> transition
- EMPHASIZE -> emphasize
- ENUMERATE -> enumerate
- COMPARE -> compare
- REFERENCE -> point
- DEFINE -> open_explain 或 summarize（择其更合适者）
- EXAMPLE -> open_explain 或 neutral（择其更合适者）

补充约束：
1. 若 intent = QUESTION，通常 primitive 不应写成 point
2. 若 intent = EXPLAIN，通常 primitive 不应因为伴随指向而改成 point
3. 只有当主功能本身就是指向/指代时，primitive 才优先用 point
4. 若没有明显表达原语，再使用 neutral

====================
八、rhythm 判定规则
====================

1. rhythm 主要依据音频韵律摘要
2. speech_rate_level 是第一依据
3. pause_style / voicing_continuity / prosody_note 是辅助依据
4. 不要主要依据文本长短来判定 rhythm
5. 文本是问句，不自动意味着 fast；文本很长，也不自动意味着 slow

====================
九、reference_flag / reference_target 判定规则
====================

1. reference_flag 与 intent 不同：
- intent = 主教学功能
- reference_flag = 是否伴随明确指向/指代行为

2. reference_flag = true 的条件：
必须满足以下至少两点：
- 视频中存在明确、稳定、可验证的指向动作
- 文本中出现明显指代/观察对象的表达
- 指向目标较清晰（黑板、屏幕、学生等）

3. 以下情况通常不要开启 reference_flag：
- 只是一般朝向黑板/屏幕
- 只是短暂抬手
- 只是开放手势
- 指向动作只在个别弱帧中模糊出现
- 只是“看起来像在指”，但证据不稳定

4. reference_target：
- 没有明确指代行为 -> none
- 有指代行为但目标不明确 -> unknown

====================
十、writing_state 判定规则
====================

1. 只有在视频动作证据明确支持板书/书写时，writing_state = true
2. 不能因为“面朝黑板”就设为 true
3. 若只是靠近黑板或朝向黑板，但没有明显书写动作，必须为 false

====================
十一、interaction_flag 判定规则
====================

1. 以下情况通常设为 true：
- 提问
- 邀请学生回答
- 点名回应
- 明显面向学生发出互动动作

2. 若只是普通叙述或解释，即使朝向学生，也不一定为 true
3. 不要因为文本里出现“同学们”就机械设 true；仍需结合句子功能

====================
十二、sentence_independence 判定规则
====================

1. 若单独看本句就能成立一个完整的教学动作，优先标为 independent。
例如：
- 一个完整问句
- 一个完整提醒句
- 一个完整过渡句
- 一个完整解释句
- 一个完整指代句

2. 只有以下情况才标为 context_dependent：
- 本句语义明显残缺
- 必须依赖前后句才能判断其教学功能
- 脱离上下文后，本句几乎无法独立成立

3. 不要因为一句话“承接上文”就自动判成 context_dependent。
很多问句、提醒句、引导句虽然承接前文，但仍然是独立教学动作。

====================
十三、confidence 判定规则
====================

confidence 不要默认偏高。

建议参考：
- 0.85 ~ 0.95：文本主功能非常明确，音频/视频也一致支持
- 0.70 ~ 0.84：主判断较清楚，但辅助证据有限或存在轻微歧义
- 0.55 ~ 0.69：主判断基本合理，但边界存在明显歧义
- < 0.55：证据不足或冲突较大

当以下情况存在时，应降低 confidence：
- QUESTION 与 REFERENCE 边界不清
- EXPLAIN 与 TRANSITION 边界不清
- sentence_independence 存在明显歧义
- 视频证据较弱但被用于开启 reference_flag
- 主要结论依赖猜测而非明确文本

====================
十四、reason_brief 写法
====================

reason_brief 用 1~2 句，简洁说明：
1. intent 主要依据什么判定
2. rhythm 主要依据什么判定
3. 视频具体修正了哪个辅助字段

要求：
- 不要重复整个输入
- 不要写成长段分析
- 要明确说“主要依据文本 / 主要依据音频 / 视频支持了什么”

====================
十五、最后约束
====================

1. 只能输出一个主 intent
2. 只能输出一个主 primitive
3. 不要让视频中的弱指向动作主导 intent
4. 明确问句通常优先保 QUESTION
5. 解释/叙述/过渡 + 指向动作，通常仍保持原主 intent，只开启 reference_flag
6. 若不确定，保持保守，不要过度判为 REFERENCE
7. 输出必须是单个合法 JSON 对象

当前样本：
__INPUT_JSON__
""".strip()


# =========================
# JSON 修复提示词
# =========================
JSON_REPAIR_TEMPLATE = """
你是一个 JSON 修复器。

任务：
给定一段原始模型输出，请你把它修复成一个“单个、合法、可解析的 JSON 对象”。

要求：
1. 只输出修复后的 JSON 对象。
2. 不要输出 markdown 代码块。
3. 不要输出解释文字。
4. 不要新增原本不存在的字段。
5. 若字段值明显存在但 JSON 语法损坏，应尽量保留原值并修复语法。
6. 输出字段必须只包含以下这些：
- intent
- primitive
- rhythm
- interaction_flag
- reference_flag
- reference_target
- body_orientation
- writing_state
- sentence_independence
- confidence
- reason_brief

原始输出：
__BROKEN_JSON__
""".strip()


# =========================
# 工具函数
# =========================
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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        candidate = brace_match.group(1)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("无法从模型输出中解析 JSON。原始输出如下：\n" + text)


def normalize_semantic_code(obj: Dict[str, Any]) -> Dict[str, Any]:
    allowed_keys = [
        "intent",
        "primitive",
        "rhythm",
        "interaction_flag",
        "reference_flag",
        "reference_target",
        "body_orientation",
        "writing_state",
        "sentence_independence",
        "confidence",
        "reason_brief",
    ]

    normalized = {k: obj.get(k, None) for k in allowed_keys}

    if normalized["reference_target"] is None:
        normalized["reference_target"] = "none"
    if normalized["confidence"] is None:
        normalized["confidence"] = 0.0
    if normalized["reason_brief"] is None:
        normalized["reason_brief"] = ""

    for k in ["interaction_flag", "reference_flag", "writing_state"]:
        v = normalized[k]
        if isinstance(v, bool):
            continue
        if isinstance(v, str):
            normalized[k] = v.strip().lower() == "true"
        else:
            normalized[k] = bool(v)

    try:
        normalized["confidence"] = float(normalized["confidence"])
    except Exception:
        normalized["confidence"] = 0.0

    return normalized


def build_input_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    if "llm_inference_input" in job and isinstance(job["llm_inference_input"], dict):
        payload = job["llm_inference_input"]
    else:
        payload = {
            "task": "infer_teaching_semantic_code",
            "text": job.get("text", ""),
            "start": job.get("start", None),
            "end": job.get("end", None),
            "duration": job.get("duration", None),
            "audio_prosody": job.get("audio_prosody", {}),
            "video_action_evidence": job.get("video_action_evidence", {}),
        }
    return payload


def build_prompt(job: Dict[str, Any]) -> str:
    payload = build_input_payload(job)
    input_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return PROMPT_TEMPLATE.replace("__INPUT_JSON__", input_json)


def build_repair_prompt(raw_text: str) -> str:
    return JSON_REPAIR_TEMPLATE.replace("__BROKEN_JSON__", raw_text)


def call_main_inference(
    client: OpenAI,
    model_name: str,
    job: Dict[str, Any],
) -> str:
    prompt = build_prompt(job)

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        stream=False,
    )

    raw_text = completion.choices[0].message.content
    if not raw_text:
        raise ValueError("模型返回为空。")

    return raw_text


def call_json_repair(
    client: OpenAI,
    model_name: str,
    raw_text: str,
) -> str:
    prompt = build_repair_prompt(raw_text)

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        stream=False,
    )

    repaired = completion.choices[0].message.content
    if not repaired:
        raise ValueError("JSON 修复返回为空。")
    return repaired


def infer_semantic_code_with_repair(
    client: OpenAI,
    model_name: str,
    job: Dict[str, Any],
) -> Dict[str, Any]:
    raw_text = call_main_inference(client, model_name, job)

    try:
        parsed = safe_json_loads(raw_text)
        return {
            "raw_text": raw_text,
            "parsed_json": normalize_semantic_code(parsed),
            "repair_used": False,
            "repair_raw_text": "",
        }
    except Exception as first_err:
        repaired_text = call_json_repair(client, model_name, raw_text)

        try:
            repaired_json = safe_json_loads(repaired_text)
            return {
                "raw_text": raw_text,
                "parsed_json": normalize_semantic_code(repaired_json),
                "repair_used": True,
                "repair_raw_text": repaired_text,
            }
        except Exception as second_err:
            raise ValueError(
                f"首次解析失败: {first_err}\n"
                f"修复后仍失败: {second_err}\n"
                f"原始输出: {raw_text}\n"
                f"修复输出: {repaired_text}"
            )


# =========================
# 主流程
# =========================
def main() -> None:
    api_key = load_api_key()
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    jobs = read_jsonl(INPUT_JSONL)
    done_ids = load_done_ids(OUT_JSONL) if SKIP_DONE else set()

    all_jobs_count = len(jobs)

    if MAX_SAMPLES is None:
        selected_jobs = jobs[START_INDEX:]
    else:
        selected_jobs = jobs[START_INDEX: START_INDEX + MAX_SAMPLES]

    print(f"[INFO] 全部样本数: {all_jobs_count}")
    print(f"[INFO] 本次起始索引: {START_INDEX}")
    print(f"[INFO] 本次计划运行样本数: {len(selected_jobs)}")
    print(f"[INFO] 已完成样本数: {len(done_ids)}")
    print(f"[INFO] 输出文件: {OUT_JSONL}")

    for job in tqdm(selected_jobs, desc="推理教学语义码"):
        seg_id = str(job.get("seg_id", ""))
        if not seg_id:
            continue
        if seg_id in done_ids:
            continue

        try:
            result = infer_semantic_code_with_repair(
                client=client,
                model_name=MODEL_NAME,
                job=job,
            )

            row = {
                "seg_id": seg_id,
                "text": job.get("text", ""),
                "status": "ok",
                "semantic_code": result["parsed_json"],
                "repair_used": result["repair_used"],
                "raw_response": result["raw_text"],
                "repair_raw_response": result["repair_raw_text"],
                "input_payload": build_input_payload(job),
            }
            append_jsonl(OUT_JSONL, row)

        except Exception as e:
            print(f"[ERROR] seg_id={seg_id} | error={e}")
            row = {
                "seg_id": seg_id,
                "text": job.get("text", ""),
                "status": "error",
                "error": str(e),
            }
            append_jsonl(OUT_JSONL, row)

    print("[DONE] 教学语义码推理完成。")


if __name__ == "__main__":
    main()