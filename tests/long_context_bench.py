import os
import json
import time
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8001"
MODEL = "qwen3-14b-q4km"
API_KEY = os.environ["LLAMA_API_KEY"]

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

ANCHORS = [
    """
[权威记录 A1]
远航项目位于Z区域，风险等级为7；提交时间17:40；
已声明紧急状态，且本记录为正式案件属性。
""",
    """
[已废止记录 AUDIT-V1]
早期审计结论为“不通过”。该记录已被后续版本替代，不得作为最终依据。
""",
    """
[现行规则 P2]
项目只有在最新审计通过，并且满足以下任一条件时具有基础资格：
① 风险等级不高于6；
② 已声明紧急状态，并具有两名有效签署人。
""",
    """
[签署记录 S3]
李明和王澜已经分别完成有效签署；二人的签署均未撤销。
""",
    """
[区域规则 Z4]
Z区域项目在17:00之后提交，原则上必须取得主管额外批准。
""",
    """
[特别修订 E5]
对于Z区域紧急项目，如果在18:00前提交、风险等级低于8，
并且已有两名有效签署人，则免除Z4规定的主管额外批准。
E5发布时间晚于Z4，且特别规则优先于一般规则。
""",
    """
[最新审计 AUDIT-V3]
远航项目的最新审计结论为“通过”。
根据版本规则，V3取代AUDIT-V1。
""",
]

QUESTION = """
请根据以上材料判断远航项目最终应当“批准”还是“拒绝”。

要求：
1. 第一行严格写“结论：批准”或“结论：拒绝”。
2. 引用相关记录编号。
3. 说明AUDIT-V1为何不能作为最终依据。
4. 不使用无关填充记录。
5. 给出简洁、可核验的推理，不超过六个要点。
"""


def tokenize(text):
    response = requests.post(
        f"{BASE_URL}/tokenize",
        headers=HEADERS,
        json={"content": text},
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    tokens = data.get("tokens", [])
    return len(tokens)


def assemble(filler_count, nonce):
    fillers = [
        (
            f"无关档案{i:06d}：仓储区例行检查编号"
            f"{(i * 7919) % 100003:06d}，内容与当前审批案件无关，"
            "不得据此修改任何项目结论。\n"
        )
        for i in range(filler_count)
    ]

    blocks = [f"测试批次：{nonce}\n"]
    cursor = 0
    gap_count = len(ANCHORS) + 1
    per_gap, remainder = divmod(filler_count, gap_count)

    for position, anchor in enumerate(ANCHORS):
        take = per_gap + (1 if position < remainder else 0)
        blocks.append("".join(fillers[cursor:cursor + take]))
        cursor += take
        blocks.append(anchor)

    blocks.append("".join(fillers[cursor:]))
    blocks.append(QUESTION)
    return "".join(blocks)


# 估计每条填充记录的Token数
CAL_0 = tokenize(assemble(0, "calibration"))
CAL_200 = tokenize(assemble(200, "calibration"))
TOKENS_PER_LINE = max((CAL_200 - CAL_0) / 200, 1)


def build_target_prompt(target_tokens, nonce):
    filler_count = max(
        0,
        int((target_tokens - CAL_0) / TOKENS_PER_LINE)
    )

    prompt = ""
    count = 0

    for _ in range(5):
        prompt = assemble(filler_count, nonce)
        count = tokenize(prompt)
        difference = target_tokens - count

        if abs(difference) <= 64:
            break

        filler_count = max(
            0,
            filler_count + int(difference / TOKENS_PER_LINE)
        )

    return prompt, count


def run_test(target_tokens, thinking, max_tokens):
    nonce = f"{target_tokens}-{thinking}-{time.time_ns()}"
    prompt, raw_prompt_tokens = build_target_prompt(target_tokens, nonce)

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "top_p": 0.8,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }

    started = time.perf_counter()
    first_token_time = None
    content_parts = []
    reasoning_parts = []
    usage = {}
    finish_reason = None

    with requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        stream=True,
        timeout=(30, 1800),
    ) as response:
        response.encoding = "utf-8"
        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP {response.status_code}: {response.text[:1000]}"
            )

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue

            data = line[5:].strip()
            if data == "[DONE]":
                continue

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            if event.get("usage"):
                usage = event["usage"]

            choices = event.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}

            reasoning = (
                delta.get("reasoning_content")
                or delta.get("reasoning")
                or ""
            )
            content = delta.get("content") or ""

            if first_token_time is None and (reasoning or content):
                first_token_time = time.perf_counter()

            if reasoning:
                reasoning_parts.append(reasoning)
            if content:
                content_parts.append(content)

    ended = time.perf_counter()

    answer = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    combined = reasoning + answer

    ttft = (
        first_token_time - started
        if first_token_time is not None
        else None
    )
    total_latency = ended - started

    prompt_tokens = usage.get("prompt_tokens", raw_prompt_tokens)
    completion_tokens = usage.get("completion_tokens")

    if completion_tokens is None:
        completion_tokens = tokenize(combined) if combined else 0

    decode_time = (
        total_latency - ttft
        if ttft is not None
        else total_latency
    )
    tps = completion_tokens / decode_time if decode_time > 0 else 0
    prefill_tps = prompt_tokens / ttft if ttft and ttft > 0 else 0

    required_evidence = ["P2", "S3", "E5", "AUDIT-V3"]
    quality_pass = (
        "结论：批准" in answer
        and all(item in answer for item in required_evidence)
        and "AUDIT-V1" in answer
    )

    result = {
        "target_tokens": target_tokens,
        "actual_prompt_tokens": prompt_tokens,
        "thinking": thinking,
        "max_tokens": max_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 3) if ttft else None,
        "prefill_est_tps": round(prefill_tps, 2),
        "decode_tps": round(tps, 2),
        "total_latency_s": round(total_latency, 3),
        "finish_reason": finish_reason,
        "quality_pass": quality_pass,
        "answer": answer,
        "reasoning": reasoning,
    }

    print("\n" + "=" * 70)
    print(json.dumps(
        {k: v for k, v in result.items()
         if k not in ("answer", "reasoning")},
        ensure_ascii=False,
        indent=2,
    ))
    print("\nANSWER:")
    print(answer[-3000:])

    return result


print("Warm-up...")
run_test(2000, thinking=False, max_tokens=128)

results = []

# 长上下文性能和检索测试
for target in (8000, 32000, 64000, 88000):
    results.append(
        run_test(target, thinking=False, max_tokens=512)
    )

# 88K思考模式推理质量测试
results.append(
    run_test(88000, thinking=True, max_tokens=3072)
)

output_path = (
    Path.home()
    / "llm-deploy"
    / "logs"
    / f"long-context-bench-{int(time.time())}.json"
)

output_path.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"\n完整结果：{output_path}")