#!/usr/bin/env python3
"""Single-concurrency streaming benchmark for Qwen3 served by llama.cpp."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_PROMPT = (
    "请用中文从观测选择效应、文明寿命、大过滤器、扩张模型和探测边界五个角度，"
    "分析费米悖论。要求结构清晰、论证紧凑，并给出两个可证伪预测。 /no_think"
)


def percentile_nearest(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def load_api_key(path: Path) -> str:
    value = os.environ.get("LLAMA_API_KEY", "").strip()
    if value:
        return value
    if not path.is_file():
        raise SystemExit(f"API key file not found: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise SystemExit(f"API key file is empty: {path}")
    return value


class GpuSampler:
    def __init__(self, gpu: int, interval: float = 0.2) -> None:
        self.gpu = gpu
        self.interval = interval
        self.samples: list[tuple[float, float, float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        query = "memory.used,memory.total,utilization.gpu,power.draw"
        while not self._stop.is_set():
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        str(self.gpu),
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                parts = [float(part.strip()) for part in proc.stdout.strip().split(",")]
                if len(parts) == 4:
                    self.samples.append(tuple(parts))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval)

    def summary(self) -> dict[str, float | int | None]:
        if not self.samples:
            return {"samples": 0}
        mem_used = [s[0] for s in self.samples]
        mem_total = [s[1] for s in self.samples]
        util = [s[2] for s in self.samples]
        power = [s[3] for s in self.samples]
        return {
            "samples": len(self.samples),
            "memory_peak_mib": round(max(mem_used), 1),
            "memory_total_mib": round(max(mem_total), 1),
            "gpu_util_mean_pct": round(statistics.mean(util), 1),
            "gpu_util_peak_pct": round(max(util), 1),
            "power_mean_w": round(statistics.mean(power), 2),
            "power_peak_w": round(max(power), 2),
        }


def extract_text(delta: dict[str, Any]) -> str:
    reasoning = delta.get("reasoning_content")
    content = delta.get("content")
    return (reasoning if isinstance(reasoning, str) else "") + (
        content if isinstance(content, str) else ""
    )


def tokenize_output(server_root: str, headers: dict[str, str], text: str) -> int | None:
    if not text:
        return 0
    try:
        response = requests.post(
            f"{server_root}/tokenize",
            headers=headers,
            json={"content": text},
            timeout=60,
        )
        response.raise_for_status()
        tokens = response.json().get("tokens")
        return len(tokens) if isinstance(tokens, list) else None
    except requests.RequestException:
        return None


def run_once(
    session: requests.Session,
    api_url: str,
    server_root: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": temperature,
        "top_k": 20,
        "top_p": 0.8,
        "min_p": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    started = time.perf_counter()
    first_token_at: float | None = None
    output_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None

    with session.post(
        api_url,
        headers=headers,
        json=payload,
        stream=True,
        timeout=(30, 600),
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="strict")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            piece = extract_text(delta)
            if piece:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                output_parts.append(piece)

    ended = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("Stream finished without any reasoning_content/content token")

    output_text = "".join(output_parts)
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int):
        completion_tokens = tokenize_output(server_root, headers, output_text)
    prompt_tokens = usage.get("prompt_tokens")
    total_latency = ended - started
    ttft = first_token_at - started
    decode_seconds = max(ended - first_token_at, 1e-9)
    tps = (
        completion_tokens / decode_seconds
        if isinstance(completion_tokens, int) and completion_tokens > 0
        else None
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 4),
        "total_latency_s": round(total_latency, 4),
        "decode_tps": round(tps, 3) if tps is not None else None,
        "finish_reason": finish_reason,
        "output_chars": len(output_text),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--model", default="qwen3-14b-q4km")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path.home() / "llm-deploy/run/llama-api.key",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    api_key = load_api_key(args.api_key_file)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    server_root = args.api_url.split("/v1/", 1)[0].rstrip("/")
    session = requests.Session()

    health = session.get(f"{server_root}/health", timeout=20)
    health.raise_for_status()
    print("Health:", health.text.strip())
    print("Warm-up...")
    run_once(
        session,
        args.api_url,
        server_root,
        headers,
        args.model,
        args.prompt,
        min(args.max_tokens, 64),
        args.temperature,
    )

    sampler = GpuSampler(args.gpu)
    results: list[dict[str, Any]] = []
    sampler.start()
    measured_started = time.perf_counter()
    try:
        for index in range(1, args.runs + 1):
            result = run_once(
                session,
                args.api_url,
                server_root,
                headers,
                args.model,
                args.prompt,
                args.max_tokens,
                args.temperature,
            )
            results.append(result)
            print(f"Run {index}: {json.dumps(result, ensure_ascii=False)}")
    finally:
        measured_elapsed = time.perf_counter() - measured_started
        sampler.stop()

    ttfts = [float(r["ttft_s"]) for r in results]
    latencies = [float(r["total_latency_s"]) for r in results]
    tps_values = [float(r["decode_tps"]) for r in results if r["decode_tps"] is not None]
    prompt_counts = [r["prompt_tokens"] for r in results if isinstance(r["prompt_tokens"], int)]
    completion_counts = [
        r["completion_tokens"] for r in results if isinstance(r["completion_tokens"], int)
    ]
    summary = {
        "runs": len(results),
        "concurrency": 1,
        "prompt_tokens_mean": round(statistics.mean(prompt_counts), 1) if prompt_counts else None,
        "completion_tokens_mean": (
            round(statistics.mean(completion_counts), 1) if completion_counts else None
        ),
        "ttft_mean_s": round(statistics.mean(ttfts), 4),
        "ttft_p95_s": round(percentile_nearest(ttfts, 0.95), 4),
        "decode_tps_mean": round(statistics.mean(tps_values), 3) if tps_values else None,
        "total_latency_mean_s": round(statistics.mean(latencies), 4),
        "request_throughput_rps": round(len(results) / measured_elapsed, 4),
        "gpu": sampler.summary(),
    }
    print("\nFINAL_SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()