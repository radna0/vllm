# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import urllib.request

import pytest


def _call_completion(base_url: str, model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 32,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["text"]


@pytest.mark.skipif(
    os.getenv("EAGLE_PARITY_TEST", "0") != "1",
    reason="Set EAGLE_PARITY_TEST=1 to run parity checks.",
)
def test_greedy_parity_vllm_trtllm():
    vllm_url = os.getenv("VLLM_BASE_URL")
    trtllm_url = os.getenv("TRTLLM_BASE_URL")
    model = os.getenv("EAGLE_PARITY_MODEL", "model")
    if not vllm_url or not trtllm_url:
        pytest.skip("Set VLLM_BASE_URL and TRTLLM_BASE_URL to run parity checks.")

    prompt = "Write a short sentence about H100 GPUs."
    vllm_text = _call_completion(vllm_url, model, prompt)
    trtllm_text = _call_completion(trtllm_url, model, prompt)
    assert vllm_text == trtllm_text
