"""
Kaggle Benchmark Script for EAGLE Phase 2 Optimizations
Target Problem: 86e8e5 from reference.csv
Parameters: BS=8, MaxLen=65k, Temp=1.0, MinP=0.02, TopP=1.0, Seed=42, StreamInterval=200
"""

import os
import sys
import time
import subprocess
import threading
import queue
import pandas as pd
import json


# ==============================================================================
# LOGGING SETUP
# ==============================================================================
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


# Redirect stdout and stderr to benchmark_log.txt
LOG_FILE = "/kaggle/working/benchmark_log.txt"
sys.stdout = TeeLogger(LOG_FILE)
sys.stderr = sys.stdout

# Add working directory to path
sys.path.append("/kaggle/working")

# Benchmark Parameters
SEED = 42
MAX_MODEL_LEN = 65536
BATCH_SIZE = 8
TEMPERATURE = 1.0
TOP_P = 1.0
MIN_P = 0.02
STREAM_INTERVAL = 200
TARGET_PROBLEM_ID = "86e8e5"

# Model paths (adjust to your Kaggle input paths)
MODEL_PATH = "/kaggle/input/gpt-oss-120b/transformers/default/1"
DRAFT_MODEL_PATH = "/kaggle/input/gpt/transformers/gpt-oss-120b-eagle3/1"

print(f"{'='*80}")
print(f"EAGLE Phase 2 Benchmark - Problem {TARGET_PROBLEM_ID}")
print(f"Phase 2 Optimized: {os.environ.get('VLLM_EAGLE_PHASE2_FUSED')}")
print(f"{'='*80}\n")


def start_vllm_server():
    """Start vLLM server with EAGLE speculative decoding"""
    command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_PATH,
        "--served-model-name",
        "gpt-oss",
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        str(BATCH_SIZE),
        "--gpu-memory-utilization",
        "0.95",
        "--max-cudagraph-capture-size",
        "2048",
        "--speculative-config",
        f'{{"model":"{DRAFT_MODEL_PATH}","num_speculative_tokens":5,"method":"eagle3","draft_tensor_parallel_size":1}}',
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--stream-interval",
        str(STREAM_INTERVAL),
        "--trust-remote-code",
    ]

    vllm_proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )

    def stream_output():
        for line in vllm_proc.stdout:
            # This print will go to the TeeLogger
            print(f"[vLLM] {line}", end="", flush=True)

    threading.Thread(target=stream_output, daemon=True).start()
    return vllm_proc


def wait_for_server(client, max_attempts=60):
    """Wait for vLLM server to be ready"""
    print("Waiting for vLLM server to start...")
    for i in range(max_attempts):
        try:
            client.models.list()
            print("✓ Server is READY.")
            return True
        except:
            time.sleep(30)
    print("✗ Server failed to start.")
    return False


def run_benchmark():
    """Run TIR benchmark on problem 86e8e5"""
    from openai import OpenAI
    from transformers import set_seed
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        ReasoningEffort,
        Author,
        TextContent,
    )
    from local_python_tool import PythonTool

    set_seed(SEED)

    # Start server
    vllm_proc = start_vllm_server()
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-local")

    if not wait_for_server(client):
        os.killpg(os.getpgid(vllm_proc.pid), 9)
        return

    # Load problem
    df = pd.read_csv(
        "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
    )
    problem_row = df[df["id"] == TARGET_PROBLEM_ID].iloc[0]
    problem_text = problem_row["problem"]
    ground_truth = problem_row["answer"]

    print(f"\n{'='*80}")
    print(f"Problem {TARGET_PROBLEM_ID}:")
    print(f"{problem_text[:200]}...")
    print(f"Ground Truth: {ground_truth}")
    print(f"{'='*80}\n")

    # Initialize encoding and tools
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    python_pool = queue.Queue()
    for _ in range(BATCH_SIZE):
        python_pool.put(PythonTool(execution_backend="jupyter"))

    # Inferencer
    class SyncInferencer:
        def generate_tir(self, problem: str, k_idx: int):
            python_tool = python_pool.get()
            messages = [
                Message(
                    author=Author(role=Role.SYSTEM, name="system"),
                    content=[
                        SystemContent.new()
                        .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
                        .with_tools(python_tool.tool_config)
                    ],
                ),
                Message(
                    author=Author(role=Role.USER, name="user"),
                    content=[TextContent(text=problem)],
                ),
            ]

            total_tool_wait = 0.0
            total_tokens = 0
            start_gen = time.perf_counter()

            for turn in range(10):  # Max turns
                prompt_ids = encoding.render_conversation_for_completion(
                    Conversation.from_messages(messages), Role.ASSISTANT
                )

                response = client.completions.create(
                    model="gpt-oss",
                    prompt=prompt_ids,
                    max_tokens=2048,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    seed=SEED + k_idx,
                    extra_body=dict(
                        min_p=MIN_P,
                        stop_token_ids=stop_token_ids,
                        return_token_ids=True,
                    ),
                    timeout=600,
                )

                token_ids = response.choices[0].token_ids
                total_tokens += len(token_ids)

                new_msgs = encoding.parse_messages_from_completion_tokens(
                    token_ids, Role.ASSISTANT
                )
                messages.extend(new_msgs)

                last_msg = messages[-1]
                if last_msg.channel == "final" or token_ids[-1] == 200002:  # EOT
                    break

                if last_msg.recipient == "python":
                    code = last_msg.content[0].text
                    py_start = time.perf_counter()
                    output = python_tool.execute_code(code)
                    total_tool_wait += time.perf_counter() - py_start

                    output_msg = Message(
                        author=Author(role=Role.TOOL, name="functions.python"),
                        content=[TextContent(text=output)],
                    ).with_recipient("assistant")
                    messages.append(output_msg)

            python_pool.put(python_tool)
            total_time = time.perf_counter() - start_gen
            return messages, total_time, total_tool_wait, total_tokens

    # Run parallel samples
    inferencer = SyncInferencer()
    parallel_results = [None] * BATCH_SIZE

    def worker(i):
        try:
            parallel_results[i] = inferencer.generate_tir(problem_text, i)
        except Exception as e:
            print(f"Worker {i} error: {e}")
            parallel_results[i] = (None, 0, 0, 0)

    print(f"Running {BATCH_SIZE} parallel samples...")
    q_start = time.perf_counter()

    threads = []
    for i in range(BATCH_SIZE):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    q_elapsed = time.perf_counter() - q_start

    # Calculate metrics
    total_tokens = sum(r[3] for r in parallel_results if r[3])
    avg_tokens = total_tokens / BATCH_SIZE
    global_tps = total_tokens / q_elapsed if q_elapsed > 0 else 0

    # Print results
    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS - Problem {TARGET_PROBLEM_ID}")
    print(f"{'='*80}")
    print(f"Phase 2 Optimized: {os.environ.get('VLLM_EAGLE_PHASE2_FUSED')}")
    print(f"Total Wall-Clock: {q_elapsed:.2f}s")
    print(f"Total Tokens: {total_tokens}")
    print(f"Avg Tokens/Sample: {avg_tokens:.1f}")
    print(f"Global Decode Throughput: {global_tps:.2f} tok/s")
    print(f"{'='*80}\n")

    # ==============================================================================
    # DUMP FULL OUTPUTS TO JSON
    # ==============================================================================
    results_dump = []
    for i, res in enumerate(parallel_results):
        if res[0] is not None:
            # Convert Harmony messages to serializable dicts
            convo = []
            for msg in res[0]:
                content_list = []
                for c in msg.content:
                    if hasattr(c, "text"):
                        content_list.append({"type": "text", "text": c.text})
                    else:
                        content_list.append({"type": "other", "repr": repr(c)})

                convo.append(
                    {
                        "role": str(msg.author.role),
                        "name": msg.author.name,
                        "recipient": msg.recipient,
                        "content": content_list,
                    }
                )

            results_dump.append(
                {
                    "sample_id": i,
                    "total_time": res[1],
                    "tool_wait": res[2],
                    "tokens": res[3],
                    "conversation": convo,
                }
            )

    output_json = f"/kaggle/working/results_{TARGET_PROBLEM_ID}_{int(time.time())}.json"
    with open(output_json, "w") as f:
        json.dump(results_dump, f, indent=2)

    print(f"✓ Full conversation results dumped to: {output_json}")
    print(f"✓ Console log saved to: {LOG_FILE}")

    # Cleanup
    os.killpg(os.getpgid(vllm_proc.pid), 9)


if __name__ == "__main__":
    run_benchmark()
