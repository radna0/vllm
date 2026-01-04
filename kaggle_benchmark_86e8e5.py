"""
Full TIR Correctness Benchmark for EAGLE
Target Problem: 424e18 (Runs tournament)
Ground Truth: 21818
Benchmark focus: Quality & Correctness via Multi-Turn TIR Reasoning
"""

import os
import sys
import time
import subprocess
import threading
import queue
import re
import json
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


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


LOG_FILE = "/kaggle/working/correctness_benchmark_log.txt"
sys.stdout = TeeLogger(LOG_FILE)
sys.stderr = sys.stdout

sys.path.append("/kaggle/working")

# ==============================================================================
# BENCHMARK PARAMETERS
# ==============================================================================
SEED = 42
MAX_MODEL_LEN = 65536
K = 8  # Number of parallel samples
TEMPERATURE = 1.0
TOP_P = 1.0
MIN_P = 0.02
MAX_ITER = 100  # Safety limit (actual limit is context length)
TARGET_PROBLEM_ID = "424e18"

# Model paths
MODEL_PATH = "/kaggle/input/gpt-oss-120b/transformers/default/1"
DRAFT_MODEL_PATH = "/kaggle/input/gpt/transformers/gpt-oss-120b-eagle3/1"

# TIR Prompt (exactly as in Kaggle notebook)
TIR_PROMPT = """Please reason step by step and use the python tool to solve the math problem.
Finally, Return only the verified final answer in \\boxed{}, where the answer is an integer in [0, 99999]. Never guess."""

print(f"{'='*80}")
print(f"FULL TIR CORRECTNESS BENCHMARK - Problem {TARGET_PROBLEM_ID}")
print(f"K={K} samples, MAX_ITER={MAX_ITER}, TEMP={TEMPERATURE}")
print(f"{'='*80}\n")


# ==============================================================================
# LOCAL JUPYTER TOOL (Simplified from Harmony Notebook)
# ==============================================================================
class LocalJupyterSession:
    """Stateful Jupyter kernel session for code execution."""

    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count=5):
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            if cls._next_port > 65000:
                cls._next_port = 50000
            return ports

    def __init__(self, timeout=120.0):
        try:
            from jupyter_client import KernelManager
        except ImportError:
            raise RuntimeError("jupyter_client required")

        self._default_timeout = timeout
        ports = self._get_next_ports(5)

        km = KernelManager()
        km.shell_port = ports[0]
        km.iopub_port = ports[1]
        km.stdin_port = ports[2]
        km.hb_port = ports[3]
        km.control_port = ports[4]
        km.start_kernel()

        self._client = km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._km = km

    def execute(self, code, timeout=None):
        import queue as _queue

        effective_timeout = float(timeout or self._default_timeout)
        msg_id = self._client.execute(
            code, store_history=True, allow_stdin=False, stop_on_error=False
        )

        stdout_parts = []
        stderr_parts = []
        start = time.time()
        poll = 0.5

        while True:
            if (time.time() - start) >= effective_timeout:
                try:
                    self._km.interrupt_kernel()
                except:
                    pass
                raise TimeoutError(f"Timeout: {effective_timeout:.1f}s")

            try:
                msg = self._client.get_iopub_msg(timeout=poll)
            except _queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                tb = content.get("traceback", [])
                stderr_parts.append(
                    "\n".join(tb)
                    if tb
                    else f"{content.get('ename')}: {content.get('evalue')}"
                )
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        # Wait for shell reply
        while True:
            if (time.time() - start) >= effective_timeout:
                break
            try:
                reply = self._client.get_shell_msg(timeout=poll)
            except _queue.Empty:
                continue
            if reply.get("parent_header", {}).get("msg_id") == msg_id:
                break

        output = "".join(stdout_parts)
        if stderr_parts:
            output = output.rstrip() + "\n" + "".join(stderr_parts)
        return output if output.strip() else "[WARN] No output"

    def close(self):
        try:
            self._client.stop_channels()
            self._km.shutdown_kernel(now=True)
        except:
            pass


class PythonTool:
    def __init__(self):
        self._session = None
        self._lock = threading.Lock()

    def _ensure_session(self):
        if self._session is None:
            self._session = LocalJupyterSession(timeout=60.0)

    @property
    def tool_config(self):
        from openai_harmony import ToolNamespaceConfig

        return ToolNamespaceConfig(
            name="python",
            description="Execute Python code in a stateful Jupyter kernel.",
            tools=[],
        )

    def execute(self, code, timeout=60.0):
        self._ensure_session()
        with self._lock:
            try:
                return self._session.execute(code, timeout=timeout)
            except Exception as e:
                return f"[ERROR] {e}"

    def close(self):
        if self._session:
            self._session.close()
            self._session = None


# ==============================================================================
# VLLM SERVER
# ==============================================================================
def start_vllm_server():
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
        "64",
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
        "auto",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--stream-interval",
        "20",
        "--trust-remote-code",
    ]

    logfile_path = "/kaggle/working/vllm.log"
    with open(logfile_path, "w") as logfile:
        process = subprocess.Popen(
            command, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True
        )
    print(f"vLLM server started. Logs: {logfile_path}")
    return process


# ==============================================================================
# EXTRACT ANSWER
# ==============================================================================
def extract_boxed_text(text):
    pattern = r"oxed{(.*?)}"
    matches = re.findall(pattern, str(text))
    if matches:
        for match in reversed(matches):
            if match:
                try:
                    clean = match.strip().replace(",", "").replace(" ", "")
                    # Handle decimals or scientific notation safely
                    val = int(float(clean[:20]))
                    if 0 <= val <= 99999:
                        return val
                except:
                    pass
    return None


# ==============================================================================
# MAIN BENCHMARK
# ==============================================================================
def run_benchmark():
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
        RenderConversationConfig,
    )

    set_seed(SEED)

    # Start server
    vllm_proc = start_vllm_server()
    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1", api_key="sk-local", timeout=360
    )

    # Wait for server
    print("Waiting for vLLM server...")
    for _ in range(30):
        time.sleep(30)
        try:
            client.models.list()
            print("✓ Server is READY.")
            break
        except:
            continue
    else:
        print("✗ Server failed to start.")
        os.killpg(os.getpgid(vllm_proc.pid), 9)
        return

    # Load problem
    try:
        df = pd.read_csv(
            "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
        )
        problem_row = df[df["id"] == TARGET_PROBLEM_ID].iloc[0]
        problem_text = problem_row["problem"]
        ground_truth = problem_row["answer"]
    except Exception as e:
        print(f"Error loading problem: {e}")
        # Fallback if CSV missiong
        problem_text = "Problem 424e18 missing in CSV"
        ground_truth = 21818

    print(f"\n{'='*80}")
    print(f"Target Problem: {TARGET_PROBLEM_ID}")
    print(f"{problem_text[:500]}...")
    print(f"Ground Truth: {ground_truth}")
    print(f"{'='*80}\n")

    # Initialize encoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    # Python tool pool
    python_pool = queue.Queue(maxsize=K)
    for _ in range(K):
        python_pool.put(PythonTool())

    # -------------------------------------------------------------------------
    # TIR GENERATION OBJECTIVE
    # -------------------------------------------------------------------------
    def single_generate_tir(prompt, seed_offset=0):
        tool = python_pool.get()
        tool._ensure_session()

        messages = [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
                .with_tools(tool.tool_config),
            ),
            Message.from_role_and_content(Role.USER, prompt),
        ]

        total_tokens = 0
        iterations = 0
        final_answer = None

        print(f"[{seed_offset}] Starting reasoning...")

        # IMPORTANT: Like the original Kaggle notebook, iterations continue until:
        # 1. Context is full (max_tokens < 100)
        # 2. Final answer is found (\\boxed{})
        # 3. MAX_ITER safety limit is reached
        # The actual limit is the 65k context window, not MAX_ITER.

        for turn in range(MAX_ITER):
            iterations = turn + 1

            p_ids = encoding.render_conversation_for_completion(
                Conversation.from_messages(messages), Role.ASSISTANT
            )
            max_tokens = MAX_MODEL_LEN - len(p_ids)
            if max_tokens < 100:
                print(f"[{seed_offset}] Max context reached.")
                break

            try:
                # Streaming for early stop
                stream = client.completions.create(
                    model="gpt-oss",
                    prompt=p_ids,
                    max_tokens=2048,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    seed=SEED + seed_offset,
                    stream=True,
                    extra_body=dict(
                        min_p=MIN_P,
                        stop_token_ids=stop_token_ids,
                        return_token_ids=True,
                    ),
                    timeout=300,
                )

                t_ids = []
                t_text = ""
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    chunk_ids = getattr(chunk.choices[0], "token_ids", [])
                    chunk_text = getattr(chunk.choices[0], "text", "")
                    t_ids.extend(chunk_ids)
                    t_text += chunk_text

                    if "}" in chunk_text:
                        if extract_boxed_text(t_text) is not None:
                            break

                stream.close()
            except Exception as e:
                print(f"[{seed_offset}] Error: {e}")
                break

            if not t_ids:
                break
            total_tokens += len(t_ids)

            new_msgs = encoding.parse_messages_from_completion_tokens(
                t_ids, Role.ASSISTANT
            )
            messages.extend(new_msgs)

            last = messages[-1]
            if last.channel == "final" or t_ids[-1] == 200002:
                final_answer = extract_boxed_text(t_text)
                break

            if last.recipient == "python":
                code = last.content[0].text
                output = tool.execute(code)
                messages.append(
                    Message(
                        author=Author(role=Role.TOOL, name="python"),
                        content=[TextContent(text=output)],
                    ).with_recipient("assistant")
                )

        python_pool.put(tool)
        return {"answer": final_answer, "iter": iterations, "tokens": total_tokens}

    # -------------------------------------------------------------------------
    # RUN PARALLEL
    # -------------------------------------------------------------------------
    print(f"Running {K} parallel reasoning threads...")
    start_time = time.time()
    results = [None] * K

    def worker(i):
        results[i] = single_generate_tir(problem_text + "\n\n" + TIR_PROMPT, i)

    threads = []
    for i in range(K):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    elapsed = time.time() - start_time

    # -------------------------------------------------------------------------
    # AGGREGATE & REPORT
    # -------------------------------------------------------------------------
    ans_list = [r["answer"] for r in results if r]
    valid_ans = [a for a in ans_list if a is not None]

    if valid_ans:
        c = Counter(valid_ans)
        top_ans, top_cnt = c.most_common(1)[0]
    else:
        top_ans, top_cnt = 0, 0

    is_correct = top_ans == ground_truth

    print(f"\n{'='*80}")
    print(f"FINAL CORRECTNESS RESULTS - {TARGET_PROBLEM_ID}")
    print(f"{'='*80}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Model Plurality: {top_ans} (Frequency: {top_cnt}/{K})")
    print(f"Correct: {'✅ YES' if is_correct else '❌ NO'}")
    print(f"Total Time: {elapsed:.2f}s")
    print(f"Answers: {dict(Counter(ans_list))}")
    print(f"{'='*80}\n")

    output_data = {
        "problem_id": TARGET_PROBLEM_ID,
        "is_correct": bool(is_correct),
        "ground_truth": int(ground_truth),
        "model_answer": int(top_ans),
        "accuracy": top_cnt / K,
        "time": elapsed,
        "raw_results": results,
    }

    with open("/kaggle/working/correctness_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    # Cleanup
    os.killpg(os.getpgid(vllm_proc.pid), 9)


if __name__ == "__main__":
    run_benchmark()
