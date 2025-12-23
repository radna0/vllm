import sys
from unittest.mock import MagicMock, Mock
import asyncio
from collections import deque
import types
import typing

# Mock vllm._C to avoid import error
sys.modules["vllm.platforms.cuda"] = MagicMock()
sys.modules["vllm._C"] = MagicMock()

def mock_module(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

mock_module("vllm.model_executor")
models_mod = mock_module("vllm.model_executor.models")
models_mod.SupportsMultiModal = MagicMock()
mock_module("vllm.model_executor.layers")
batch_inv = mock_module("vllm.model_executor.layers.batch_invariant")
batch_inv.vllm_is_batch_invariant = MagicMock()
q_mod = mock_module("vllm.model_executor.layers.quantization")

from enum import Enum
class QuantizationMethods(str, Enum): # Mock Enum for type hints
    AWQ = "awq"
    GPTQ = "gptq"
    SQUEEZELLM = "squeezellm"
q_mod.QuantizationMethods = QuantizationMethods
q_mod.QUANTIZATION_METHODS = ["awq", "gptq", "squeezellm"] 

mock_module("vllm.model_executor.layers.quantization.input_quant_fp8")
lin_mod = mock_module("vllm.model_executor.layers.linear")
lin_mod.LinearBase = MagicMock()
lin_mod.UnquantizedLinearMethod = MagicMock()
lin_mod.ColumnParallelLinear = MagicMock()
lin_mod.RowParallelLinear = MagicMock()
lin_mod.QKVParallelLinear = MagicMock()
lin_mod.MergedColumnParallelLinear = MagicMock()

attn_mod = mock_module("vllm.model_executor.layers.attention_layer_base")
attn_mod.AttentionLayerBase = MagicMock()

# Patch direct_register_custom_op to avoid torch library errors
def no_op(*args, **kwargs):
    pass

import vllm.utils.torch_utils
vllm.utils.torch_utils.direct_register_custom_op = no_op

# Patch ModelConfig to avoid Pydantic validation errors
import vllm.config.model
OriginalModelConfig = vllm.config.model.ModelConfig

class MockModelConfig(OriginalModelConfig):
    def __init__(self, *args, **kwargs):
        self.max_model_len = 2048
        self.hf_config = Mock()
        self.hf_config.is_encoder_decoder = False
        self._architecture = "OPTForCausalLM" # Satisfy architecture property
        self.hf_text_config = Mock()
        self.model = kwargs.get('model', 'mock-model')
        self.tokenizer = kwargs.get('tokenizer', 'mock-tokenizer')
        self.tokenizer_mode = kwargs.get('tokenizer_mode', 'auto')
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        self.dtype = kwargs.get('dtype', 'auto')
        self.seed = kwargs.get('seed', 0)
        self.revision = kwargs.get('revision', None)
        self.tokenizer_revision = kwargs.get('tokenizer_revision', None)
        self.quantization = kwargs.get('quantization', None)
        self.enforce_eager = kwargs.get('enforce_eager', False)
        self.max_context_len_to_capture = kwargs.get('max_context_len_to_capture', 2048)
        self.max_seq_len_to_capture = kwargs.get('max_seq_len_to_capture', 2048)
        self.skip_tokenizer_init = kwargs.get('skip_tokenizer_init', False)

    @property
    def is_encoder_decoder(self):
        return False
    @property
    def is_multimodal_model(self):
        return False
    @property
    def is_hybrid(self):
        return False
    @property
    def convert_type(self):
        return None
    @property
    def attention_chunk_size(self):
        return None
    @property
    def disable_cascade_attn(self):
        return False
    def get_hidden_size(self):
        return 1024
    def verify_with_parallel_config(self, parallel_config):
        pass
    def get_num_layers(self, parallel_config):
        return 2
    def get_total_num_attention_heads(self, parallel_config):
        return 16
    def verify_dual_chunk_attention_config(self, load_config):
        pass
    def verify_quantization(self, load_config):
        pass
    @property
    def runner_type(self):
        return "generate"

vllm.config.model.ModelConfig = MockModelConfig

# Patch tokenizer loader
import vllm.tokenizers.registry
vllm.tokenizers.registry.cached_tokenizer_from_config = lambda *args, **kwargs: MagicMock()

# Mock StructuredOutputManager to avoid tokenizer loading
struct_out = mock_module("vllm.v1.structured_output")
struct_out.StructuredOutputManager = MagicMock()
struct_req = mock_module("vllm.v1.structured_output.request")
struct_req.StructuredOutputRequest = MagicMock()

import vllm.config
vllm.config.ModelConfig = MockModelConfig

import vllm.config.vllm
vllm.config.vllm.VllmConfig.try_verify_and_update_config = lambda self: None

import torch
torch.cuda.is_available = lambda: False
original_torch_device = torch.device
class MockDevice:
    def __init__(self, arg):
        pass
    def __or__(self, other):
        return typing.Union[MockDevice, other]
    def __ror__(self, other):
        return typing.Union[other, MockDevice]
torch.device = MockDevice

try:
    from tests.v1.core.utils import create_requests, create_scheduler
    from vllm.v1.request import RequestStatus
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

async def test_scheduler_critical_section():
    print("Running Scheduler Verification Tests...")
    try:
        # TEST 1: Critical Section & Interrupts
        print("\n=== Test 1: Critical Section & Interrupts ===")
        scheduler = create_scheduler(max_num_seqs=10, max_num_batched_tokens=100)
        requests = create_requests(num_requests=1, num_tokens=10)
        request = requests[0]
        scheduler.add_request(request)
        
        # Step 1: Start request (RUNNING)
        out = scheduler.schedule()
        assert len(out.scheduled_new_reqs) == 1
        assert request.status == RequestStatus.RUNNING
        print("Request scheduled.")

        # Step 2: Enter Critical Section via API
        scheduler.update_critical_sections({request.request_id: True})
        assert request.is_critical_section
        print(f"Set Critical Section: {request.is_critical_section}")

        # Step 3: Add Interrupt - Should be Queued
        interrupt_tokens = [200, 201]
        scheduler.add_interrupt(request.request_id, interrupt_tokens)
        
        # Verify
        assert len(request.tool_state.pending_interrupts) == 1
        assert request.status == RequestStatus.RUNNING
        assert 200 not in request.output_token_ids
        print("Interrupt queued correctly (not applied).")

        # Step 4: Exit Critical Section via API - Should Trigger Interrupt
        scheduler.update_critical_sections({request.request_id: False})
        print("Exited Critical Section.")

        # Verify Interrupt Applied Automatically
        assert not request.is_critical_section
        assert len(request.tool_state.pending_interrupts) == 0
        assert request.status == RequestStatus.WAITING
        assert request in scheduler.waiting
        print("Interrupt applied automatically after exit critical section.")

        # TEST 2: Trap Logic
        print("\n=== Test 2: Trap Logic ===")
        scheduler = create_scheduler(max_num_seqs=10, max_num_batched_tokens=100)
        requests = create_requests(num_requests=1, num_tokens=10)
        req_trap = requests[0]
        scheduler.add_request(req_trap)
        scheduler.schedule()
        assert req_trap.status == RequestStatus.RUNNING
        
        print("Trapping request...")
        scheduler.trap_requests([req_trap.request_id])
        
        assert req_trap.status == RequestStatus.WAIT_TRAP
        assert req_trap not in scheduler.running
        assert req_trap in scheduler.waiting_for_tool
        assert req_trap.tool_state.trap_seen
        print("Trap successful: Request removed from running and set to WAIT_TRAP.")

        # TEST 3: Resume from Trap via Add Interrupt
        print("\n=== Test 3: Resume from Trap ===")
        interrupt_tokens = [200, 201]  
        scheduler.add_interrupt(req_trap.request_id, interrupt_tokens)
        
        assert req_trap.status == RequestStatus.WAITING
        assert req_trap not in scheduler.waiting_for_tool
        # Note: scheduler.waiting is a deque usually
        assert req_trap in scheduler.waiting
        assert not req_trap.tool_state.trap_seen
        # Check tokens appended
        # We need to access the underlying list because ConstantList might not support 'in' directly? 
        # Or it does.
        assert 200 in req_trap.output_token_ids
        print("Resume successful: Request moved to WAITING and tokens appended.")

        print("\nALL TESTS PASSED!")

    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_scheduler_critical_section())
