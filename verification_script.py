import sys
from unittest.mock import MagicMock, Mock
import asyncio

# Mock vllm._C to avoid import error
sys.modules["vllm.platforms.cuda"] = MagicMock()
sys.modules["vllm._C"] = MagicMock()
import types
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

# We simply define a class that mimics the structure but don't decorate it as dataclass 
# if we inherit from a dataclass, correct? Or we do?
# If Original is a dataclass, we should be careful.
# But we just need isinstance to pass.

class MockModelConfig(OriginalModelConfig):
    def __init__(self, *args, **kwargs):
        # We DO NOT call super().__init__ to avoid validation
        self.max_model_len = 2048
        self.hf_config = Mock()
        self.hf_config.is_encoder_decoder = False
        self._architecture = "OPTForCausalLM" # Satisfy architecture property
        self.hf_text_config = Mock()
        
        # Set all potential fields to avoid AttributeError
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

# Also patch vllm.config.ModelConfig since utils imports from there
import vllm.config
vllm.config.ModelConfig = MockModelConfig

# Patch VllmConfig.try_verify_and_update_config to avoid deep imports
import vllm.config.vllm
vllm.config.vllm.VllmConfig.try_verify_and_update_config = lambda self: None

# Also mock torch.cuda for CPU only test

# Also mock torch.cuda for CPU only test
import torch
torch.cuda.is_available = lambda: False  # Force CPU?
# Mock torch.device to accept MagicMock from DeviceConfig and be usable in type hints
original_torch_device = torch.device
class MockDevice:
    def __init__(self, arg):
        pass
    def __or__(self, other):
        return typing.Union[MockDevice, other]
    def __ror__(self, other):
        return typing.Union[other, MockDevice]

# We need to make sure torch.device is a type
torch.device = MockDevice

import typing
# We also need to make sure it works when instantiated
# But wait, type hints use the class. Instantiation uses the class constructor.
# This should work.

# We need to import checking that create_scheduler works
try:
    from tests.v1.core.utils import create_requests, create_scheduler
    from vllm.v1.request import RequestStatus
    from vllm.v1.outputs import ModelRunnerOutput
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

async def test_scheduler_add_interrupt():
    print("Initializing scheduler...")
    try:
        # 1. Create scheduler
        scheduler = create_scheduler(max_num_seqs=10, max_num_batched_tokens=100)
        
        # 2. Create one request
        requests = create_requests(num_requests=1, num_tokens=10)
        request = requests[0]
        scheduler.add_request(request)
        
        # 3. Step scheduler to get it running
        print(f"DEBUG: Before schedule: num_computed_tokens={request.num_computed_tokens}")
        out = scheduler.schedule()
        print(f"DEBUG: After schedule: num_computed_tokens={request.num_computed_tokens}")
        assert len(out.scheduled_new_reqs) == 1
        print("Request scheduled.")
        
        # Mock model runner output
        mock_mr_output = Mock(spec=ModelRunnerOutput)
        mock_mr_output.sampled_token_ids = [[1]] # 1 token generated
        mock_mr_output.req_id_to_index = {request.request_id: 0}
        mock_mr_output.logprobs = None
        mock_mr_output.prompt_logprobs_dict = {}
        mock_mr_output.pooler_output = None
        mock_mr_output.num_nans_in_logits = None
        mock_mr_output.kv_connector_output = None
        mock_mr_output.cudagraph_stats = None
        
        scheduler.update_from_output(out, mock_mr_output)
        print(f"DEBUG: After update_from_output: num_computed_tokens={request.num_computed_tokens}")
        
        # Verify it is RUNNING
        assert request.status == RequestStatus.RUNNING
        assert len(scheduler.running) == 1
        print(f"Request status: {request.status}")
        
        # 4. Simulate moving to waiting_for_tool
        if request in scheduler.running:
            scheduler.running.remove(request)
        scheduler.waiting_for_tool.add(request)
        request.status = RequestStatus.WAITING_FOR_TOOL
        print("Moved to WAITING_FOR_TOOL.")
        
        # 5. Add interrupt
        interrupt_tokens = [100, 101, 102]
        scheduler.add_interrupt(request.request_id, interrupt_tokens)
        print("Interrupt added.")
        
        # 6. Verify status and queue
        assert request.status == RequestStatus.WAITING
        assert request in scheduler.waiting
        assert request not in scheduler.waiting_for_tool
        
        # Verify tokens appended
        # 10 prompt + 1 generated + 3 interrupt = 14
        assert request.num_tokens == 14
        print(f"Tokens check passed: {request.num_tokens} tokens.")
        
        # 7. Step scheduler again
        print(f"DEBUG: num_tokens={request.num_tokens}, num_computed_tokens={request.num_computed_tokens}")
        out_interrupt = scheduler.schedule()
        
        # Verify it schedules the interrupt tokens
        assert request.request_id in out_interrupt.num_scheduled_tokens
        scheduled_count = out_interrupt.num_scheduled_tokens[request.request_id]
        print(f"Scheduled count: {scheduled_count}")
        
        # 14 total - 10 computed = 4 new (1 generated + 3 interrupt)
        assert scheduled_count == 4
        
        assert request in scheduler.running
        print("Test PASSED!")

    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_scheduler_add_interrupt())
