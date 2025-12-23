
import pytest
from unittest.mock import Mock
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.request import RequestStatus
from vllm.v1.outputs import ModelRunnerOutput

def test_scheduler_add_interrupt():
    # 1. Create scheduler
    scheduler = create_scheduler(max_num_seqs=10, max_num_batch_tokens=100)
    
    # 2. Create one request
    requests = create_requests(num_requests=1, num_tokens=10)
    request = requests[0]
    scheduler.add_request(request)
    
    # 3. Step scheduler to get it running
    out = scheduler.schedule()
    assert len(out.scheduled_new_reqs) == 1
    
    # Mock model runner output for the first step (1 token waiting generated)
    # We simulate 1 token generated
    mock_mr_output = Mock(spec=ModelRunnerOutput)
    mock_mr_output.sampled_token_ids = [[1]]
    mock_mr_output.req_id_to_index = {request.request_id: 0}
    mock_mr_output.logprobs = None
    mock_mr_output.prompt_logprobs_dict = None
    mock_mr_output.pooler_output = None
    mock_mr_output.num_nans_in_logits = 0
    mock_mr_output.kv_connector_output = None
    mock_mr_output.cudagraph_stats = None
    
    scheduler.update_from_output(out, mock_mr_output)
    
    # Verify it is RUNNING
    assert request.status == RequestStatus.RUNNING
    assert len(scheduler.running) == 1
    
    # 4. Simulate moving to waiting_for_tool (Manual step until Phase 3)
    scheduler.running.remove(request)
    scheduler.waiting_for_tool.add(request)
    request.status = RequestStatus.WAITING_FOR_TOOL
    
    # 5. Add interrupt
    interrupt_tokens = [100, 101, 102]
    # Verify we can add interrupt
    scheduler.add_interrupt(request.request_id, interrupt_tokens)
    
    # 6. Verify status and queue
    assert request.status == RequestStatus.WAITING
    assert request in scheduler.waiting
    assert request not in scheduler.waiting_for_tool
    
    # Verify tokens appended
    # Initial: 10 prompt tokens. 
    # Generated: 1 token.
    # Interrupt: 3 tokens.
    # Total should be 14.
    assert request.num_tokens == 14
    
    # 7. Step scheduler again
    out_interrupt = scheduler.schedule()
    
    # Verify it schedules the interrupt tokens
    assert request.request_id in out_interrupt.num_scheduled_tokens
    scheduled_count = out_interrupt.num_scheduled_tokens[request.request_id]
    
    # It should schedule the 3 interrupt tokens. 
    # Note: request.num_computed_tokens should be 11 (10 prompt + 1 generated).
    # request.num_tokens is 14.
    # So 14 - 11 = 3 new tokens.
    assert scheduled_count == 3
    
    # Verify it is strictly in RUNNING (or scheduled list)
    assert request in scheduler.running

