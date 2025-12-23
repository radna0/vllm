
import asyncio
import json
import logging
import aiohttp
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AsyncClient")

BASE_URL = "http://localhost:8000"

async def run_client():
    """
    Simulates a client sending a request and handling AsyncLM interrupts via SSE.
    """
    logger.info("Starting AsyncLM Client...")
    
    # 1. Define Request
    prompt = "Calculate 123 + 456 using the python tool."
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    request_data = {
        "model": "gpt-oss-120b", # Assuming aliased or just checking server
        "messages": messages,
        "max_tokens": 1024,
        "stream": True,
        # Enable usage of AsyncLM features if specific parameters required?
        # The server enables it based on model/config usually.
    }

    async with aiohttp.ClientSession() as session:
        # A. Trigger Chat Completion
        logger.info(f"Sending Chat Completion Request: {messages}")
        
        # Note: In a real scenario, we need to handle the streaming response line by line.
        try:
            async with session.post(f"{BASE_URL}/v1/chat/completions", json=request_data) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Failed to start request: {response.status} - {text}")
                    return

                logger.info("Request started. Listening for SSE events...")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    
                    if line.startswith("event: async_tool_event"):
                        # Parse next data line
                        # Typically SSE sends 'event: ...\ndata: ...\n\n'
                        # We might need to handle the buffer better in production, but here we assume close coupling.
                        pass
                        # The next line should be data:
                    elif line.startswith("data: "):
                        data_str = line[len("data: "):]
                        
                        if data_str == "[DONE]":
                            logger.info("Stream finished.")
                            break
                        
                        try:
                            # We might have received async_tool_event data in previous logic, 
                            # but logic here is simple line iteration.
                            # Actually, aiohttp generic iterator yields bytes chunks or lines?
                            # content.iter_any() or similar.
                            # Standard SSE parsing:
                            pass
                        except Exception as e:
                            logger.error(f"Error parsing data: {e}")

        # Let's implement robust SSE parsing
        except Exception as e:
             logger.error(f"Connection error: {e}")

async def robust_sse_client():
    messages = [{"role": "user", "content": "Please use python to calculate 12345 * 67890."}]
    payload = {
        "model": "facebook/opt-125m", # Use safe default model name or whatever is loaded
        "messages": messages,
        "stream": True,
        "tool_choice": "auto", 
        # Add 'tools' definition if required by the model/parser logic
        # For GPT-OSS / Harmony, tools might be implicit or in system prompt?
        # Let's assume the server logic handles it or we define it.
    }
    
    logger.info("Connecting to stream...")
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BASE_URL}/v1/chat/completions", json=payload) as resp:
            if resp.status != 200:
                 logger.error(f"Error: {await resp.text()}")
                 return

            current_event_type = None
            
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if not line:
                    continue
                
                if line.startswith("event:"):
                    current_event_type = line.split(":", 1)[1].strip()
                    logger.info(f"Received Event Type: {current_event_type}")
                
                elif line.startswith("data:"):
                    data_str = line.split(":", 1)[1].strip()
                    if data_str == "[DONE]":
                        logger.info("Done.")
                        break
                    
                    data = json.loads(data_str)
                    
                    if current_event_type == "async_tool_event":
                        logger.info(f"Handling Async Tool Event: {data}")
                        await handle_async_tool_event(session, data)
                        current_event_type = None # Reset
                    else:
                        # Normal delta
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                sys.stdout.write(content)
                                sys.stdout.flush()

async def handle_async_tool_event(session, event_data):
    """
    Executes tool and posts result back.
    """
    request_id = event_data.get("request_id")
    tool_call_id = event_data.get("tool_call_id") # Maybe implicit?
    tool_name = event_data.get("tool_call_id", "python") # event_data might differ
    
    # Check structure in serving_chat.py:
    # event_data = { "type": ..., "tool_name": ..., "tool_args": ..., "tool_call_id": ..., "request_id": ... }
    
    logger.info(f"Executing tool {event_data.get('tool_name')} with args {event_data.get('tool_args')}")
    
    # Mock execution
    if "12345 * 67890" in str(event_data.get("tool_args", "")):
        result = str(12345 * 67890)
    else:
        result = "42"
        
    logger.info(f"Tool Result: {result}")
    
    # Post result
    result_payload = {
        "request_id": request_id, # This is the main ID
        "token_ids": [100, 200] # Mock token IDs representing result?
        # Wait, the protocol defines add_interrupt(request_id, token_ids).
        # We need to Tokenize the result first!
        # The client needs a tokenizer or the server api should accept text?
        # The API defined accepts 'token_ids'. 
        # Ideally the server endpoint should accept text and tokenize it, or the client does it.
        # Our `AsyncToolResultRequest` likely expects token_ids based on implementation details.
        # Let's check `api_server.py`.
    }
    
    # Wait, `AsyncToolResultRequest` in `api_server.py` calls `client.add_interrupt(request.request_id, request.token_ids)`.
    # `EngineCoreClient` expects token_ids.
    # So the client MUST tokenize.
    # For this mock, I will send dummy tokens.
    
    result_payload = {
        "request_id": request_id, 
        "token_ids": [1, 2, 3] # Mock tokens
    }
    
    logger.info(f"Sending Result to {BASE_URL}/v1/async_tool_results")
    async with session.post(f"{BASE_URL}/v1/async_tool_results", json=result_payload) as res:
        if res.status == 200:
            logger.info("Result submitted successfully.")
        else:
            logger.error(f"Failed to submit result: {await res.text()}")

if __name__ == "__main__":
    try:
        asyncio.run(robust_sse_client())
    except KeyboardInterrupt:
        pass
