# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Async tool results endpoint for native async function tool calling.

This endpoint allows external tool executors to inject tool results
into in-flight requests without restarting them.
"""

from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from vllm.v1.engine import InterruptRequest
from vllm.entrypoints.openai.protocol import (
    AsyncToolResultRequest,
    ErrorInfo,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_engine import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

async_tool_router = APIRouter()


@async_tool_router.post(
    "/v1/async_tool_results",
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
    },
)
async def inject_async_tool_results(
    request: AsyncToolResultRequest, raw_request: Request
):
    """
    Inject tool results into an in-flight async tool calling request.

    This endpoint allows external tool executors to push tool results back
    into the generation stream without restarting the request.
    """
    # Get engine client to access scheduler
    client: EngineClient = raw_request.app.state.engine_client

    try:
        # Inject interrupt
        interrupt = InterruptRequest(
            request_id=request.session_id,
            call_id=request.call_id,
            content=request.content,
            status=request.status,
        )
        await client.inject_interrupt(interrupt)

        return JSONResponse(
            content={
                "status": "queued",
                "session_id": request.session_id,
                "call_id": request.call_id,
            }
        )
    except Exception as e:
        logger.exception("Error injecting async tool result: %s", e)
        return JSONResponse(
            content=ErrorResponse(
                error=ErrorInfo(
                    message=f"Failed to inject tool result: {str(e)}",
                    type="internal_error",
                    code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            ).model_dump(),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
