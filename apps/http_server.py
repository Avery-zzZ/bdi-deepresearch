from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import MessageLikeRepresentation, convert_to_messages
from pydantic import BaseModel, Field

from apps.assemble import assemble_deep_researcher
from events import (
    AIMessageChunk,
    Event,
    NewSubResearch,
    ReportChunk,
    SubResearchAction,
    SubResearchDone,
    stream_graph_to_events,
)
from schemas import DeepResearchContext
from settings import settings
from utils import temp_dir_incr


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    workspace = settings.working_dir
    workspace.mkdir(exist_ok=True)
    app.state.workspace = workspace

    yield


app = FastAPI(lifespan=lifespan)


class DRRequest(BaseModel):
    messages: list[MessageLikeRepresentation] = Field(
        examples=[[{"role": "user/ai", "content": "xxxxx"}]],
    )
    dataset_name: str = None


def get_workspace(request: Request) -> Path:
    return request.app.state.workspace


def _to_sse_chunk_str(event: str, data: BaseModel) -> str:
    return f"event: {event}\ndata: {data.model_dump_json()}\n\n"


async def sse_stream(stream: AsyncGenerator[Event, any]) -> AsyncGenerator[str, any]:
    async for e in stream:
        match e:
            case AIMessageChunk():
                yield _to_sse_chunk_str("ai_message_chunk", e)
            case NewSubResearch():
                yield _to_sse_chunk_str("new_sub_research", e)
            case SubResearchAction():
                yield _to_sse_chunk_str("sub_research_action", e)
            case SubResearchDone():
                yield _to_sse_chunk_str("sub_research_done", e)
            case ReportChunk():
                yield _to_sse_chunk_str("report_chunk", e)


@app.post("/deepresearch")
async def deep_research(
    r: DRRequest, workspace: Annotated[Path, Depends(get_workspace)],
) -> StreamingResponse:
    try:
        messages = convert_to_messages(r.messages)
    except ValueError as err:
        raise HTTPException from err(
            400,
            'Wrong message format. It should be like {"role": "user/ai", "content": "xxxxx"}.',
        )
    if not messages:
        raise HTTPException(400, "No message found in request.")

    with temp_dir_incr(workspace, "s") as sub_workspace:
        agent = assemble_deep_researcher(settings, sub_workspace)
        stream = stream_graph_to_events(agent, {"messages": messages}, context = DeepResearchContext(r.dataset_name))
        return StreamingResponse(sse_stream(stream), media_type="text/event-stream")
