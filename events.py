from collections.abc import AsyncGenerator

from langchain.messages import AIMessageChunk as LC_AIMessageChunk
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel


class AIMessageChunk(BaseModel):
    content: str
    end: bool


class NewSubResearch(BaseModel):
    id: str
    type: str
    title: str
    description: str


class SubResearchAction(BaseModel):
    id: str
    type: str
    description: str


class SubResearchDone(BaseModel):
    id: str


class ReportChunk(BaseModel):
    content: str
    end: bool


Event = (
    AIMessageChunk | NewSubResearch | SubResearchAction | SubResearchDone | ReportChunk
)


def _handle_main_tool_tasks(data: dict) -> Event:

    if data.get("result") and data["result"]["messages"][0].name.endswith("_research"):
        id = f"tools:{data['id']}"
        return SubResearchDone(id=id)

    return None


def _handle_subgraph_tasks(ns: tuple, data: dict, research_map: dict) -> Event:
    if data.get("input"):
        tc = data["input"]["tool_call"]
        # start a new internet search and start by a research
        if not research_map.get(ns[0], None):
            return None
        match tc["name"]:
            case "internet_search":
                query = tc["args"]["query"]
                return SubResearchAction(id=ns[0], type="网络搜索", description=f"{query}")
            case "kb_search_chunk":
                query = tc["args"]["query"]
                return SubResearchAction(id=ns[0], type="知识库搜索", description=f"{query}")

    return None


def _unescape_basic(s: str) -> str:
    return (
        s.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace("\\\\", "\\")
    )


def _create_report_chunk(content: str, *, end: bool) -> ReportChunk:
    content = _unescape_basic(content)
    return ReportChunk(content=content, end=end)


def _handle_report_chunk(chunk: LC_AIMessageChunk, states: dict) -> Event:
    for tcc in chunk.tool_call_chunks:
        if states["report_tool_id"] != chunk.id:
            continue
        if not tcc["args"]:
            return None
        if states["report_content_status"] == 1:
            if '"' not in tcc["args"]:
                states["report_last_args"] = tcc["args"]
                return None
            if ":" in states["report_last_args"] or ":" in tcc["args"]:
                states["report_content_status"] = 2
                content: str = states["report_last_args"] + tcc["args"]
                if ': "' in content:
                    content = content.split(': "', 1)[1]
                elif ':"' in content:
                    content = content.split(':"', 1)[1]
                else:
                    raise ValueError
                if (idx := content.rfind('"')) != -1 and (
                    idx == 0 or content[idx - 1] != "\\"
                ):
                    content = content[:idx]
                    states["report_content_status"] = 3
                    return _create_report_chunk(content, end=True)
                states["report_last_args"] = content
                if content:
                    return _create_report_chunk(content, end=False)
        elif states["report_content_status"] == 2:
            if '"' in tcc["args"]:
                content = states["report_last_args"] + tcc["args"]
                if (idx := content.rfind('"')) != -1 and (
                    idx == 0 or content[idx - 1] != "\\"
                ):
                    content = tcc["args"][: idx - len(states["report_last_args"])]
                    states["report_content_status"] = 3
                    return _create_report_chunk(content, end=True)
            states["report_last_args"] = tcc["args"]
            return _create_report_chunk(tcc["args"], end=False)
        states["report_last_args"] = tcc["args"]
    return None


def _handle_main_ai_msg_chunk(chunk: LC_AIMessageChunk, _meta: dict, states: dict) -> Event:
    if states["report_content_status"] in [1, 2]:
        return _handle_report_chunk(chunk, states)
    content = getattr(chunk, "content", "")
    if content:
        states["in_message"] = True
    if states["in_message"]:
        is_last = chunk.chunk_position == "last"
        if is_last:
            states["in_message"] = False
        if content or is_last:
            return AIMessageChunk(content=content, end=is_last)
    if chunk.tool_call_chunks:
        for tcc in chunk.tool_call_chunks:
            if tcc["name"] == "submit_report":
                states["report_tool_id"] = chunk.id
                states["report_content_status"] = 1
                if tcc["args"]:
                    return _handle_report_chunk(chunk, states)
    return None


def _handle_events(e: Event, research_map: dict) -> Event:
    match e:
        case NewSubResearch():
            research_map[e.id] = e.model_dump()
    return e


async def stream_graph_to_events(
    graph: CompiledStateGraph, inputs: dict, context: any = None, **kwargs: any,
) -> AsyncGenerator:
    research_map = {}
    states = {
        "report_tool_id": None,
        # 0 - not seen, 1 - tool start but content not start
        # 2 - content start, 3 - content end
        "report_content_status": 0,
        "report_last_args": "",
        "in_message": False,
    }

    async for ns, mode, data in graph.astream(
        inputs,
        stream_mode=["tasks", "messages", "custom"],
        subgraphs=True,
        context=context,
        **kwargs,
    ):
        match mode:
            case "tasks":
                data: dict
                if data["name"] != "tools":
                    continue

                # handle main agent events
                if not ns:
                    to_yield = _handle_main_tool_tasks(data)
                # handle researcher events
                else:
                    to_yield = _handle_subgraph_tasks(ns, data, research_map)

            case "messages":
                if len(ns):
                    continue
                msg_chunk, metadata = data
                if isinstance(msg_chunk, LC_AIMessageChunk):
                    to_yield = _handle_main_ai_msg_chunk(msg_chunk, metadata, states)

            case "custom":
                to_yield = _handle_events(data, research_map)

        if to_yield:
            yield (to_yield)
