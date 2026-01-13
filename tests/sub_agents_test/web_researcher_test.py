from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

from apps.assemble import assemble_web_researcher
from settings import settings


async def stream_tokens(agent: CompiledStateGraph, q: str):
    try:
        async for ns, mode, data in agent.astream(
            {"messages": [{"role": "user", "content": q}]},
            stream_mode=["tasks", "messages"],
            subgraphs=True,
        ):
            match mode:
                case "tasks":
                    data: dict
                    if data["name"] != "tools":
                        continue
                    if not data.get("input"):
                        continue
                    tc: dict = data["input"]["tool_call"]
                    args_str = str(tc["args"])
                    if tc["name"] == "research_done":
                        print(f"研究结束 {args_str}")
                    else:
                        print(
                            f"\n\n🔧 工具调用{'-subagent' if ns else ''}: {tc['name']}\n   参数: {args_str[:80]+('...' if len(args_str) > 80 else '')}",
                            flush=True,
                        )

                case "messages":
                    msg_chunk, metadata = data
                    msg_type = getattr(msg_chunk, "type", type(msg_chunk).__name__)
                    if msg_type == "AIMessageChunk":
                        content = getattr(msg_chunk, "content", "")
                        if content:
                            print(content, end="", flush=True)

        print()
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "timeout" in error_str:
            print(f"\n\n⚠️ 网络连接问题: {e}")
        else:
            print(f"\n\n❌ 错误: {e}")



async def test_agent():
    p = Path("temp")
    agent = assemble_web_researcher(settings, p, as_tool=False)
    await stream_tokens(agent, "说说肉夹馍的做法")
