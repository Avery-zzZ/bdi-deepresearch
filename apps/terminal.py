import traceback
from pathlib import Path

from apps.assemble import assemble_deep_researcher
from events import AIMessageChunk, ReportChunk, stream_graph_to_events
from settings import Settings


class TerminalApp:

    def __init__(self, settings: Settings, workspace: Path) -> None:
        self.graph = assemble_deep_researcher(settings, workspace)

    async def run(self, query: str, context: any = None) -> None:
        print()
        try:
            async for ns, mode, data in self.graph.astream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode=["tasks", "messages"],
                subgraphs=True,
                context=context,
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
                            print(f"\n子研究结束 {args_str}")
                        elif tc["name"] == "submit_report":
                            print(f"\n-----------------研究结束-----------------\n{tc['args']['content']}")
                        else:
                            print(
                                f"\n\n🔧 工具调用{'-subagent' if ns else ''}: {tc['name']}\n   参数: {args_str[:80]+('...' if len(args_str) > 80 else '')}",
                                flush=True,
                            )

                    case "messages":
                        msg_chunk, _metadata = data
                        msg_type = getattr(msg_chunk, "type", type(msg_chunk).__name__)
                        if msg_type == "AIMessageChunk":
                            content = getattr(msg_chunk, "content", "")
                            if content:
                                print(content, end="", flush=True)

            print()
        except Exception as e:
            traceback.print_exc()
            print(f"\n\n❌ 错误: {e}")

    async def run_on_events(self, query: str, context: any = None) -> None:
        print()
        async for event in stream_graph_to_events(
            self.graph, {"messages": [{"role": "user", "content": query}]},
            context = context,
        ):
            if isinstance(event, (AIMessageChunk, ReportChunk)):
                print(event.content, end="", flush=True)
                if event.end:
                    print("\n-----------------------------------------------")
            else:
                print(type(event).__name__, event)
