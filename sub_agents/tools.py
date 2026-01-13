from langchain.agents.middleware import before_model, hook_config
from langchain.tools import tool
from langgraph.runtime import Runtime


@tool(parse_docstring=True)
def research_done(brief: str) -> str:  # noqa: ARG001
    """报告写完并保存后调用该Tool向上级汇报成果

    Args:
        brief (str): 阐明成果摘要，包括报告保存路径和简要总结
    """
    return "$"


@before_model
@hook_config(can_jump_to=["end"])
def research_done_detactor(state: dict, _runtime: Runtime) -> dict | None:
    last_message = state["messages"][-1]
    if last_message.content == "$":
        return {"jump_to": "end"}
    return None
