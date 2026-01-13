import shutil
from functools import lru_cache
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from settings import settings

from .assemble import (
    assemble_deep_researcher,
    assemble_kb_researcher,
    assemble_web_researcher,
)


@lru_cache
def get_ws() -> Path:
    p = Path("temp")
    shutil.rmtree(p)
    p.mkdir()
    return p


def make_web_researcher(config: RunnableConfig) -> CompiledStateGraph | None:  # noqa: ARG001
    if settings.web_researcher.enable:
        return assemble_web_researcher(settings, get_ws(), as_tool=False)
    return None


def make_kb_researcher(config: RunnableConfig) -> CompiledStateGraph | None:  # noqa: ARG001
    if settings.kb_researcher.enable:
        return assemble_kb_researcher(settings, get_ws(), as_tool=False)
    return None


deep_researcher = assemble_deep_researcher(settings, get_ws())
