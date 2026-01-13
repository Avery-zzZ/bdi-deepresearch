from pathlib import Path

from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from agent import create_agent as create_dr_agent
from middlewares.au2_compression import AU2CompressionMiddleware
from settings import LLMSettings, Settings, WebResearcherSettings
from sub_agents.kb_researcher import (
    create_agent as create_kr,
)
from sub_agents.kb_researcher import (
    create_ragflow,
)
from sub_agents.kb_researcher import (
    create_subagent_tool as create_kr_tool,
)
from sub_agents.schemas import SubAgentTool
from sub_agents.web_researcher import (
    create_agent as create_wr,
)
from sub_agents.web_researcher import (
    create_subagent_tool as create_wr_tool,
)
from tools import get_internet_search_tool


def assemble_llm(llm_settings: LLMSettings) -> BaseChatModel:
    return init_chat_model(
        model=f"openai:{llm_settings.model}",
        api_key=llm_settings.api_key,
        base_url=llm_settings.base_url,
        temperature=llm_settings.temperature,
        max_tokens=llm_settings.max_output,
    )


def assemble_au2_middleware(settings: Settings) -> AU2CompressionMiddleware:
    au2_llm = None if not settings.middlewares else settings.middlewares.au2_llm
    llm_settings = au2_llm or settings.main_llm
    model = assemble_llm(llm_settings)
    return AU2CompressionMiddleware(
        model,
        max_context_window=llm_settings.context_length,
        max_output_tokens=llm_settings.max_output,
    )


def assemble_internet_search_tool(settings: WebResearcherSettings) -> BaseTool:
    return get_internet_search_tool(settings.tavily_api_key)


def assemble_web_researcher(
    settings: Settings,
    workspace: Path,
    *,
    as_tool: bool,
) -> SubAgentTool | CompiledStateGraph:
    llm_settings = settings.web_researcher.llm or settings.main_llm
    model = assemble_llm(llm_settings)
    func = create_wr_tool if as_tool else create_wr
    return func(
        model,
        workspace,
        assemble_internet_search_tool(settings.web_researcher),
        [assemble_au2_middleware(settings)],
    )


def assemble_kb_researcher(
    settings: Settings,
    workspace: Path,
    *,
    as_tool: bool,
) -> SubAgentTool | CompiledStateGraph:
    llm_settings = settings.kb_researcher.llm or settings.main_llm
    model = assemble_llm(llm_settings)
    rf = create_ragflow(
        settings.kb_researcher.ragflow_api_key,
        settings.kb_researcher.ragflow_base_url,
    )
    func = create_kr_tool if as_tool else create_kr
    return func(
        model,
        workspace,
        rf,
        [assemble_au2_middleware(settings)],
    )


def assemble_deep_researcher(settings: Settings, workspace: Path) -> CompiledStateGraph:
    subagent_tools = []
    if settings.web_researcher.enable:
        subagent_tools.append(
            assemble_web_researcher(settings, workspace, as_tool=True),
        )
    if settings.kb_researcher.enable:
        subagent_tools.append(assemble_kb_researcher(settings, workspace, as_tool=True))
    if not subagent_tools:
        msg = 'No researcher configured. Check ".env" file.'
        raise ValueError(msg)

    return create_dr_agent(
        assemble_llm(settings.main_llm),
        workspace,
        subagent_tools,
        max_concurrent_research=settings.max_concurrent_research,
        max_iterations=settings.max_iterations,
    )
