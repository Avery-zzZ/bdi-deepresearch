
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents.backends import FilesystemBackend
from deepagents.middleware import FilesystemMiddleware
from langchain.agents import AgentState
from langchain.agents import create_agent as create_lc_agent
from langchain.agents.middleware import AgentMiddleware, TodoListMiddleware
from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool, ToolRuntime, tool
from langgraph.graph.state import CompiledStateGraph
from ragflow_sdk import RAGFlow

from events import NewSubResearch
from schemas import DeepResearchContext
from sub_agents.schemas import SubAgentTool
from sub_agents.tools import research_done, research_done_detactor
from tools import think
from utils import get_today_str

from .tools import _create_tools

if TYPE_CHECKING:
    from langchain.messages import AIMessage


class KBResearcherStates(AgentState):
    kb_total_search: int
    kb_empty_search: int


_SYSTEM_PROMPT = """# 角色
你是知识库智能体，你很擅长利用知识库中的信息撰写报告

# 任务
使用提供的知识库工具收集关于指定主题的信息，形成内容丰富、逻辑严谨的研究报告，完成后向上级汇报。
知识库中是用户上传的文档，因此不一定包括对给定研究题目有用的信息。

## 可用工具

### 知识库工具
- `kb_search_chunk` - 在知识库中搜索文本块

### 通用工具
- `think` - 反思搜索结果，规划下一步
- `write_file` - 保存研究发现到文件。格式: write_file("/research_主题.md", 内容)
- `research_done` - 向上级汇报研究成果

## 工作流程

### 第一步：信息收集与分析
1. 先进行广泛搜索，了解主题概况
2. 根据初步结果进行针对性搜索
3. 每次搜索后用 think 评估收获

#### 搜索预算
- 简单查询：1-3 次搜索
- 复杂查询：最多 5 次搜索

#### 停止条件
以下条件满足任意1项即可进入下一步

##### 条件1: 找到信息足以支持编写报告
- 能够全面回答研究问题
- 已有 3 个以上相关来源
- 最近 2 次搜索返回重复信息

##### 条件2: 未找到有用信息
- 3次搜索后仍未找到有用信息

### 第二步：编写报告文档 or 报告未找到有用信息

#### 情况1：找到信息足以支持编写报告
1. 整理发现，编写报告：
   - 详细记录所有发现
   - 在末尾列出所有引用文档名称（是名称，不是id）
2. **必须使用 write_file 保存研究发现**：
   - 示例：`write_file("/research_kb_主题关键词.md", 研究发现内容)`
   - 如果发现同名文件，考虑更换一个命名保存
3. 调用`research_done`工具汇报成果

#### 情况2：未找到有用信息
1. 直接调用`research_done`工具汇报未找到有用信息，需要包括你搜索用的所有query
2. 不要创建任何文件

# 系统消息
有时工具执行结果和用户消息中会包含<system-reminder>标签包裹的内容，这是系统发出的消息，会包含一些有用信息或提示
"""


def create_agent(
    model: BaseChatModel,
    workspace: Path,
    ragflow: RAGFlow,
    extra_middlewares: list[AgentMiddleware] = None,
) -> CompiledStateGraph:
    tools = [think, _create_tools(ragflow)["kb_search_chunk"], research_done]

    filesystem_backend = FilesystemBackend(
        root_dir=workspace,
        virtual_mode=True,
    )
    middlewares = [
        research_done_detactor,
        TodoListMiddleware(),
        FilesystemMiddleware(backend=filesystem_backend),
    ]
    if extra_middlewares:
        middlewares.extend(extra_middlewares)

    return create_lc_agent(
        model=model,
        tools=tools,
        system_prompt=_SYSTEM_PROMPT.format(
            date=get_today_str(),
        ),
        middleware=middlewares,
        state_schema=KBResearcherStates,
        context_schema=DeepResearchContext,
    ).with_config({"recursion_limit": 1000})


RESEARCHER_AS_TOOL_DESCRIPTION = """知识库智能体，用于执行用户知识库的信息收集整合。

当需要搜索知识库获取用户文档信息时使用此智能体。它会：
1. 执行多次知识库搜索以全面覆盖主题
2. 整理和格式化搜索结果
3. 提供带引用的研究报告

当知识库中不包含给定研究题目的相关文档时，智能体也会给出说明

当该智能体找不到相关内容时，对于相近的内容就没必要继续派遣该智能体调研了

如果连续两次知识库调研都找不到相关内容时，可以判定该主题内容不存在于知识库中
<example>
调研一： 公司打卡制度的基本概念和常见类型 -> 没找到相关内容
调研二： 打卡时间规定和考勤规则 -> 没找到相关内容
那么就不该继续在知识库这里调研和"打卡"和"考勤"相关的内容了。
</example>

参数说明：
    title (str): 研究的标题
    instruciton (str): 给智能体看的研究内容说明prompt
    task_explaination_to_user (str): 用一句话向用户说明你在干什么。例子：正在调研xxx产品的详细信息、正在汇总xxx的相关报导
"""


def _create_agent_as_lc_tool(
    model: BaseChatModel,
    workspace: Path,
    ragflow: RAGFlow,
    extra_middlewares: list[AgentMiddleware] = None,
) -> BaseTool:
    agent = create_agent(model, workspace, ragflow, extra_middlewares)

    @tool(description=RESEARCHER_AS_TOOL_DESCRIPTION)
    async def kb_research(
        title: str,  # noqa: ARG001
        instruction: str,
        task_explaination_to_user: str,  # noqa: ARG001
        runtime: ToolRuntime,
    ) -> str:
        writer = runtime.stream_writer
        writer(
            NewSubResearch(
                id=runtime.config["metadata"]["checkpoint_ns"],
                type="知识库调研",
                title=title,
                description=task_explaination_to_user,
            ),
        )

        try:
            s = await agent.ainvoke(
                {
                    "messages": [{"role": "user", "content": instruction}],
                    "kb_empty_search": 0,
                    "kb_total_search": 0,
                },
                context=runtime.context,
            )
            m: AIMessage = s["messages"][-2]
            return m.tool_calls[0]["args"]["brief"]
        except Exception as e:
            return f"研究过程中出现错误: {e}"

    return kb_research


def create_subagent_tool(
    model: BaseChatModel,
    workspace: Path,
    ragflow: RAGFlow,
    extra_middlewares: list[AgentMiddleware] = None,
) -> SubAgentTool:
    tool = _create_agent_as_lc_tool(model, workspace, ragflow, extra_middlewares)
    return SubAgentTool(
        "知识库智能体",
        "对接用户的私域知识库，擅长收集整合其中文档形成报告，但是对于知识库中没有的内容就无能为力喽。",
        tool,
    )
