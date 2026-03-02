from pathlib import Path
from typing import TYPE_CHECKING

from deepagents.backends import FilesystemBackend
from deepagents.middleware import FilesystemMiddleware
from langchain.agents import create_agent as create_lc_agent
from langchain.agents.middleware import AgentMiddleware, TodoListMiddleware
from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool, ToolRuntime, tool
from langgraph.graph.state import CompiledStateGraph
from openai import models

from events import NewSubResearch
from schemas import DeepResearchContext
from tools import think
from utils import get_today_str

from .schemas import SubAgentTool
from .tools import research_done, research_done_detactor

if TYPE_CHECKING:
    from langchain.messages import AIMessage

_RESEARCHER_SYSTEM_PROMPT = """你是一个专业的研究助手，负责收集整合信息最后编写研究报告。

## 任务
使用提供的搜索工具收集关于指定主题的信息，形成内容丰富、逻辑严谨的研究报告，完成后向上级汇报。

## 可用工具
1. `{search_tool_name}` - 网络搜索，获取最新信息
2. `think` - 反思搜索结果，规划下一步
3. `write_file_with_unique_name` - 保存研究发现到文件，自动处理文件名冲突。格式: write_file_with_unique_name("/research_主题.md", 内容)
4. `research_done` - 向上级汇报研究成果

## 文件系统
- 工作目录是根目录 `/`
- 完成研究后，将研究发现保存到文件
- 文件名格式：`/research_主题关键词.md`

## 工作流程

### 第一步：信息收集与分析
1. 先进行广泛网络搜索，了解主题概况
2. 根据初步结果进行针对性搜索
3. 每次搜索后用 think 评估收获

#### 搜索预算
- 简单查询：1-3 次搜索
- 复杂查询：最多 5 次搜索

#### 停止条件
- 能够全面回答研究问题
- 已有 3 个以上相关来源
- 最近 2 次搜索返回重复信息

### 第二步：编写报告文档
1. 整理发现，编写报告：
   - 详细记录所有发现
   - 使用 [标题](URL) 格式引用来源
   - 在末尾列出所有来源
2. **必须使用 write_file_with_unique_name 保存研究发现**：
   - 格式：`write_file_with_unique_name("/research_主题关键词.md", 研究发现内容)`
   - 例如：`write_file_with_unique_name("/research_deepseek.md", "# Deepseek研究发现\n...")`
   - 如果发现同名文件，工具会自动在文件名后添加数字后缀，例如：`/research_主题关键词_1.md`、`/research_主题关键词_2.md`等

#### 报告风格
尽量使用完整的长句表达，仅在必要时使用list样式的罗列，用户更喜欢段落式的完整表述，因为这样可以让报告看起来更专业

### 第三步：向上级汇报成果（必须）
调用`research_done`工具汇报成果

## 系统消息
有时工具执行结果和用户消息中会包含<system-reminder>标签包裹的内容，这是系统发出的消息，会包含一些有用信息或提示

## 帮助信息
今天日期: {date}
"""


def get_unique_filename(filepath: str) -> str:
    """获取唯一的文件名，如果文件已存在则添加数字后缀
    
    Args:
        filepath: 原始文件路径
        
    Returns:
        唯一的文件路径
    """
    import os
    if not os.path.exists(filepath):
        return filepath
    
    base, ext = os.path.splitext(filepath)
    counter = 1
    while True:
        new_filepath = f"{base}_{counter}{ext}"
        if not os.path.exists(new_filepath):
            return new_filepath
        counter += 1


def create_agent(
    model: BaseChatModel,
    workspace: Path,
    search_tool: BaseTool,
    extra_middlewares: list[AgentMiddleware] = None,
) -> CompiledStateGraph:
    from langchain.tools import tool
    
    @tool
    def write_file_with_unique_name(filepath: str, content: str) -> str:
        """保存研究发现到文件，自动处理文件名冲突
        
        Args:
            filepath: 文件路径
            content: 文件内容
            
        Returns:
            保存结果
        """
        import os
        # 构建完整的文件路径
        full_path = os.path.join(str(workspace), filepath.lstrip('/'))
        # 获取唯一的文件名
        unique_path = get_unique_filename(full_path)
        # 转换回相对路径格式
        relative_path = '/' + os.path.relpath(unique_path, str(workspace))
        # 使用FilesystemMiddleware的write_file功能
        from deepagents.middleware.filesystem import FilesystemMiddleware
        backend = FilesystemBackend(
            root_dir=workspace,
            virtual_mode=True,
        )
        # 直接使用backend写入文件
        backend.write(relative_path, content)
        return f"文件已保存到: {relative_path}"

    tools = [think, search_tool, write_file_with_unique_name, research_done]

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
        system_prompt=_RESEARCHER_SYSTEM_PROMPT.format(
            date=get_today_str(),
            search_tool_name="internet_search",
        ),
        middleware=middlewares,
        context_schema=DeepResearchContext,
    ).with_config({"recursion_limit": 1000})


RESEARCHER_AS_TOOL_DESCRIPTION = """网络调研智能体，用于执行网络信息收集整合。

当需要搜索互联网获取信息时使用此智能体。它会：
1. 执行多次网络搜索以全面覆盖主题
2. 整理和格式化搜索结果
3. 提供带引用的研究报告

参数说明：
    title (str): 研究的标题
    instruciton (str): 给智能体看的研究内容说明prompt
    task_explaination_to_user (str): 用一句话向用户说明你在干什么。例子：正在调研xxx产品的详细信息、正在汇总xxx的相关报导
"""


def _create_agent_as_lc_tool(
    model: BaseChatModel,
    workspace: Path,
    search_tool: BaseTool,
    extra_middlewares: list[AgentMiddleware] = None,
) -> BaseTool:
    agent = create_agent(model, workspace, search_tool, extra_middlewares)

    @tool(description=RESEARCHER_AS_TOOL_DESCRIPTION)
    async def internet_research(
        title: str,  # noqa: ARG001
        instruction: str,
        task_explaination_to_user: str,  # noqa: ARG001
        runtime: ToolRuntime,
    ) -> str:
        writer = runtime.stream_writer
        writer(
            NewSubResearch(
                id=runtime.config["metadata"]["checkpoint_ns"],
                type="网络调研",
                title=title,
                description=task_explaination_to_user,
            ),
        )

        try:
            s = await agent.ainvoke(
                {"messages": [{"role": "user", "content": instruction}]},
            )
            # 遍历所有消息，找到最后一次调用research_done工具的消息
            research_done_message = None
            for msg in s["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call["name"] == "research_done":
                            research_done_message = msg
                            break
            
            if research_done_message:
                # 找到最后一次调用research_done的工具调用
                for tool_call in research_done_message.tool_calls:
                    if tool_call["name"] == "research_done":
                        return tool_call["args"].get("brief", "研究完成，但未提供摘要")
            
            return "研究完成，但未找到research_done工具调用"
        except Exception as e:
            return f"研究过程中出现错误: {e}"

    return internet_research


def create_subagent_tool(
    model: BaseChatModel,
    workspace: Path,
    search_tool: BaseTool,
    extra_middlewares: list[AgentMiddleware] = None,
) -> SubAgentTool:
    tool = _create_agent_as_lc_tool(model, workspace, search_tool, extra_middlewares)
    return SubAgentTool(
        "网络调研智能体",
        "网上冲浪高手，擅长使用搜索引擎收集整合网页资料形成报告。",
        tool,
    )
