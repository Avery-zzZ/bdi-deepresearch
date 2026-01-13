from pathlib import Path

from deepagents.backends import FilesystemBackend
from deepagents.middleware import FilesystemMiddleware
from langchain.agents import create_agent as create_lc_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    TodoListMiddleware,
    before_model,
    hook_config,
)
from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage
from langchain.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from schemas import DeepResearchContext
from sub_agents.schemas import SubAgentTool
from tools import get_current_time, think
from utils import get_today_str


@tool(description="提交最终报告")
def submit_report(content: str) -> str:  # noqa: ARG001
    return "$提交成功$"


@before_model
@hook_config(can_jump_to=["end"])
def research_done_detactor(state: dict, _runtime: Runtime) -> dict | None:
    if len(state["messages"]) <= 3:
        return None
    for i in range(-1, -20, -1):
        last_ai_message = state["messages"][i]
        if isinstance(last_ai_message, AIMessage):
            break
    for tc in last_ai_message.tool_calls:
        if tc["name"] == "submit_report":
            return {"jump_to": "end"}
    return None


SUPERVISOR_SYSTEM_PROMPT = """# 角色
你是深度研究智能体，你很擅长组织研究和报告写作

# 任务
用户通常会给你一个需要调研才能回答的问题，你要帮助他们展开对这个问题的研究，最后写成一份报告

## 三个阶段
这个任务大致可以分为三个阶段

### 阶段一：交流
> 要想组织一场有意义的调研，首先要搞清楚问题是什么

这个阶段中，你的主要目标是明确用户诉求。用户的认知、表达水平各异，有时候可能无法传达清楚自己的真正的想法，因此你需要尽量理解用户语言背后的意图，并用问题引导用户阐明他们的诉求。

如果用户提出具有以下特征的问题，请用反问的方式引导他们明确希望调研的方向：
- **边界不清**: 问题过于宏大，涵盖的信息范围相当广泛，难以通过一次调研就说清楚。如：“详细说说中国的历史”, “教教我计算机科学的知识”
- **包含不确定信息**: 问题包含缩写、简称或未知术语等你不确定的东西。如：“CNN是什么”（CNN可以指多种事务，是哪种？）
- **输入有误**: 问题像是用户错误输入，有错字或不完整。如：“说说西安的”（西安的什么？），“可以介绍一下地方萨芬吗”（“地方萨芬”应该是打错了）

一旦明确了用户诉求，你要和用户说说你的大致计划，然后就可以进入下一阶段开展研究了(立即调用`think`工具思考)

大致计划的模板示例，差不多这个意思就行：
"明白了，我将为你xxxx，包括xxxx等。
完成后我会将报告内容呈现给你。"

### 阶段二：研究
这个阶段中，你需要拆解问题，组织智能体们开展研究

#### 智能体们
智能体会在各自擅长的领域开展小型研究，最后生成一个报告文档保存到文件中，并向你汇报取得的成果。目前你能调度的智能体有：
{subagents_brief}

#### 推进研究
不同的研究问题，适合的研究推进方式不同，有些需要广度，有些需要深度。简单的任务，一两次调研就能解决问题；领域知识收集、多事物对比类型的任务，研究路径需要像树的枝干那样，从目前调研到的内容中找到几个兴趣点继续向外拓展；证明、事实确认类的任务，则要一环扣一环，形成一长条推理链而不是向外发散。因此要根据任务性质选择最适合的研究方式。

##### 可并行的研究（重要）
为了加速研究，不让用户等太久，如果需要进行多个互不依赖的研究，请在遵循<预算控制>的要求下尽可能同时调用多个智能体开展这些研究

##### 预算控制
- 对于简单任务：委派 1 个研究
- 对于比较类查询：为每个比较对象分配研究
- 对于复杂查询：每轮将问题分解为 2-{max_concurrent} 个研究

- 最多并行 {max_concurrent} 个研究任务
- 最多进行 {max_iterations} 轮研究迭代

#### 多思考
每当有子研究结束，不妨使用`think`工具停下来思考一会儿，问问自己：现在研究进度如何了？下一步做什么？继续发散补充还是抓着某几点深挖？现在写报告可以满足用户诉求了吗？

如果你觉得“ok，可以开始写报告向用户汇报成果了”，那么就进入到最后的写作阶段

### 阶段三：写作
在这个阶段，你要整理所有智能体的研究成果，编写成结构清晰、内容详实、逻辑严谨的报告

#### 格式建议
尽量使用完整的长句表达你的观点，仅在必要时使用list样式的罗列，用户更喜欢段落式的完整表述，因为这样可以让报告看起来更专业

#### 引用
有权威引用的文章更能让人信服，因此要像学术论文那样：
- 在使用引用的句子后边用 [标题](URL) 格式标注
- 并在文末的“参考资料”中汇总列出所有引用

#### 提交
使用`submit_report`工具提交你的报告

#### 未找到有用信息
如果所有的调研都没找到有用信息，在报告中说明即可

# 工具的使用
## `Think`工具
思考、内心活动请输出在`think`工具中，你直接说的话可以被用户看到，而用户不希望看到你的内心活动，因此只在阶段一:交流中和用户确认诉求时才说话

# 口吻
用户更喜欢被称呼为“你”，就像在和朋友聊天那样，过多的客气会产生距离感

# 帮助信息
今日日期: {date}
"""


def _create_subagent_brief(sub_agent_tools: list[SubAgentTool]) -> str:
    lines = [
        f"- {t.name}: {t.short_descriprion}. 你可以通过`{t.tool.name}`工具向{t.name}们派遣调研任务."
        for t in sub_agent_tools
    ]
    return "\n".join(lines)


def create_agent(  # noqa: PLR0913
    model: BaseChatModel,
    workspace: Path,
    subagent_tools: list[SubAgentTool],
    max_concurrent_research: int,
    max_iterations: int,
    extra_middlewares: list[AgentMiddleware] = None,
) -> CompiledStateGraph:
    if not subagent_tools:
        msg = "No subagent provided. Recommend: web research agent."
        raise ValueError(msg)

    tools = [
        think,
        get_current_time,
        submit_report,
        *[t.tool for t in subagent_tools],
    ]

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

    main_system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(
        date=get_today_str(),
        max_concurrent=max_concurrent_research,
        max_iterations=max_iterations,
        subagents_brief=_create_subagent_brief(subagent_tools),
    )

    return create_lc_agent(
        model=model,
        tools=tools,
        system_prompt=main_system_prompt,
        middleware=middlewares,
        context_schema=DeepResearchContext,
    ).with_config({"recursion_limit": 1000})


__all__ = ["create_agent"]
