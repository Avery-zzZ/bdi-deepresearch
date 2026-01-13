### 装依赖
```bash
uv sync
```

### 配置
```bash
cp .env.example .env
```
TAVILY_API_KEY 到[tavily](https://www.tavily.com/)注册获取
BOCHAI_API_KEY 没用暂时

### 跑起来

#### FastAPI

```bash
fastapi dev --entrypoint apps.http_server:app
```

接口用的SSE流式输出，fastapi自带的swagger不支持SSE实时打印，开f12看
具体数据结构说明看[这个文档](/docs/http服务器SSE协议数据结构说明.md)


#### Terminal

参考 tests/deep_research_test/apps_test/terminal_test.py

```bash
uv pip install pytest pytest-asyncio

pytest -s tests/deep_research_test/apps_test/terminal_test.py::TestTerminalApp::test_run_on_event
pytest -s tests/deep_research_test/apps_test/terminal_test.py::TestTerminalApp::test_run
```

**注意**：如果开启了`知识库调研智能体`，需在入口函数参数填写DeepResearchContext("ragflow dataset名称")

#### LangSmith

用LangSmith的缺点：看不到子智能体的行为

在.env中填写LANGSMITH_API_KEY，[官网](https://smith.langchain.com/)免费获取

```bash
langgraph dev --config langgraph.json --allow-blocking
```

**注意**：如果开启了`知识库调研智能体`，需在下方Input > Manage Assistants > kb_ragflow_dataset_name填写ragflow dataset名称