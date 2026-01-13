from typing import Literal

from langchain.tools import BaseTool, tool
from tavily import AsyncTavilyClient


def get_internet_search_tool(tavily_api_key: str) -> BaseTool:
    _tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

    @tool
    async def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        *,
        include_raw_content: bool = False,
    ) -> dict:
        """使用 Tavily 执行网络搜索。"""
        try:
            result = await _tavily_client.search(
                query,
                max_results=max_results,
                include_raw_content=include_raw_content,
                topic=topic,
            )

            formatted_results = []
            for i, item in enumerate(result.get("results", []), 1):
                formatted_results.append(
                    f"--- 来源 {i}: {item.get('title', 'Unknown')} ---\n"
                    f"URL: {item.get('url', 'N/A')}\n"
                    f"摘要: {item.get('content', 'N/A')}\n",
                )

            return {
                "query": query,
                "results": (
                    "\n\n".join(formatted_results)
                    if formatted_results
                    else "未找到结果"
                ),
                "result_count": len(result.get("results", [])),
            }
        except Exception as e:
            return {"error": f"Tavily search failed: {e}"}

    return internet_search
