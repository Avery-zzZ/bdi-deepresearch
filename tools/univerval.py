import datetime

from langchain.tools import tool


@tool
def think(reflection: str) -> str:
    """反思工具 - 用于战略规划和结果分析。

    Args:
        reflection: 你的反思和思考内容

    Returns:
        确认反思已记录
    """
    return f"反思已记录: {reflection}"


@tool
def get_current_time() -> str:
    """获取当前时间。涉及到时间有关的操作，都需要先查询时间"""
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"
