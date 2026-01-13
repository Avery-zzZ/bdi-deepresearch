from dataclasses import dataclass

from langchain.tools import BaseTool


@dataclass
class SubAgentTool:

    name:str
    short_descriprion: str
    tool: BaseTool
