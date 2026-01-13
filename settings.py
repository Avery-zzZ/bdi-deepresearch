import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    model: str
    api_key: str
    base_url: str = None

    temperature: float = 0.5
    context_length: int = 131072
    max_output: int = 8192


class WebResearcherSettings(BaseModel):
    enable: bool = True
    llm: LLMSettings | None = None
    tavily_api_key: str | None = None
    bocha_api_key: str | None = None

    @model_validator(mode="after")
    def at_least_one_web_sesearch(self) -> BaseSettings:
        if not any([self.tavily_api_key, self.bocha_api_key]):
            msg = "No web search service configured. Choose 1: tavily or bocha"
            raise ValueError(msg)

        return self


class KnowledgeBaseReseacherSettings(BaseModel):
    enable: bool = True
    llm: LLMSettings | None = None
    ragflow_api_key: str
    ragflow_base_url: str = "http://localhost"


class MiddlewareSettings(BaseModel):
    au2_threshold_coeff: float = Field(0.5, gt=0, le=1)
    au2_threshold_max: int | None = None
    au2_llm: LLMSettings | None = None

    def get_threshold_length(self, context_length: int) -> int:
        return int(
            min(context_length * self.au2_threshold_coeff, self.au2_threshold_max),
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        env_ignore_empty=True,
        extra="ignore",
    )

    main_llm: LLMSettings
    max_concurrent_research: int = 3
    max_iterations: int = 5

    web_researcher: WebResearcherSettings
    kb_researcher: KnowledgeBaseReseacherSettings

    working_dir: Path = Path("_workspace")

    middlewares: MiddlewareSettings | None = None


def load_settings() -> Settings:
    env = os.getenv("ENV", "dev")

    env_file_map = {
        "dev": ".env",
        "test": ".env.test",
        "prod": ".env.prod",
    }

    return Settings(_env_file=env_file_map[env])


settings = load_settings()
