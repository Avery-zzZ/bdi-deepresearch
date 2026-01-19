import shutil
from pathlib import Path

import pytest

from apps.terminal import TerminalApp
from schemas import DeepResearchContext
from settings import settings


class TestTerminalApp:

    @pytest.fixture(autouse=True)
    def setup(self):
        p = Path("temp")
        shutil.rmtree(p)
        p.mkdir()

        self.app = TerminalApp(settings, p)

    async def test_run(self):
        await self.app.run("公司现在希望投资10W开个子公司，审批流程是怎样的，有哪些注意事项？", DeepResearchContext(kb_ragflow_dataset_name="test"))

    async def test_run_on_event(self):
        await self.app.run_on_events("简单介绍一下家常炒青菜的做法", DeepResearchContext())
