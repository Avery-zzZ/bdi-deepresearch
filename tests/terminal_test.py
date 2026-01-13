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
        await self.app.run("炒青菜怎么做")

    async def test_run_on_event(self):
        await self.app.run_on_events("简单介绍一下家常炒青菜的做法", DeepResearchContext())
