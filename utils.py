import shutil
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DEFAULT_CONTEXT_WINDOW = 131072
DEFAULT_MAX_OUTPUT_TOKENS = 8192


def get_today_str() -> str:
    """获取今天的日期字符串"""
    return datetime.now().strftime("%Y-%m-%d")  # noqa: DTZ005


@contextmanager
def temp_dir_incr(base: Path, name: str, *, delete_on_exit: bool = True) -> Generator:
    base.mkdir(parents=True, exist_ok=True)

    for i in range(10000):  # 防止死循环
        suffix = "" if i == 0 else f"_{i}"
        path = base / f"{name}{suffix}"
        try:
            path.mkdir()
            break
        except FileExistsError:
            continue
    else:
        msg = "Failed to create unique temp directory"
        raise RuntimeError(msg)

    try:
        yield path
    finally:
        if delete_on_exit:
            shutil.rmtree(path)
