import shutil
from pathlib import Path

p = Path("temp")
shutil.rmtree(p)
p.mkdir()
