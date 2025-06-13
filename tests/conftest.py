import sys
from pathlib import Path

# ensure src is on PYTHONPATH
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))
