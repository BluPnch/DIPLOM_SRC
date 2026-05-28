"""
Точка входа Streamlit (запускайте этот файл).

  python -m streamlit run app/detect_slugs.py

Код UI выполняется в этом namespace, чтобы Streamlit корректно отрисовывал виджеты.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_UI_FILE = _REPO / "slug_detection" / "app" / "detect_slugs.py"
_globals = globals()
_globals["__file__"] = str(Path(__file__).resolve())
with open(_UI_FILE, encoding="utf-8") as _f:
    exec(compile(_f.read(), str(Path(__file__).resolve()), "exec"), _globals)
