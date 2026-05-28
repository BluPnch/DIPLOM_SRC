"""
Совместимость со старыми скриптами.

Предпочтительно: from slug_detection.config import dataset
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from slug_detection.config.dataset import *  # noqa: F403
