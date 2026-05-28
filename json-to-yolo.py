"""
Совместимость: python json-to-yolo.py

Реализация: scripts/data/json_to_yolo.py
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.data.json_to_yolo import main

if __name__ == "__main__":
    main()
