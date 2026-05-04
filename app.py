# ============================================================
# SynID — HuggingFace Space entry point
# HuggingFace Spaces requires the entry point to be app.py.
# This file loads the full UI from synid_ui.py.
# ============================================================

import os
import sys

# Ensure the current directory is on the path so relative
# exec() calls inside synid_ui.py resolve correctly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(os.path.dirname(__file__), "synid_ui.py"), "r", encoding="utf-8") as _f:
    exec(_f.read(), globals())
