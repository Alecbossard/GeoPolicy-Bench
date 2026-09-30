"""Project-local Windows adaptation; no global driver/system changes."""

import os
from pathlib import Path


def setup():
    os.environ.setdefault("MUJOCO_GL", "wgl" if os.name == "nt" else "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.name == "nt":
        import mujoco

        # robosuite's ctypes loader searches its own directory for MuJoCo.
        os.add_dll_directory(str(Path(mujoco.__file__).parent))
