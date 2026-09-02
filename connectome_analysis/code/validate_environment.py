# Validates the Python environment by importing required packages and reporting their installed versions.
from __future__ import annotations

import importlib
import platform


MODULES = (
    "caveclient",
    "cloudvolume",
    "fafbseg",
    "gspread",
    "k3d",
    "matplotlib",
    "meshparty",
    "navis",
    "networkx",
    "neuprint",
    "nglui",
    "numpy",
    "pandas",
    "pcg_skel",
    "plotly",
    "pymaid",
    "scipy",
    "seaborn",
    "skeletor",
    "sklearn",
    "trimesh",
)


def main() -> None:
    print(f"Python: {platform.python_version()}")
    for name in MODULES:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "installed")
        print(f"{name}: {version}")
    print("Environment validation: PASS")


if __name__ == "__main__":
    main()
