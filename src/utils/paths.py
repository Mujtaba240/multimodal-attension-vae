"""Centralized path management — auto-detects Colab vs local."""

import os
from pathlib import Path


def is_colab() -> bool:
    try:
        import google.colab
        return True
    except ImportError:
        return False


def get_paths(data_dir: str = None, ckpt_dir: str = None) -> dict:
    if is_colab():
        drive_root = "/content/drive/MyDrive/Sapienza University/DL&AI_Project"
        defaults = {
            "repo_dir": "/content/repo",
            "data_dir": os.path.join(drive_root, "data"),
            "ckpt_dir": os.path.join(drive_root, "checkpoints"),
            "figures_dir": os.path.join(drive_root, "results", "figures"),
            "tables_dir": os.path.join(drive_root, "results", "tables"),
        }
    else:
        project_root = Path(__file__).resolve().parents[2]
        defaults = {
            "repo_dir": str(project_root),
            "data_dir": str(project_root / "data"),
            "ckpt_dir": str(project_root / "results" / "checkpoints"),
            "figures_dir": str(project_root / "results" / "figures"),
            "tables_dir": str(project_root / "results" / "tables"),
        }

    if data_dir:
        defaults["data_dir"] = data_dir
    if ckpt_dir:
        defaults["ckpt_dir"] = ckpt_dir

    for path in defaults.values():
        os.makedirs(path, exist_ok=True)

    return defaults
