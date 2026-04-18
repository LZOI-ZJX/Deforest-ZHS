"""项目内路径：缓存目录、默认截图路径等。"""

from __future__ import annotations

from pathlib import Path

# auto_exam_agent/ 根目录（本文件位于 auto_exam_agent/utils/）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SCREENSHOT_BASENAME = "viewport.png"


def project_root() -> Path:
    return _PROJECT_ROOT


def cache_dir() -> Path:
    """截图与 OCR 侧车缓存文件所在目录（``auto_exam_agent/cache``）。"""
    d = _PROJECT_ROOT / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_screenshot_path() -> Path:
    """默认视口截图路径：``cache/viewport.png``。"""
    return cache_dir() / DEFAULT_SCREENSHOT_BASENAME
