"""仅测试点击选项 B（坐标来自 cache/viewport_ocr.json，须先跑过 main 或 OCR）。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from core.browser_agent import BrowserAgent
from utils.paths import cache_dir, project_root


def _cdp_port() -> int:
    p = project_root() / "config.yaml"
    if not p.is_file():
        return 9222
    cfg: Dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return int((cfg.get("browser") or {}).get("cdp_port", 9222))


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    p = cache_dir() / "viewport_ocr.json"
    if not p.is_file():
        print("未找到 cache/viewport_ocr.json，请先运行 main.py 生成 OCR 缓存。", file=sys.stderr)
        sys.exit(1)
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data.get("option_centers") or {}
    oc = {k: (float(v[0]), float(v[1])) for k, v in raw.items()}
    if "B" not in oc:
        print("缓存中无选项 B 的坐标。", file=sys.stderr)
        sys.exit(1)

    port = _cdp_port()
    agent = BrowserAgent()
    try:
        await agent.connect(port=port)
        src = data.get("source_image")
        if src and Path(str(src)).is_file():
            oc = await agent.scale_option_centers_to_viewport(oc, str(src))
        print("连接 CDP，将双击 B，坐标:", oc["B"])
        await agent.click_option_labels(["B"], oc, click_count=2)
        print("完成：已对 B 执行双击")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
