"""封装 PaddleOCR：文字识别与坐标提取。"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from utils.paths import cache_dir

# 在导入 Paddle 之前设置，减轻 Windows 上 oneDNN/PIR 组合触发的推理异常
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from paddleocr import PaddleOCR


class OCRBoxItem(TypedDict):
    """单条 OCR 识别结果的内部结构（排序与拼接用）。"""

    text: str
    confidence: float
    cx: float
    cy: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class OCREngine:
    """
    基于 PaddleOCR 的截图识别引擎。

    - 将识别结果按阅读顺序（先上后下、同行从左到右）排序后拼成一段文本；
    - 从排序结果中提取 A/B/C/D 选项标签对应文本框的中心坐标，供自动化点击。
    """

    # 选项行：字母 + 常见 OCR 分隔符（半角/全角点号、顿号、逗号、空白等）
    _OPTION_PATTERN = re.compile(r"^\s*([A-Da-d])[\.．、,，\s:：]")

    def __init__(
        self,
        *,
        lang: str = "ch",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
    ) -> None:
        """
        初始化 PaddleOCR。

        :param lang: 识别语言，默认中文（含英文混合场景常见）。
        :param use_angle_cls: 是否使用方向分类器，倾斜文字建议开启。
        :param use_gpu: 是否使用 GPU（新版 PaddleOCR 使用 ``device`` 参数）。
        """
        # PaddleOCR 3.x：设备通过 device 指定；enable_mkldnn=False 避免部分环境下 PIR+oneDNN 报错
        self._use_textline_orientation = use_angle_cls
        self._ocr = PaddleOCR(
            use_textline_orientation=use_angle_cls,
            lang=lang,
            device="gpu" if use_gpu else "cpu",
            enable_mkldnn=False,
        )

    @staticmethod
    def _normalize_ocr_result(raw: Any) -> List[List[Any]]:
        """
        将 ``predict`` / ``ocr`` 的返回值规整为单张图片的检测列表 ``[[box, (text, conf)], ...]``。

        - PaddleOCR 2.x：常见 ``[[[box, (text, score)], ...]]``；
        - PaddleOCR 3.x（PaddleX）：``[OCRResult, ...]``，内含 ``rec_texts`` / ``rec_scores`` / ``rec_polys``。
        """
        if raw is None:
            return []
        if not isinstance(raw, list) or len(raw) == 0:
            return []
        first = raw[0]
        if first is None:
            return []

        rec_texts = first.get("rec_texts") if hasattr(first, "get") else None
        rec_polys = first.get("rec_polys") if hasattr(first, "get") else None
        if rec_texts is not None and rec_polys is not None:
            rec_scores = (first.get("rec_scores") if hasattr(first, "get") else None) or []
            detections: List[List[Any]] = []
            for i, text in enumerate(rec_texts):
                conf = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                poly = rec_polys[i]
                if hasattr(poly, "tolist"):
                    poly = poly.tolist()
                if not isinstance(poly, (list, tuple)) or len(poly) < 4:
                    continue
                box = [[float(p[0]), float(p[1])] for p in poly]
                detections.append([box, (text, conf)])
            return detections

        if isinstance(first, list) and len(first) > 0:
            elem0 = first[0]
            if isinstance(elem0, list) and len(elem0) >= 2:
                if not isinstance(elem0[0], (list, tuple)):
                    return []
                return first
        return []

    @staticmethod
    def _box_to_metrics(box: List[List[float]]) -> Tuple[float, float, float, float, float, float]:
        """
        从四点框计算 min/max 与中心点。

        :return: (cx, cy, x_min, x_max, y_min, y_max)
        """
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        return cx, cy, x_min, x_max, y_min, y_max

    @staticmethod
    def _median(values: List[float]) -> float:
        """简单中位数，避免额外依赖。"""
        if not values:
            return 0.0
        s = sorted(values)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return float(s[mid])
        return (float(s[mid - 1]) + float(s[mid])) / 2.0

    def _cluster_into_rows(self, items: List[OCRBoxItem], y_threshold: float) -> List[List[OCRBoxItem]]:
        """
        将检测框按「行」聚类：同一行内 Y 接近，行与行之间按平均 Y 从上到下排列。

        聚类策略：按中心 Y 递增依次处理；每个框并入「行平均 Y 与自身差距小于阈值」且距离最近的一行；
        否则新开一行。最后再按行平均 Y 排序，行内按中心 X 排序。
        """
        if not items:
            return []
        sorted_by_y = sorted(items, key=lambda it: (it["cy"], it["cx"]))
        rows: List[List[OCRBoxItem]] = []

        for item in sorted_by_y:
            best_idx: Optional[int] = None
            best_dist = float("inf")
            for i, row in enumerate(rows):
                mean_y = sum(x["cy"] for x in row) / len(row)
                dist = abs(item["cy"] - mean_y)
                if dist < y_threshold and dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is None:
                rows.append([item])
            else:
                rows[best_idx].append(item)

        rows.sort(key=lambda r: sum(x["cy"] for x in r) / len(r))
        for row in rows:
            row.sort(key=lambda x: x["cx"])
        return rows

    def _reading_order_items(self, items: List[OCRBoxItem]) -> List[OCRBoxItem]:
        """阅读顺序：先按行（Y），同行按 X。行间距阈值由框高中位数估计。"""
        if not items:
            return []
        heights = [max(1.0, it["y_max"] - it["y_min"]) for it in items]
        median_h = self._median(heights)
        # 同一行允许的中心 Y 偏差：与常见行高成比例，并设下限避免过小图全挤一行
        y_threshold = max(10.0, median_h * 0.55)
        rows = self._cluster_into_rows(items, y_threshold)
        ordered: List[OCRBoxItem] = []
        for row in rows:
            ordered.extend(row)
        return ordered

    def _parse_detections(self, detections: List[List[Any]], min_confidence: float) -> List[OCRBoxItem]:
        """将 Paddle 原始检测列表转为带几何信息的结构化条目，并做置信度过滤。"""
        out: List[OCRBoxItem] = []
        for det in detections:
            if not isinstance(det, (list, tuple)) or len(det) < 2:
                continue
            box, rec = det[0], det[1]
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            if not isinstance(rec, (list, tuple)) or len(rec) < 2:
                continue
            text, conf = rec[0], rec[1]
            try:
                confidence = float(conf)
            except (TypeError, ValueError):
                continue
            if confidence < min_confidence:
                continue
            if not isinstance(text, str):
                text = str(text)
            cx, cy, x_min, x_max, y_min, y_max = self._box_to_metrics(box)
            out.append(
                OCRBoxItem(
                    text=text.strip(),
                    confidence=confidence,
                    cx=cx,
                    cy=cy,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                )
            )
        return out

    def _write_ocr_cache(
        self,
        source_image: Path,
        full_text: str,
        option_centers: Dict[str, Tuple[float, float]],
        min_confidence: float,
    ) -> Tuple[Path, Path]:
        """
        将 OCR 结果写入 ``cache/{图片主名}_ocr.json`` 与 ``cache/{图片主名}_ocr.txt``。
        """
        stem = source_image.stem
        cdir = cache_dir()
        json_path = cdir / f"{stem}_ocr.json"
        txt_path = cdir / f"{stem}_ocr.txt"
        payload: Dict[str, Any] = {
            "source_image": str(source_image.resolve()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "min_confidence": min_confidence,
            "full_text": full_text,
            "option_centers": {k: [float(v[0]), float(v[1])] for k, v in option_centers.items()},
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        txt_path.write_text(full_text + "\n", encoding="utf-8")
        return json_path, txt_path

    def process_image(
        self,
        image_path: str,
        *,
        min_confidence: float = 0.8,
        write_cache: bool = True,
    ) -> Tuple[str, Dict[str, Tuple[float, float]]]:
        """
        读取本地截图，返回整段文本与 A/B/C/D 选项中心坐标。

        :param image_path: 图片绝对路径（或可被 Path 解析的路径）。
        :param min_confidence: 置信度下限，低于此值的检测框丢弃。
        :param write_cache: 为 True 时，将结果写入 ``cache`` 目录下的 ``{主文件名}_ocr.json`` / ``.txt``。
        :return:
            - 第一段：所有保留文本按阅读顺序用空格拼接；
            - 第二段：键为大写字母 A-D，值为该中心点 (center_x, center_y)。
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"图片不存在或不是文件: {path.resolve()}")

        raw = self._ocr.predict(
            str(path.resolve()),
            use_textline_orientation=self._use_textline_orientation,
        )
        detections = self._normalize_ocr_result(raw)
        items = self._parse_detections(detections, min_confidence=min_confidence)
        ordered = self._reading_order_items(items)

        # 拼接：不用换行，统一空格，便于 LLM 当连续题干阅读
        full_text = " ".join(it["text"] for it in ordered if it["text"])

        option_centers: Dict[str, Tuple[float, float]] = {}
        for it in ordered:
            t = it["text"]
            m = self._OPTION_PATTERN.match(t)
            if not m:
                continue
            label = m.group(1).upper()
            if label not in ("A", "B", "C", "D"):
                continue
            # 同一字母多次出现时保留首次（通常一题一组选项）；如需覆盖可改为后写覆盖
            if label in option_centers:
                continue
            cx = (it["x_min"] + it["x_max"]) / 2.0
            cy = (it["y_min"] + it["y_max"]) / 2.0
            option_centers[label] = (cx, cy)

        if write_cache:
            self._write_ocr_cache(path, full_text, option_centers, min_confidence)

        return full_text, option_centers


if __name__ == "__main__":
    """
    独立测试：命令行传入一张本地图片路径。

    示例：
        python -m core.ocr_engine D:\\shots\\question.png
        或在 auto_exam_agent 目录下：
        python core/ocr_engine.py D:\\shots\\question.png
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if len(sys.argv) < 2:
        print("用法: python ocr_engine.py <图片路径>", file=sys.stderr)
        sys.exit(1)

    img = sys.argv[1]
    engine = OCREngine()
    text, opts = engine.process_image(img)
    print("===== 拼接文本 =====")
    print(text)
    print("===== 选项中心坐标 =====")
    for k in sorted(opts.keys()):
        print(f"  {k}: {opts[k]}")
    stem = Path(img).stem
    print("===== 缓存文件 (cache/) =====")
    print(f"  {(cache_dir() / f'{stem}_ocr.json').resolve()}")
    print(f"  {(cache_dir() / f'{stem}_ocr.txt').resolve()}")
