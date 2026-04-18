"""批量自动考试：从作业/考试列表扫描未做项 → 点击入口（常为「开始答题」新开标签页）→ OCR + LLM + 点选 → 提交 → 关闭考试页、刷新列表再循环。"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.browser_agent import BrowserAgent
from core.llm_brain import LLMBrain, LLMBrainError
from core.ocr_engine import OCREngine
from utils.paths import cache_dir, project_root

from playwright.async_api import Frame, Locator, Page


def _load_config() -> Dict[str, Any]:
    p = project_root() / "config.yaml"
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cdp_port() -> int:
    cfg = _load_config()
    return int((cfg.get("browser") or {}).get("cdp_port", 9222))


def _deepseek_api_key_from_config() -> Optional[str]:
    ds = _load_config().get("deepseek") or {}
    k = ds.get("api_key")
    if isinstance(k, str) and k.strip():
        return k.strip()
    return None


def _normalize_labels(answer: Any) -> List[str]:
    order: List[str] = []
    if not isinstance(answer, list):
        return order
    for raw in answer:
        L = str(raw).strip().upper()[:1]
        if L not in "ABCD":
            continue
        if L in order:
            continue
        order.append(L)
    return order


def _log(msg: str) -> None:
    print(msg, flush=True)


def _log_timing_summary(
    t_enter: float,
    t_end: float,
    llm_by_question: Dict[str, float],
    *,
    submit_attempted: bool,
    submitted: bool,
) -> None:
    total = max(0.0, t_end - t_enter)
    sum_llm = sum(llm_by_question.values())
    other = max(0.0, total - sum_llm)

    _log("[统计] ========== 用时（本脚本单次运行）==========")
    if submit_attempted and submitted:
        _log(f"[统计] 进入系统 → 末次阶段结束: {total:.2f} s")
    elif submit_attempted and not submitted:
        _log(f"[统计] 进入系统 → 末次阶段结束（提交未确认成功）: {total:.2f} s")
    else:
        _log(f"[统计] 进入系统 → 末次阶段结束（含未自动提交）: {total:.2f} s")

    if llm_by_question:
        parts = [f"{k} {v:.2f}s" for k, v in sorted(llm_by_question.items())]
        _log("[统计] LLM（各题耗时）: " + " | ".join(parts))
        _log(f"[统计] LLM 累计: {sum_llm:.2f} s")
    else:
        _log("[统计] LLM：无有效计时记录")

    _log(f"[统计] 其他约: {other:.2f} s")
    _log("[统计] ========================================")


def _write_exam_session_file(lines: List[str]) -> Optional[Path]:
    if not lines:
        return None
    cache_dir().mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = cache_dir() / f"exam_session_{stamp}"
    header = (
        "# 考试运行日志\n\n"
        f"- 保存时间: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "---\n\n"
    )
    body = "\n".join(lines)
    md_path = base.with_suffix(".md")
    txt_path = base.with_suffix(".txt")
    md_path.write_text(header + body, encoding="utf-8")
    txt_path.write_text(body, encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# 本地「考试已完成」标记（cache/exam_completion.json） + 列表 / OCR 标题
# ---------------------------------------------------------------------------
_EXAM_COMPLETION_FILE = "exam_completion.json"


def _exam_completion_path() -> Path:
    return cache_dir() / _EXAM_COMPLETION_FILE


def _normalize_exam_title(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _load_completed_title_set() -> set[str]:
    p = _exam_completion_path()
    if not p.is_file():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        arr = data.get("completed_titles") or []
        return {_normalize_exam_title(x) for x in arr if isinstance(x, str) and _normalize_exam_title(x)}
    except Exception:
        return set()


def _persist_completed_title(title: str) -> None:
    t = _normalize_exam_title(title)
    if len(t) < 2:
        return
    s = _load_completed_title_set()
    s.add(t)
    payload = {
        "completed_titles": sorted(s),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    cache_dir().mkdir(parents=True, exist_ok=True)
    _exam_completion_path().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log(
        "[标记] 已写入本地完成记录: "
        + (t[:80] + ("…" if len(t) > 80 else ""))
    )


def is_exam_title_marked_completed(title: str) -> bool:
    """是否已在本地标记完成；短标题会做子串互匹配（OCR 略有出入时仍能对上）。"""
    t = _normalize_exam_title(title)
    if len(t) < 2:
        return False
    done = _load_completed_title_set()
    if t in done:
        return True
    for c in done:
        if len(c) < 4:
            continue
        if c in t or t in c:
            return True
    return False


def parse_title_from_ocr_fulltext(ft: str) -> str:
    """从整段 OCR 文本中解析试卷/作业名称（Paddle 输出多为空格连接）。"""
    if not ft:
        return ""
    s = ft.replace("\u3000", " ").strip()
    for pat in (
        r"(?:作业|试卷|考试|测验|课堂)\s*名称\s*[:：]?\s*([^\s【]{2,100}?)(?=(\s+【)|\s+总题数|\s+第|\Z)",
        r"(?:作业|试卷|考试)[:：]\s*([^\s【]{2,100}?)(?=\s|【|总题数|第)",
        r"《\s*([^》]{2,50})\s*》",
    ):
        m = re.search(pat, s)
        if m:
            cand = _normalize_exam_title(m.group(1))
            if len(cand) >= 2:
                return cand[:120]
    parts = re.split(r"\s*(?:【\s*单选题|【\s*多选题|总题数|第\s*1\s*题)", s, maxsplit=1)
    head = (parts[0] if parts else s).strip()[:200]
    toks = [x for x in head.split() if len(x) > 1][:12]
    if toks:
        guess = _normalize_exam_title(" ".join(toks))
        if 4 <= len(guess) <= 120:
            return guess
    return ""


async def extract_exam_title_hybrid(agent: BrowserAgent, ocr: OCREngine) -> str:
    """视口截图 OCR + 浏览器 title 兜底，得到当前考试名称。"""
    await asyncio.sleep(random.uniform(0.2, 0.45))
    try:
        shot = await agent.capture_exam_area()
        full_text, _opt = ocr.process_image(shot, write_cache=True)
        t = parse_title_from_ocr_fulltext(full_text or "")
        if t:
            return t
    except Exception as exc:
        _log(f"[考试] 提取名称（OCR）失败: {exc}")
    try:
        pt = await agent.page.title()
        pt = _normalize_exam_title(pt)
        if len(pt) > 3:
            return pt[:120]
    except Exception:
        pass
    return ""


async def scrape_list_exam_items(list_page: Page) -> List[str]:
    """在作业考试列表页抓取含入口按钮的行文案，便于日志对照。"""
    try:
        raw = await list_page.evaluate(
            r"""() => {
            const out = new Set();
            const keys = ['开始答题','去作答','去考试','开始考试','继续答题','参加考试','去完成'];
            for (const el of document.querySelectorAll('a,button,[role="button"],span,.btn')) {
              const tx = (el.innerText||'').trim();
              if (!keys.some(k => tx.includes(k))) continue;
              let n = el;
              for (let d = 0; d < 14 && n; d++) {
                const row = n.closest('tr, li, [class*="item"], [class*="card"], [class*="row"], section, article, .list-group-item');
                const part = row ? (row.innerText||'') : '';
                const one = part.split(/\s+/).filter(Boolean).slice(0, 12).join(' ').slice(0, 220);
                if (one.length > 6) out.add(one);
                n = n.parentElement;
              }
            }
            return Array.from(out);
        }"""
        )
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
    except Exception:
        pass
    return []


def _print_question(
    idx: int,
    total: int,
    question_text: str,
    *,
    source: str = "OCR",
    max_chars: int = 12000,
) -> None:
    n = len(question_text)
    body = question_text[:max_chars]
    if n > max_chars:
        body += f"\n… （已截断，全文共 {n} 字）"
    _log("")
    _log(f"======== 第 {idx}/{total} 题 ｜ 题目（{source}，共 {n} 字）========")
    _log(body)
    _log("======== 题目结束 ========")


def _print_answer(idx: int, labels: List[str], result: Optional[Dict[str, Any]] = None) -> None:
    if labels:
        _log(f"[第 {idx} 题] 答案（选项）: {'、'.join(labels)}")
    else:
        _log(f"[第 {idx} 题] 答案（选项）: （无）")
    if result is not None:
        _log(f"[第 {idx} 题] 答案（JSON）: {json.dumps(result, ensure_ascii=False)}")


async def detect_total_questions(agent: BrowserAgent, *, default: int = 10) -> int:
    try:
        body = await agent.page.locator("body").inner_text()
    except Exception as exc:
        _log(f"[题量] 读取页面文本失败: {exc} — 使用默认总题数 {default}")
        return default

    patterns = [
        re.compile(r"总题数[：:\s]*(\d+)"),
        re.compile(r"试题总数[：:\s]*(\d+)"),
        re.compile(r"共\s*(\d+)\s*题"),
        re.compile(r"(\d+)\s*道题"),
        re.compile(r"\d+\s*[\/／]\s*(\d+)"),
        re.compile(r"第\s*\d+\s*题\s*[\/／]\s*共\s*(\d+)"),
    ]
    for pat in patterns:
        m = pat.search(body.replace("\n", " "))
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= 500:
                    _log(f"[题量] 从 DOM 解析到总题数: {n}")
                    return n
            except ValueError:
                continue

    _log(f"[提示] 无法在页面中解析总题数，使用默认值: {default}")
    return default


async def human_gap(lo: float = 1.0, hi: float = 2.0) -> None:
    await asyncio.sleep(random.uniform(lo, hi))


async def wait_after_next_question(agent: BrowserAgent) -> None:
    page = agent.page
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=15000)
    except Exception:
        pass
    try:
        await page.wait_for_load_state("networkidle", timeout=12000)
    except Exception:
        pass
    await asyncio.sleep(random.uniform(0.6, 1.2))


async def fetch_question_text(agent: BrowserAgent, ocr: OCREngine) -> tuple[str, str]:
    try:
        shot = await agent.capture_exam_area()
        full_text, _opt = ocr.process_image(shot, write_cache=True)
        ft = (full_text or "").strip()
        if ft:
            return ft, "OCR"
        _log("[提取题目] OCR 结果为空，尝试 DOM …")
    except Exception as exc:
        _log(f"[提取题目] OCR 异常: {exc}")

    try:
        dom = (await agent.extract_text_via_dom()).strip()
        if dom:
            return dom, "DOM回退"
    except Exception as exc:
        _log(f"[提取题目] DOM 回退异常: {exc}")
    return "", "无"


async def click_option_labels_playwright_only(agent: BrowserAgent, labels: List[str]) -> None:
    for L in labels:
        ok = False
        for attempt in range(3):
            try:
                ok = await agent.click_option_via_dom(L)
                if ok:
                    break
            except Exception as exc:
                _log(f"[点击] 选项 {L} 第 {attempt + 1} 次异常: {exc}")
            await asyncio.sleep(0.45)
        if not ok:
            _log(f"[点击] 选项 {L} 多次尝试仍失败")
        await human_gap(1.0, 2.0)


async def click_next_question(agent: BrowserAgent) -> bool:
    page = agent.page
    candidates = ("下一题", "下一页", "后一题", "Next", "next")
    for label in candidates:
        try:
            loc = page.get_by_text(label, exact=False)
            n = await loc.count()
            for i in range(n):
                cand = loc.nth(i)
                if await cand.is_visible():
                    await cand.scroll_into_view_if_needed(timeout=8000)
                    await cand.click(timeout=8000, delay=random.randint(40, 140), force=True)
                    return True
            loc0 = loc.first
            await loc0.scroll_into_view_if_needed(timeout=8000)
            await loc0.click(timeout=8000, delay=random.randint(40, 140), force=True)
            return True
        except Exception:
            continue
    try:
        btn = page.get_by_role("button", name=re.compile(r"下一", re.I)).first
        await btn.scroll_into_view_if_needed(timeout=5000)
        await btn.click(timeout=5000, delay=random.randint(40, 140), force=True)
        return True
    except Exception:
        pass
    _log("[翻页] 未找到「下一题」按钮")
    return False


async def _click_first_visible(locator: Locator, *, timeout_ms: int = 8000) -> bool:
    try:
        n = await locator.count()
        for i in range(n):
            cand = locator.nth(i)
            try:
                if await cand.is_visible():
                    await cand.scroll_into_view_if_needed(timeout=5000)
                    await cand.click(timeout=timeout_ms, delay=random.randint(40, 120))
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


async def _click_first_force(locator: Locator, *, timeout_ms: int = 5000) -> bool:
    """可见性校验易卡住自定义组件时，用 ``force=True`` 兜底。"""
    try:
        n = await locator.count()
        for i in range(n):
            cand = locator.nth(i)
            try:
                await cand.scroll_into_view_if_needed(timeout=3000)
                await cand.click(timeout=timeout_ms, delay=random.randint(30, 100), force=True)
                return True
            except Exception:
                continue
    except Exception:
        pass
    return False


async def _try_submit_entry_on_frame(frame: Frame, *, force: bool) -> bool:
    """在某一 frame（含主文档）内尝试点击「交卷 / 提交」主按钮。"""
    try:
        await frame.evaluate(
            "() => { window.scrollTo(0, document.documentElement.scrollHeight); }"
        )
        await asyncio.sleep(0.12)
    except Exception:
        pass

    submit_labels = (
        "提交作业",
        "提交试卷",
        "交卷",
        "提交答案",
        "保存并提交",
        "确认交卷",
        "我要交卷",
        "马上交卷",
        "立即提交",
        "结束考试",
        "完成考试",
        "提交",
        "去完成",
    )
    click_fn = _click_first_force if force else _click_first_visible
    for label in submit_labels:
        try:
            loc = frame.get_by_text(label, exact=False)
            if await click_fn(loc, timeout_ms=10000 if not force else 6000):
                _log(f"[提交] 已点击入口（{'force' if force else '可见'}）：{label}")
                return True
        except Exception:
            continue

    for role in ("button", "link"):
        try:
            loc = frame.get_by_role(
                role,
                name=re.compile(r"(提交|交卷|交作业|完成作业|去提交|结束\s*考试)", re.I),
            )
            if await click_fn(loc, timeout_ms=10000 if not force else 6000):
                _log(f"[提交] 已点击 {role}（{'force' if force else '可见'}，名称正则）")
                return True
        except Exception:
            continue

    try:
        loc = frame.locator(
            "button, a, [role='button'], [role='link'], input[type='button'], input[type='submit']"
        ).filter(
            has_text=re.compile(r"(提交|交卷|交作业)", re.I)
        )
        if await click_fn(loc, timeout_ms=8000 if not force else 5000):
            _log(f"[提交] 已通过 CSS+文本过滤器点击（{'force' if force else '可见'}）")
            return True
    except Exception:
        pass

    return False


async def _try_confirm_on_all_frames(page: Page, *, force: bool) -> bool:
    click_fn = _click_first_force if force else _click_first_visible
    confirm_names = ("确定", "确认", "是的", "是", "提交", "我知道了", "完成", "好的")
    for frame in page.frames:
        for name in confirm_names:
            try:
                loc = frame.get_by_role("button", name=name)
                if await click_fn(loc, timeout_ms=5000):
                    _log(f"[提交] 已点击确认：{name}")
                    return True
            except Exception:
                continue
        for rx in (r"^\s*确定\s*$", r"^\s*确认\s*$", r"提交\s*确认", r"^\s*是的\s*$"):
            try:
                loc = frame.get_by_text(re.compile(rx))
                if await click_fn(loc, timeout_ms=5000):
                    _log(f"[提交] 已点击确认文本：{rx}")
                    return True
            except Exception:
                continue
    return False


async def submit_exam_with_confirm(agent: BrowserAgent) -> bool:
    """
    在主文档与**各 iframe** 中查找「提交 / 交卷」；自动接受原生 dialog（可多步）。
    先可见点击，整轮失败后再 ``force`` 点击兜底（部分教学站点自定义按钮不报可访问名）。
    """
    page = agent.page
    loop = asyncio.get_running_loop()

    def _on_dialog(dialog: Any) -> None:
        try:
            loop.create_task(dialog.accept())
        except Exception as exc:
            _log(f"[提交] 原生 dialog 接受失败: {exc}")

    page.on("dialog", _on_dialog)
    try:
        await asyncio.sleep(random.uniform(0.25, 0.45))

        clicked = False
        for force in (False, True):
            for frame in page.frames:
                if await _try_submit_entry_on_frame(frame, force=force):
                    clicked = True
                    fu = frame.url[:80] + ("…" if len(frame.url) > 80 else "")
                    _log(f"[提交] 目标 frame：{fu!r}")
                    break
            if clicked:
                break

        if not clicked:
            _log("[提交] 未找到提交类按钮（已遍历主文档+iframe，含 force 兜底）")
            return False

        await asyncio.sleep(random.uniform(0.45, 0.9))

        if await _try_confirm_on_all_frames(page, force=False):
            return True
        if await _try_confirm_on_all_frames(page, force=True):
            return True

        _log("[提交] 已点交卷入口但未匹配到常规确认按钮；单层提交时可能已成功")
        return True
    finally:
        try:
            page.remove_listener("dialog", _on_dialog)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 列表页入口：优先「开始答题」（作业/考试列表进入新页答题）
# ---------------------------------------------------------------------------
PRIMARY_LIST_ENTRY = "开始答题"
FALLBACK_LIST_ENTRY_PATTERNS: tuple[str, ...] = (
    "去作答",
    "去考试",
    "开始考试",
    "参加考试",
    "去完成",
    "继续答题",
)


async def gather_multi_frame_text(page: Page) -> str:
    """主文档与 iframe 内 body 文本拼接，用于判断「是否在答题中 / 已提交」。"""
    chunks: List[str] = []
    try:
        chunks.append(await page.locator("body").inner_text())
    except Exception:
        pass
    for fr in page.frames:
        if fr == page.main_frame:
            continue
        try:
            chunks.append(await fr.locator("body").inner_text())
        except Exception:
            continue
    return "\n".join(chunks)


def classify_exam_ui_state(text: str) -> str:
    """
    根据可见文本粗分类：``exam``（题干区）、``completed``（已提交/结束）、``unknown``。
    优先识别题干，避免把「已…」类提示误判（保守）。
    """
    if not text or len(text.strip()) < 4:
        return "unknown"
    # 明显处于答题界面
    if re.search(r"【\s*[单多]\s*选题\s*】|【\s*判断题|总题数\s*[:：]|第\s*\d+\s*题\s*[/／]", text):
        return "exam"
    if "【多选题】" in text or "【单选题】" in text:
        return "exam"
    # 已结束 / 已提交类（无题干时）
    if re.search(
        r"答卷?\s*已\s*提交|已\s*交卷|提交\s*成功|本次?已提交|"
        r"考试已结束|考试\s*时间\s*已截止|时间已截止|"
        r"不可再次作答|禁止重复提交|没有剩余次数|本次考试已结束|"
        r"答题已结束|成绩已公布|您已提交",
        text,
    ):
        return "completed"
    if re.search(r"(^|[\r\n\s])已完成([\r\n\s：:]|$)", text):
        if "未完成" in text[:800]:
            return "unknown"
        return "completed"
    return "unknown"


async def list_entry_row_looks_completed(entry: Locator) -> bool:
    """仅在入口所在「行/卡片」内判断是否已完成；不向整页祖先爬，避免误判。"""
    try:
        return await entry.evaluate(
            r"""(el) => {
            let row = el.closest(
              'tr, li, [role="row"], [class*="list-item"], [class*="List-item"], [class*="card"], section'
            );
            if (!row) {
              let n = el;
              for (let i = 0; i < 4 && n; i++) {
                n = n.parentElement;
                if (!n) break;
                const snip = (n.innerText || '');
                if (snip.length > 20 && snip.length < 1000) {
                  row = n;
                  break;
                }
              }
            }
            if (!row) return false;
            const t = (row.innerText || '').slice(0, 2000);
            const hasPendingEntry = /开始答题|去作答|继续答题|参加考试|去完成|退回重做/.test(t);
            if (!hasPendingEntry) return false;
            if (/(^|[\s\r\n\u3000])已完成([\s\r\n\u3000：:]|$)/.test(t)) return true;
            if (/已提交|已交卷/.test(t)) return true;
            if (/(查看成绩|查看答卷)/.test(t) && !/开始答题/.test(t)) return true;
            return false;
        }"""
        )
    except Exception:
        return False


async def _filter_list_entries_not_completed_row(cands: List[Locator]) -> List[Locator]:
    out: List[Locator] = []
    for c in cands:
        try:
            if await list_entry_row_looks_completed(c):
                _log("[列表] 跳过位于「已完成/已提交」行内的入口")
                continue
        except Exception:
            pass
        out.append(c)
    return out


async def wait_exam_ready_or_completed(
    agent: BrowserAgent, *, timeout_s: float = 28.0
) -> str:
    """
    进入考试 tab 后轮询：识别到题干则 ``ready``；识别到已提交/结束则 ``completed``；否则 ``timeout``。
    """
    deadline = time.monotonic() + timeout_s
    page = agent.page

    while time.monotonic() < deadline:
        text = await gather_multi_frame_text(page)
        st = classify_exam_ui_state(text)
        if st == "exam":
            await asyncio.sleep(random.uniform(0.35, 0.65))
            return "ready"
        if st == "completed":
            return "completed"
        await asyncio.sleep(0.28)

    return "timeout"


async def scan_pending_exam_locators(page: Page) -> List[Locator]:
    """
    在作业/考试列表中**优先**查找「开始答题」，再兼容其它未做入口。
    排除位于「已完成 / 已提交 / 查看成绩」等行内的入口。
    """
    out: List[Locator] = []

    # ① 文本「开始答题」
    try:
        loc = page.get_by_text(PRIMARY_LIST_ENTRY, exact=False)
        n = await loc.count()
        for i in range(n):
            cand = loc.nth(i)
            try:
                if await cand.is_visible():
                    out.append(cand)
            except Exception:
                continue
    except Exception:
        pass

    # ② 可访问名含「开始答题」的 button / link
    if not out:
        for role in ("button", "link"):
            try:
                rloc = page.get_by_role(role, name=re.compile(r"开始答题"))
                m = await rloc.count()
                for i in range(m):
                    c = rloc.nth(i)
                    try:
                        if await c.is_visible():
                            out.append(c)
                    except Exception:
                        continue
            except Exception:
                continue

    if out:
        return await _filter_list_entries_not_completed_row(out)

    # ③ 其它常见入口（列表无「开始答题」时）
    for text in FALLBACK_LIST_ENTRY_PATTERNS:
        try:
            loc = page.get_by_text(text, exact=False)
            n = await loc.count()
            for i in range(n):
                cand = loc.nth(i)
                try:
                    if await cand.is_visible():
                        out.append(cand)
                except Exception:
                    continue
        except Exception:
            continue
    return await _filter_list_entries_not_completed_row(out)


async def wait_exam_content_ready(agent: BrowserAgent, *, timeout_s: float = 25.0) -> None:
    """进入答题页后等待题干区域出现。"""
    page = agent.page
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            body = await page.locator("body").inner_text()
        except Exception:
            await asyncio.sleep(0.25)
            continue
        if "【单选题】" in body or "【多选题】" in body or "总题数" in body:
            await asyncio.sleep(random.uniform(0.4, 0.9))
            return
        await asyncio.sleep(0.25)


async def click_list_entry_open_exam(list_page: Page, first: Locator) -> tuple[Page, bool]:
    """
    点击列表上的入口。若稍后出现**新标签页**则在该页答题，返回 ``(新页, True)``；
    否则假定**当前标签页**已进入考试，返回 ``(list_page, False)``。
    """
    ctx = list_page.context
    before_pages = list(ctx.pages)
    before_set = set(before_pages)
    await first.scroll_into_view_if_needed(timeout=10000)
    await first.click(timeout=15000, delay=random.randint(50, 150), force=True)
    deadline = time.monotonic() + 35.0
    while time.monotonic() < deadline:
        now = list(ctx.pages)
        new_ones = [p for p in now if p not in before_set]
        if new_ones:
            exam_page = new_ones[-1]
            try:
                await exam_page.wait_for_load_state("domcontentloaded", timeout=15000)
            except Exception:
                pass
            _log("[批量] 检测到新标签页，已切换为在考试页操作")
            return exam_page, True
        await asyncio.sleep(0.12)
    _log("[批量] 未检测到新标签页，在当前标签页继续答题流程")
    return list_page, False


async def finalize_exam_return_to_list_refresh(
    agent: BrowserAgent,
    list_page: Page,
    exam_page: Page,
    exam_used_new_tab: bool,
    list_fallback_url: Optional[str],
) -> None:
    """
    关闭考试标签（若与列表分属不同 ``Page``），再在**作业列表标签**上 ``reload`` 以便下一轮扫描。
    新标签场景下**不** ``bring_to_front`` 列表页：由浏览器在关闭考试页后自然切到相邻标签，避免强行抢焦点。
    """
    same_tab = exam_page is list_page
    if not same_tab:
        try:
            if not exam_page.is_closed():
                await exam_page.close()
                _log("[批量] 已关闭考试标签页（与列表页非同一 Page）")
        except Exception as exc:
            _log(f"[批量] 关闭考试标签页失败: {exc}")
    elif exam_used_new_tab:
        _log("[批量] 注意：exam_used_new_tab 为 True 但 exam_page 与 list_page 为同一对象，无法按「关标签」收尾")

    # 仅更新 Playwright 当前绑定；不把列表标签拉到最前（符合「不必跳回作业考试标签」）
    await agent.switch_to_page(list_page, focus=False)

    if same_tab:
        try:
            await list_page.go_back(wait_until="domcontentloaded", timeout=20000)
            await asyncio.sleep(random.uniform(0.4, 0.8))
            _log("[批量] 已 go_back 返回列表（单标签内进入考试）")
        except Exception as exc:
            _log(f"[批量] go_back 失败: {exc}")
            if list_fallback_url:
                try:
                    await list_page.goto(list_fallback_url, wait_until="domcontentloaded", timeout=20000)
                    _log("[批量] 已用 fallback URL 回到列表页")
                except Exception as exc2:
                    _log(f"[批量] goto 列表失败: {exc2}")

    try:
        await list_page.reload(wait_until="domcontentloaded", timeout=20000)
        _log("[批量] 已在列表标签页 reload（未强制切到该标签前台）")
    except Exception as exc:
        _log(f"[批量] 刷新列表失败: {exc}")


async def recover_list_page(
    agent: BrowserAgent,
    fallback_url: Optional[str],
    *,
    list_page: Page,
) -> None:
    """列表入口点击失败或异常时，在**列表标签页**上尽量恢复可扫描状态。"""
    await agent.switch_to_page(list_page)
    try:
        await list_page.go_back(wait_until="domcontentloaded", timeout=15000)
        _log("[恢复] 已执行 list_page.go_back()")
        return
    except Exception as exc:
        _log(f"[恢复] go_back 失败: {exc}")
    if fallback_url:
        try:
            await list_page.goto(fallback_url, wait_until="domcontentloaded", timeout=20000)
            _log("[恢复] 已导航 fallback URL")
            return
        except Exception as exc:
            _log(f"[恢复] goto 失败: {exc}")
    try:
        await list_page.reload(wait_until="domcontentloaded", timeout=20000)
        _log("[恢复] 已 reload 列表页")
    except Exception as exc:
        _log(f"[恢复] reload 失败: {exc}")


async def process_single_exam(
    agent: BrowserAgent,
    brain: LLMBrain,
    ocr: OCREngine,
    *,
    submit_after: bool,
    default_n: int,
    llm_secs: Dict[str, float],
    exam_key: str,
    exam_title: Optional[str] = None,
) -> bool:
    """
    单套试卷完整流程：题量 → 逐题 OCR/LLM/点击 → 可选提交。

    若 ``submit_after`` 为真且成功完成「提交 + 确认」，返回 ``True``；
    若未开启自动提交，答完全部题目且无致命中断则返回 ``True``（便于外层关闭考试页并刷新列表）。
    若进入时页面已为「已提交/已结束」，返回 ``True`` 并跳过答题。
    ``exam_title`` 非空且做完最后一题时，会写入本地 ``cache/exam_completion.json``。
    否则返回 ``False``。
    """
    try:
        pre = await gather_multi_frame_text(agent.page)
        if classify_exam_ui_state(pre) == "completed":
            _log("[单套] 检测到考试已结束或已提交，跳过答题与提交")
            return True
    except Exception:
        pass

    total = await detect_total_questions(agent, default=default_n)
    if total < 1:
        _log("[单套] 无法解析题量")
        return False

    for idx in range(1, total + 1):
        _log(f"======== [{exam_key}] 第 {idx}/{total} 题 ========")
        try:
            qtext, qsrc = await fetch_question_text(agent, ocr)
            if not qtext:
                _log(f"[单套] 第 {idx} 题 题目文本为空，跳过")
                if idx < total:
                    await click_next_question(agent)
                    await human_gap(1.0, 2.0)
                continue

            _print_question(idx, total, qtext, source=qsrc)

            result: Dict[str, Any] = {}
            labels: List[str] = []
            q_llm = 0.0
            from_ocr_flag = qsrc == "OCR"
            for attempt in range(2):
                t_llm0 = time.perf_counter()
                try:
                    result = brain.get_answer(qtext, from_ocr=from_ocr_flag)
                    q_llm += time.perf_counter() - t_llm0
                    labels = _normalize_labels(result.get("answer"))
                    if labels:
                        break
                except LLMBrainError as exc:
                    q_llm += time.perf_counter() - t_llm0
                    _log(f"[LLM] 第 {attempt + 1} 次失败: {exc}")
                    await asyncio.sleep(1.2)
            if q_llm > 0.0:
                llm_secs[f"{exam_key}-{idx}"] = q_llm

            _print_answer(idx, labels, result if result else None)

            if not labels:
                _log(f"[单套] 第 {idx} 题 LLM 无有效选项，跳过点击")
            else:
                await click_option_labels_playwright_only(agent, labels)

            if idx < total:
                _log("[翻页] 前往下一题 …")
                await click_next_question(agent)
                await wait_after_next_question(agent)
                await human_gap(0.5, 1.2)
            else:
                _log("[单套] 已是最后一题")
                if exam_title:
                    _persist_completed_title(exam_title)

        except Exception as exc:
            _log(f"[单套] 第 {idx} 题 异常: {exc}")
            if idx < total:
                try:
                    await click_next_question(agent)
                except Exception:
                    pass
                await human_gap(1.0, 2.0)

    if not submit_after:
        _log("[单套] 已按配置跳过提交；本套答题流程结束")
        return True

    # 不少平台最后一题作答后需再点一次「下一题」才会出现「交卷」或切换到交卷页
    _log("[单套] 最后一题后尝试唤醒交卷入口（如有「下一题」则点一次）…")
    try:
        if await click_next_question(agent):
            await asyncio.sleep(random.uniform(0.6, 1.2))
            await wait_after_next_question(agent)
    except Exception:
        pass
    await asyncio.sleep(random.uniform(0.4, 0.8))

    _log("[单套] 尝试提交 …")
    try:
        try:
            await agent.page.evaluate(
                "() => { window.scrollTo(0, document.documentElement.scrollHeight); }"
            )
            await asyncio.sleep(random.uniform(0.35, 0.7))
        except Exception:
            pass
        ok = bool(await submit_exam_with_confirm(agent))
        if ok:
            _log("[单套] 提交并完成确认")
        else:
            _log("[单套] 提交未确认成功")
        return ok
    except Exception as exc:
        _log(f"[单套] 提交过程异常: {exc}")
        return False


async def run_batch_exams(
    agent: BrowserAgent,
    brain: LLMBrain,
    ocr: OCREngine,
    *,
    submit_after: bool,
    default_n: int,
    list_fallback_url: Optional[str] = None,
) -> None:
    """
    外层任务分发：**启动时当前标签页应为「作业/考试列表」**（记为 ``list_page``，循环中保持不变）。
    循环：扫描未做入口 → 点第一个（常为**新标签**打开考试）→ 在考试页答题 → 提交（可选）→
    关闭考试页（多标签时）→ **后台** refresh 列表标签（不把列表拉到前台）→ 间隔 5～10s → 直至无入口。
    """
    t_enter = time.perf_counter()
    llm_secs: Dict[str, float] = {}
    submit_ok_last = False
    exam_round = 0

    list_page = agent.page
    _log("[批量] 预设：当前标签页为「作业/考试列表」（固定列表标签）；开始扫描未作答入口 …")
    try:
        items = await scrape_list_exam_items(list_page)
        if items:
            _log(f"[列表] 抓取到 {len(items)} 条含入口的考试/作业行（节选）")
            for i, line in enumerate(items[:15], 1):
                _log(f"  · {line[:110]}{'…' if len(line) > 110 else ''}")
    except Exception as exc:
        _log(f"[列表] 抓取考试条目失败（可忽略）: {exc}")

    while True:
        pending = await scan_pending_exam_locators(list_page)
        if not pending:
            _log("所有考试已完成！")
            break

        _log(f"[批量] 本次扫描到 {len(pending)} 个可点击入口，将点第一个")
        first = pending[0]
        try:
            exam_page, exam_used_new_tab = await click_list_entry_open_exam(list_page, first)
        except Exception as exc:
            _log(f"[批量] 点击列表入口失败: {exc} — 尝试恢复列表后重试扫描")
            await recover_list_page(agent, list_fallback_url, list_page=list_page)
            await asyncio.sleep(random.uniform(2.0, 4.0))
            continue

        await agent.switch_to_page(exam_page)

        state = await wait_exam_ready_or_completed(agent, timeout_s=28.0)
        if state == "completed":
            _log("[检查] 当前考试页为已提交或已结束，不进入答题流程")
            submit_ok_last = True
            await finalize_exam_return_to_list_refresh(
                agent,
                list_page,
                exam_page,
                exam_used_new_tab,
                list_fallback_url,
            )
            gap_skip = random.uniform(5.0, 10.0)
            _log(f"[批量] 防风控间歇 {gap_skip:.1f} s …")
            await asyncio.sleep(gap_skip)
            continue

        if state == "timeout":
            _log("[检查] 未在限时内识别题干或已结束提示，再短时等待题干加载 …")
            await wait_exam_content_ready(agent, timeout_s=14.0)

        exam_title = await extract_exam_title_hybrid(agent, ocr)
        if exam_title:
            _log(f"[考试] 当前考试名称（OCR/标题）: {exam_title}")
        else:
            _log("[考试] 未能从 OCR/标题解析考试名称；最后一题后的本地完成标记可能无法写入")

        if exam_title and is_exam_title_marked_completed(exam_title):
            _log("[标记] 该考试已在本地标记为「已完成答题」，跳过逐题作答，尝试直接提交作业 …")
            exam_round += 1
            exam_key = f"E{exam_round}"
            ok_mark = True
            if submit_after:
                try:
                    await agent.page.evaluate(
                        "() => { window.scrollTo(0, document.documentElement.scrollHeight); }"
                    )
                    await asyncio.sleep(random.uniform(0.35, 0.6))
                    ok_mark = bool(await submit_exam_with_confirm(agent))
                    if ok_mark:
                        _log("[标记] 提交作业流程已执行")
                    else:
                        _log("[标记] 未确认点击到提交作业按钮")
                except Exception as exc:
                    _log(f"[标记] 提交过程异常: {exc}")
                    ok_mark = False
            else:
                _log("[标记] 已配置不自动提交，仅关闭考试页")
            submit_ok_last = ok_mark
            await finalize_exam_return_to_list_refresh(
                agent,
                list_page,
                exam_page,
                exam_used_new_tab,
                list_fallback_url,
            )
            gap_m = random.uniform(5.0, 10.0)
            _log(f"[批量] 防风控间歇 {gap_m:.1f} s …")
            await asyncio.sleep(gap_m)
            continue

        exam_round += 1
        exam_key = f"E{exam_round}"

        try:
            ok = await process_single_exam(
                agent,
                brain,
                ocr,
                submit_after=submit_after,
                default_n=default_n,
                llm_secs=llm_secs,
                exam_key=exam_key,
                exam_title=exam_title or None,
            )
            submit_ok_last = ok
        except Exception as exc:
            _log(f"[批量] process_single_exam 未捕获外溢异常: {exc!r}")
            submit_ok_last = False
            await finalize_exam_return_to_list_refresh(
                agent,
                list_page,
                exam_page,
                exam_used_new_tab,
                list_fallback_url,
            )
            await asyncio.sleep(random.uniform(5.0, 10.0))
            continue

        await finalize_exam_return_to_list_refresh(
            agent,
            list_page,
            exam_page,
            exam_used_new_tab,
            list_fallback_url,
        )

        gap = random.uniform(5.0, 10.0)
        _log(f"[批量] 防风控间歇 {gap:.1f} s …")
        await asyncio.sleep(gap)

    t_end = time.perf_counter()
    _log_timing_summary(
        t_enter,
        t_end,
        llm_secs,
        submit_attempted=submit_after,
        submitted=submit_ok_last,
    )


async def run_test_list_scrape_only() -> None:
    """连接 CDP，对**当前标签页**执行 `scrape_list_exam_items`，打印结果后退出（不调 LLM/OCR）。"""
    agent: Optional[BrowserAgent] = None
    try:
        port = _cdp_port()
        _log(f"[测试] 连接 CDP 端口 {port} …")
        agent = BrowserAgent()
        await agent.connect(port=port)
        items = await scrape_list_exam_items(agent.page)
        _log(f"[测试] 共抓取 {len(items)} 条含入口的课程/作业行")
        for i, line in enumerate(items, 1):
            _log(f"  [{i:02d}] {line}")
        if not items:
            _log("[测试] 未抓取到条目：请将**作业考试列表**置于当前标签，并确保页面上有「开始答题」等按钮。")
    except Exception as exc:
        _log(f"[测试] 失败: {exc}")
        raise
    finally:
        if agent is not None:
            try:
                await agent.close()
            except Exception:
                pass
        _log("[测试] 已断开 Playwright / CDP（浏览器窗口保留）")


async def run_entrypoint(
    *,
    submit_after: bool,
    default_n: int,
    list_fallback_url: Optional[str],
) -> None:
    """连接 CDP → 批量循环（或你可改为在此直接调用 process_single_exam 做单套调试）。"""
    session_lines: List[str] = []
    g = globals()
    _orig_log = g["_log"]

    def _log_collect(msg: str) -> None:
        _orig_log(msg)
        session_lines.append(msg)

    g["_log"] = _log_collect
    agent: Optional[BrowserAgent] = None

    try:
        cfg = _load_config()
        port = _cdp_port()
        api_key = _deepseek_api_key_from_config()

        brain = LLMBrain(api_key=api_key)
        agent = BrowserAgent()

        _log(f"[初始化] 连接 CDP 端口 {port} …")
        await agent.connect(port=port)
        ocr_engine = OCREngine()

        await run_batch_exams(
            agent,
            brain,
            ocr_engine,
            submit_after=submit_after,
            default_n=default_n,
            list_fallback_url=list_fallback_url,
        )

    except Exception as exc:
        _log(f"[致命] {exc}")
        raise
    finally:
        if agent is not None:
            try:
                await agent.close()
            except Exception:
                pass
        _log("[结束] 已断开 Playwright / CDP（浏览器窗口保留）")
        g["_log"] = _orig_log
        saved_log = _write_exam_session_file(session_lines)
        if saved_log is not None:
            txt = saved_log.with_suffix(".txt")
            _orig_log(f"[会话] 运行日志已保存: {saved_log} 、 {txt.name}")
        else:
            _orig_log("[会话] 无日志内容写入文件（可能异常过早退出）")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="批量自动考试（CDP + OCR + LLM + Playwright）")
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="答完后不自动提交（单套内跳过提交，便于调试）",
    )
    parser.add_argument(
        "--list-url",
        type=str,
        default=None,
        metavar="URL",
        help="列表页备用 URL；单标签进入考试时 go_back 失败则 goto 以恢复列表",
    )
    parser.add_argument(
        "--test-list-scrape",
        action="store_true",
        help="仅连接浏览器并测试当前页的考试列表条目抓取，然后退出（不调 LLM）",
    )
    args = parser.parse_args()

    if args.test_list_scrape:
        asyncio.run(run_test_list_scrape_only())
        return

    cfg = _load_config()
    exam_cfg = cfg.get("exam") or {}
    yaml_submit = exam_cfg.get("submit")
    if isinstance(yaml_submit, bool):
        submit_after = yaml_submit
    else:
        submit_after = True
    if args.no_submit:
        submit_after = False

    default_n = int(exam_cfg.get("default_total_questions", 10))
    fallback = args.list_url or exam_cfg.get("list_page_url")
    if isinstance(fallback, str) and not fallback.strip():
        fallback = None

    asyncio.run(
        run_entrypoint(
            submit_after=submit_after,
            default_n=default_n,
            list_fallback_url=fallback if isinstance(fallback, str) else None,
        )
    )


if __name__ == "__main__":
    main()
