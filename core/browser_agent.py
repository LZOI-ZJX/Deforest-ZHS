"""封装 Playwright：CDP 连接已有 Chrome、视口截图、仿真点击。"""

from __future__ import annotations

import asyncio
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from PIL import Image
from playwright.async_api import Browser, BrowserContext, Locator, Page, Playwright, async_playwright

from utils.paths import default_screenshot_path


class BrowserAgent:
    """
    通过 Chrome DevTools Protocol（CDP）接管本机已启动的调试模式 Chrome。

    设计约定
    --------
    - **不**负责启动浏览器进程。请先用远程调试参数启动 Chrome，例如：
      ``chrome.exe --remote-debugging-port=9222``（路径与参数因系统而异）。
    - ``close()`` 仅断开 Playwright 与 CDP 的会话，**不会**退出用户真实的浏览器窗口与标签页。
    - **首选**：用 DOM 读文本（:meth:`extract_text_via_dom`）与按选项字母点击（:meth:`click_option_via_dom`），
      避免 DPR / 多显示器 / 滚动带来的坐标问题。
    - **备选**：截图 + OCR + :meth:`human_click_by_coordinate` / :meth:`click_option_labels`（与 ``scale=css`` 视口一致）。
    """

    def __init__(
        self,
        *,
        hide_selectors_before_capture: Optional[Iterable[str]] = None,
    ) -> None:
        """
        :param hide_selectors_before_capture:
            截图前**临时**通过 JavaScript 将匹配到的首个元素 ``display: none`` 的选择器列表
            （例如 ``\"header\"``、``\"nav\"``）。截图结束后会恢复为默认显示方式。
            默认不隐藏任何元素；若无需遮挡处理，保持 ``None`` 即可，作为扩展位预留。

            等价于在页面内执行类似：
            ``document.querySelector('header').style.display = 'none'``（本类在
            :meth:`capture_exam_area` 内批量、可恢复地执行）。
        """
        self.hide_selectors_before_capture: List[str] = (
            list(hide_selectors_before_capture) if hide_selectors_before_capture else []
        )

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    @property
    def page(self) -> Page:
        """当前绑定的活动页面对象；未连接时访问会抛出 ``RuntimeError``。"""
        if self._page is None:
            raise RuntimeError("尚未连接浏览器，请先调用 await connect()")
        return self._page

    @property
    def context(self) -> BrowserContext:
        """当前浏览器上下文；未连接时访问会抛出 ``RuntimeError``。"""
        if self._context is None:
            raise RuntimeError("尚未连接浏览器，请先调用 await connect()")
        return self._context

    def _ensure_page(self) -> Page:
        return self.page

    async def connect(self, port: int = 9222) -> None:
        """
        连接到本机指定调试端口的 Chrome（纯异步，基于 ``playwright.async_api``）。

        使用 ``chromium.connect_over_cdp(f\"http://127.0.0.1:{port}\")`` 附着到已有 Chrome；
        随后取 ``browser.contexts[0]`` 作为当前 ``context``，并取 ``context.pages[0]`` 作为活动 ``page``
        （多标签、多窗口场景下若需「当前焦点页」，可后续改为遍历 ``pages`` 或结合 CDP 目标选择）。

        :param port: Chrome 远程调试端口，默认 9222。
        :raises RuntimeError: 未找到任何上下文或默认上下文中没有页面时。
        :raises Exception: 网络或 CDP 握手失败时由 Playwright 抛出，调用方需捕获处理。
        """
        # 避免重复 connect 导致 Playwright / Browser 实例泄漏
        if self._playwright is not None:
            await self.close()

        self._playwright = await async_playwright().start()
        # 使用 127.0.0.1：Windows 上 localhost 常解析为 ::1，而 Chrome 可能仅监听 IPv4，导致 ECONNREFUSED
        endpoint = f"http://127.0.0.1:{port}"
        self._browser = await self._playwright.chromium.connect_over_cdp(endpoint)

        contexts = self._browser.contexts
        if not contexts:
            await self.close()
            raise RuntimeError(
                f"已通过 CDP 连接 {endpoint}，但未发现任何 BrowserContext。"
                "请确认 Chrome 已用远程调试启动且至少打开一个常规窗口/标签页。"
            )

        self._context = contexts[0]
        pages = self._context.pages
        if not pages:
            await self.close()
            raise RuntimeError(
                "默认上下文中没有任何页面（pages 为空）。请先打开一个标签页后再连接。"
            )

        self._page = pages[0]
        try:
            await self._page.bring_to_front()
        except Exception:
            pass

    async def switch_to_page(self, page: Page, *, focus: bool = True) -> None:
        """绑定当前脚本使用的 ``Page``；若 ``focus=True`` 则将该浏览器标签置于最前。"""
        self._page = page
        if not focus:
            return
        try:
            await page.bring_to_front()
        except Exception:
            pass

    async def _apply_hide_selectors(self, hide: bool) -> None:
        """
        在截图前后临时隐藏/恢复配置的选择器对应元素。

        - ``hide=True``：对列表中每个选择器执行 ``querySelector``，存在则 ``style.display = 'none'``；
        - ``hide=False``：存在则 ``style.display = ''``，交还浏览器默认层叠显示。
        """
        page = self._ensure_page()
        selectors = self.hide_selectors_before_capture
        if not selectors:
            return

        if hide:
            await page.evaluate(
                """(selectorList) => {
                    for (const sel of selectorList) {
                        const el = document.querySelector(sel);
                        if (el) el.style.display = 'none';
                    }
                }""",
                selectors,
            )
        else:
            await page.evaluate(
                """(selectorList) => {
                    for (const sel of selectorList) {
                        const el = document.querySelector(sel);
                        if (el) el.style.display = '';
                    }
                }""",
                selectors,
            )

    async def capture_exam_area(self, save_path: Optional[str] = None) -> str:
        """
        截取**当前视口**可见区域（``full_page=False``），供 OCR 使用；不对整页做长图。

        流程说明
        --------
        1. （可选）根据 ``hide_selectors_before_capture`` 注入脚本，短暂隐藏可能遮挡题干的节点；
        2. 调用 ``page.screenshot``，仅视口；
        3. ``finally`` 中无论截图是否成功，都恢复被隐藏元素的显示，避免影响用户继续操作页面。

        扩展：若仅需一次性脚本而不维护选择器列表，可自行在子类中重写本方法，或先 ``page.evaluate``
        执行例如 ``document.querySelector('header') && (header.style.display='none')``（注意事后恢复）。

        :param save_path: 截图保存路径；默认 ``cache/viewport.png``（见 ``utils.paths``）。
        :return: 保存后的绝对路径字符串（规范化后）。
        """
        page = self._ensure_page()
        out = Path(save_path) if save_path is not None else default_screenshot_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(out.resolve())

        await self._apply_hide_selectors(True)
        try:
            # full_page 默认为 False：与「仅视口」需求一致
            # scale='css'：截图 1 像素对应 1 个 CSS 像素，与 page.mouse / 视口坐标一致。
            # 默认 scale='device' 时高 DPI 下图约为 2× 宽高，OCR 坐标会与鼠标错位。
            await page.screenshot(path=resolved, full_page=False, scale="css")
        finally:
            await self._apply_hide_selectors(False)

        return resolved

    async def scale_option_centers_to_viewport(
        self,
        option_centers: Dict[str, Tuple[float, float]],
        screenshot_path: Union[str, Path],
    ) -> Dict[str, Tuple[float, float]]:
        """
        将基于截图像素得到的 OCR 坐标换算为当前页 **CSS 视口** 坐标。

        当 PNG 宽高与 ``page.viewport_size`` 不一致时（极少数环境或旧缓存），按比例缩放；
        一致时原样返回。
        """
        page = self._ensure_page()
        vp = page.viewport_size
        if vp is None:
            return option_centers
        path = Path(screenshot_path)
        if not path.is_file():
            return option_centers
        with Image.open(path) as im:
            iw, ih = im.size
        vw, vh = float(vp["width"]), float(vp["height"])
        if iw <= 0 or ih <= 0 or vw <= 0 or vh <= 0:
            return option_centers
        if abs(iw - vw) < 0.5 and abs(ih - vh) < 0.5:
            return option_centers
        sx = vw / iw
        sy = vh / ih
        return {k: (float(v[0]) * sx, float(v[1]) * sy) for k, v in option_centers.items()}

    async def extract_text_via_dom(self) -> str:
        """
        从当前页面读取可见纯文本，供大模型理解题干与选项（不依赖截图 / OCR）。

        使用 ``body`` 的 ``inner_text()``，包含子树内可见文本；若需进一步去噪可在调用方处理。
        """
        page = self._ensure_page()
        raw = await page.locator("body").inner_text()
        return (raw or "").strip()

    async def _click_first_working_locator(
        self,
        root: Locator,
        *,
        timeout_ms: float,
    ) -> bool:
        """
        对定位器内所有匹配项：优先点击 **当前可见** 的节点（翻页后旧题节点常仍以 display:none 留在 DOM，
        若用 ``.first`` 会点到隐藏节点导致失败）。
        """
        delay_ms = random.randint(80, 220)
        try:
            n = await root.count()
        except Exception:
            n = 0
        if n == 0:
            return False

        for i in range(n):
            cand = root.nth(i)
            try:
                if not await cand.is_visible():
                    continue
                await cand.scroll_into_view_if_needed(timeout=timeout_ms)
                await cand.click(timeout=timeout_ms, delay=delay_ms, force=True)
                return True
            except Exception:
                continue

        for i in range(n):
            cand = root.nth(i)
            try:
                await cand.scroll_into_view_if_needed(timeout=timeout_ms)
                await cand.click(timeout=timeout_ms, delay=delay_ms, force=True)
                return True
            except Exception:
                continue
        return False

    async def click_option_via_dom(self, option_letter: str, *, timeout_ms: float = 12000.0) -> bool:
        """
        按选项字母（如 ``A``）用 DOM 定位并点击，不依赖 OCR。

        - 文本匹配：行首 ``字母 + 分隔符``（``.``、顿号、冒号、空白等）。
        - **多匹配时**遍历并优先点当前 **可见** 项，避免点到上一题残留节点。
        - 备选：常见 ``span.ABCase`` / 含 ``ABCase`` 的 class。
        - ``force=True`` 应对站点对「可见性」判定过严的情况。
        """
        page = self._ensure_page()
        L = str(option_letter).strip().upper()[:1]
        if L not in "ABCD":
            return False
        pattern = re.compile(rf"^{re.escape(L)}[.．、,，:：\s]")
        short_pat = re.compile(rf"^{re.escape(L)}\.")

        strategies: List[Locator] = [
            page.get_by_text(pattern),
            page.locator("span.ABCase, span[class*='ABCase']").filter(has_text=short_pat),
            page.locator("label, li, div, span").filter(has_text=pattern),
        ]

        for strat in strategies:
            try:
                if await self._click_first_working_locator(strat, timeout_ms=timeout_ms):
                    return True
            except Exception as e:
                print(
                    f"[BrowserAgent] click_option_via_dom 策略异常（{L}）: {e}",
                    flush=True,
                )
                continue

        print(
            f"[BrowserAgent] click_option_via_dom 失败（选项 {L}）：所有定位策略均未命中可点击节点",
            flush=True,
        )
        return False

    async def human_click_by_coordinate(
        self,
        x: float,
        y: float,
        *,
        click_count: int = 1,
        jitter_max: float = 0.0,
        use_dom_click: bool = True,
        use_playwright_mouse: bool = True,
        after_dom_use_mouse: bool = True,
    ) -> None:
        """
        **（基于 OCR 视口坐标的备用方法）** 在**页面视口 CSS 坐标**下点击 ``(x, y)``（与 OCR / ``scale=css`` 截图一致）。

        - **红点**：挂在 ``documentElement`` 下全屏遮罩内，避免父级 ``transform`` 导致 ``position:fixed`` 错位。
        - **抖动**：默认 ``jitter_max=0``（避免 ±5px 点偏选项）；需要拟人时可设 ``2～5``。
        - **点击**：在**顶层** ``elementFromPoint`` 往往点到 ``iframe`` 外壳，合成事件进不了子文档。此处递归进入**同源**
          iframe 再取点并派发；遇**跨域** iframe 则无法注入，必须依赖 Playwright 物理鼠标。
        - 默认每一轮在 DOM 派发后**再发一次物理鼠标**（考试页常为 iframe + 跨域或 React，双路径更稳）。
          若出现重复切换选项，可设 ``use_dom_click=False`` 或 ``after_dom_use_mouse=False``。

        :param click_count: ``1`` 单击，``2`` 双击。
        :param jitter_max: 随机抖动最大像素，``0`` 表示不抖（推荐 OCR 坐标）。
        :param use_dom_click: 是否在解析到的目标元素上派发 DOM 事件。
        :param use_playwright_mouse: 是否使用 ``page.mouse`` 物理点击。
        :param after_dom_use_mouse: 为真时，在 DOM 派发后**仍然**执行物理鼠标（推荐 True）。
        """
        page = self._ensure_page()
        if click_count < 1:
            click_count = 1

        jm = max(0.0, float(jitter_max))
        jitter_x = random.uniform(-jm, jm) if jm > 0 else 0.0
        jitter_y = random.uniform(-jm, jm) if jm > 0 else 0.0
        target_x = float(x) + jitter_x
        target_y = float(y) + jitter_y

        vp = page.viewport_size
        if vp is not None:
            w, h = float(vp["width"]), float(vp["height"])
            if w > 0.0 and h > 0.0:
                target_x = max(0.0, min(w - 1.0, target_x))
                target_y = max(0.0, min(h - 1.0, target_y))

        delay_s = random.uniform(0.3, 0.8)
        await asyncio.sleep(delay_s)

        # 调试用：全屏遮罩 + 红点，避免挂 body 受 transform 影响；控制台打印视口信息
        await page.evaluate(
            """([px, py]) => {
                const oid = '__autoexam_viewport_overlay';
                document.getElementById(oid)?.remove();
                const overlay = document.createElement('div');
                overlay.id = oid;
                overlay.style.cssText = [
                    'position:fixed','left:0','top:0','right:0','bottom:0',
                    'width:100%','height:100%','pointer-events:none','z-index:2147483647',
                    'margin:0','isolation:isolate',
                ].join(';');
                const dot = document.createElement('div');
                dot.style.cssText = [
                    'position:absolute',
                    'left:' + px + 'px',
                    'top:' + py + 'px',
                    'width:10px','height:10px',
                    'margin:-5px 0 0 -5px',
                    'background:red','border-radius:50%','box-sizing:border-box',
                    'box-shadow:0 0 2px #000',
                ].join(';');
                overlay.appendChild(dot);
                document.documentElement.appendChild(overlay);
                const vv = window.visualViewport;
                console.log('[autoexam] click CSS', px, py, 'viewport', window.innerWidth, window.innerHeight,
                    'dpr', window.devicePixelRatio,
                    vv ? ('vv offset', vv.offsetLeft, vv.offsetTop, 'scale', vv.scale) : 'no-vv');
            }""",
            [target_x, target_y],
        )

        for round_i in range(click_count):
            dom_ok: Dict[str, Any] = {"ok": False}
            if use_dom_click:
                dom_ok = await page.evaluate(
                    """([vx, vy]) => {
                        function dispatchOn(el, cx, cy, win) {
                            if (!el || !win) return;
                            const base = { bubbles: true, cancelable: true, clientX: cx, clientY: cy, view: win };
                            el.dispatchEvent(new PointerEvent('pointerdown', Object.assign({
                                pointerId: 1, pointerType: 'mouse', isPrimary: true, button: 0, buttons: 1
                            }, base)));
                            el.dispatchEvent(new MouseEvent('mousedown', Object.assign({ button: 0, buttons: 1 }, base)));
                            el.dispatchEvent(new MouseEvent('mouseup', Object.assign({ button: 0, buttons: 0 }, base)));
                            el.dispatchEvent(new MouseEvent('click', Object.assign({ button: 0, buttons: 0 }, base)));
                        }
                        let el = document.elementFromPoint(vx, vy);
                        let depth = 0;
                        let localX = vx;
                        let localY = vy;
                        while (el && (el.tagName === 'IFRAME' || el.tagName === 'FRAME') && depth < 24) {
                            let idoc = null;
                            try {
                                idoc = el.contentDocument || (el.contentWindow && el.contentWindow.document);
                            } catch (e) {
                                return { ok: false, crossOrigin: true, depth: depth };
                            }
                            if (!idoc) {
                                return { ok: false, crossOrigin: true, depth: depth };
                            }
                            const r = el.getBoundingClientRect();
                            localX = vx - r.left;
                            localY = vy - r.top;
                            el = idoc.elementFromPoint(localX, localY);
                            depth += 1;
                        }
                        if (!el) {
                            return { ok: false, reason: 'elementFromPoint null', depth: depth };
                        }
                        const win = el.ownerDocument.defaultView;
                        dispatchOn(el, localX, localY, win);
                        return {
                            ok: true,
                            tag: el.tagName,
                            depth: depth,
                            cls: (el.className || '').toString().slice(0, 120),
                        };
                    }""",
                    [target_x, target_y],
                )
            need_mouse = bool(use_playwright_mouse) and (
                after_dom_use_mouse
                or not use_dom_click
                or not (isinstance(dom_ok, dict) and dom_ok.get("ok"))
                or (isinstance(dom_ok, dict) and dom_ok.get("crossOrigin"))
            )
            if need_mouse:
                await page.mouse.move(target_x, target_y)
                await page.mouse.down()
                await asyncio.sleep(0.1)
                await page.mouse.up()
            if round_i < click_count - 1:
                await asyncio.sleep(0.05)

    async def click_option_labels(
        self,
        answer_labels: List[str],
        option_centers: Dict[str, Tuple[float, float]],
        *,
        pause_between_s: Tuple[float, float] = (0.35, 0.85),
        click_count: int = 2,
    ) -> None:
        """
        根据 OCR 得到的 ``option_centers``（键为 ``A``/``B``/``C``/``D``）依次点击 LLM 给出的选项。

        坐标须与**最近一次同视口截图**的 OCR 结果一致；多选时按字母顺序依次点击，同字母只点一次。
        默认 ``click_count=2``（双击），便于部分页面正确触发选中；若需单击可传 ``click_count=1``。
        """
        order: List[str] = []
        for raw in answer_labels:
            L = str(raw).strip().upper()[:1]
            if L not in "ABCD":
                continue
            if L in order:
                continue
            order.append(L)
        if not order:
            raise ValueError("answer_labels 中无有效选项 A～D")

        for i, L in enumerate(order):
            if L not in option_centers:
                raise RuntimeError(
                    f"OCR 未提供选项 {L} 的中心坐标，当前键: {sorted(option_centers.keys())}"
                )
            x, y = option_centers[L]
            if i > 0:
                lo, hi = pause_between_s
                await asyncio.sleep(random.uniform(lo, hi))
            await self.human_click_by_coordinate(float(x), float(y), click_count=click_count)

    async def close(self) -> None:
        """
        释放 Playwright 资源并断开 CDP 连接。

        对 ``connect_over_cdp`` 得到的 ``Browser`` 调用 ``close()`` 在 Playwright 语义下
        **仅断开自动化会话**，不会关闭用户本机已打开的 Chrome 进程。
        """
        self._page = None
        self._context = None

        if self._browser is not None:
            await self._browser.close()
            self._browser = None

        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None


async def _demo() -> None:
    """
    演示流：连接本机调试端口上的 Chrome → 截取当前视口 → 保存到本地 → 断开连接。

    使用前请先启动 Chrome（示例，Windows 下路径请按本机修改）::

        "C:\\\\Program Files\\\\Google\\\\Chrome\\\\Application\\\\chrome.exe" --remote-debugging-port=9222

    然后在本仓库中执行（需在已安装 playwright 与浏览器驱动的环境中）::

        python -m core.browser_agent

    或在 ``auto_exam_agent`` 目录下::

        python core/browser_agent.py
    """
    agent = BrowserAgent(
        # 扩展位：若顶栏遮挡题干，可传入选择器列表，例如：
        # hide_selectors_before_capture=["header"],
    )
    try:
        await agent.connect(port=9222)
        save_to = await agent.capture_exam_area()
        print(f"Screenshot saved: {save_to}")
    finally:
        await agent.close()
        # 避免 Windows cp932 控制台无法编码中文导致演示脚本异常退出
        print("Playwright CDP disconnected (Chrome window left open).")


if __name__ == "__main__":
    asyncio.run(_demo())
