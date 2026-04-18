"""封装 DeepSeek：OpenAI 兼容 Chat Completions、Prompt 与 JSON 解析。"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict, Union

import httpx
import yaml

from utils.paths import project_root

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
ENV_API_KEY = "DEEPSEEK_API_KEY"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# 考试作答：题干可来自 OCR 或 DOM；模型须仅输出 JSON（若偶发带 ``` 仍会在解析时剥离）
EXAM_SYSTEM_PROMPT = """你是一个高智商的考试解答智能体。你的任务是阅读考试题目文本（包含题干和选项），并输出绝对准确的答案。

【输入特征说明】
1. 文本可能来自网页 OCR 截图或 DOM 可见文本；可能存在轻微错别字、页眉页脚噪声、标点丢失或换行混乱，请自动纠噪并理解核心题意。
2. 题目可能是「单选题」，也可能是「多选题」。你需要根据题意或常识自行判断。

【强制输出约束】
你必须且只能输出一个合法的 JSON 对象，不要包含任何解释性文本，不要包含 Markdown 格式的语法（例如不要输出 ```json 和 ```）。
JSON 只能包含一个键 answer，格式如下：
{
    "answer": ["A", "B"]
}

【answer 字段特殊要求】
- 必须是一个数组 (Array)。
- 数组内的元素必须是选项的大写字母。
- 如果是单选题，输出示例：["C"]
- 如果是多选题，输出示例：["A", "B", "D"]
"""

# 解析失败时追加一轮对话，要求模型严格只输出可解析 JSON
EXAM_RETRY_USER_PROMPT = """你上一条回复无法被程序解析为合法答案。请只输出**一个** JSON 对象，不要任何其它文字、不要用 markdown 代码块。
格式必须是：{"answer":["X"]}，X 为大写字母 A～D；多选示例：{"answer":["A","B","D"]}。"""


class LLMBrainError(RuntimeError):
    """DeepSeek 请求失败或响应异常。"""


class ChatMessage(TypedDict):
    role: str
    content: str


def _load_deepseek_section() -> Dict[str, Any]:
    path = project_root() / "config.yaml"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("deepseek") or {}


def _strip_json_fences(text: str) -> str:
    """去掉 ```json ... ``` 或 ``` ... ``` 包裹。"""
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", t, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def _extract_first_json_object(text: str) -> str:
    """从可能含前后废话的文本中取出第一个 ``{...}`` 片段（括号平衡）。"""
    s = _strip_json_fences(text.strip())
    start = s.find("{")
    if start < 0:
        return s
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s


# 从 answer 键后截取数组（单/双引号键名，不含嵌套 []）
_RE_ANSWER_KEY_ARRAY = re.compile(
    r'["\']answer["\']\s*:\s*(\[[^\]]*\])',
    re.IGNORECASE | re.DOTALL,
)
# 文本中任意 [...] 内带引号字母
_RE_BRACKET_LETTERS = re.compile(r'\[\s*([^\]]+)\s*\]')
_RE_QUOTED_LETTER = re.compile(r'["\']([A-Da-d])["\']')
# 无 JSON 时：答案：C / 选A / Answer: B
_RE_LOOSE_ANSWER = re.compile(
    r"(?:答案|选择|选项|Answer)\s*[:：]?\s*([ABCDabcd](?:\s*[,，、]\s*[ABCDabcd])*)",
    re.IGNORECASE,
)


def _coerce_answer_value(ans: Any) -> Optional[List[str]]:
    """将模型给出的 answer 字段规范为 ['A','B']，非法则返回 None。"""
    if ans is None:
        return None
    if isinstance(ans, str):
        letters = [c for c in ans.upper() if c in "ABCD"]
        # 连续字母 "AC" -> ['A','C']
        if letters and all(len(x) == 1 for x in letters):
            return letters
        return None
    if isinstance(ans, list):
        out: List[str] = []
        for x in ans:
            if not isinstance(x, str):
                continue
            s = x.strip().upper()
            if len(s) == 1 and s in "ABCD":
                out.append(s)
        return out if out else None
    return None


def _json_loads_array_loose(arr_str: str) -> Any:
    """解析 ``[...]`` 片段；兼容单引号（非严格 JSON）。"""
    s = arr_str.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(s.replace("'", '"'))
    except json.JSONDecodeError:
        return None


def _extract_answer_from_text(raw: str) -> Optional[List[str]]:
    """
    从模型原文中提取选项列表：优先整段 JSON，其次正则截取 ``"answer"`` 数组、``[...]`` 片段，
    最后尝试「答案：A」类行内表述。
    """
    t = _strip_json_fences(raw.strip())

    # 1) 完整 JSON 对象（兼容键/字符串使用单引号）
    blob = _extract_first_json_object(t)
    data: Any = None
    if blob:
        for candidate in (blob, blob.replace("'", '"')):
            try:
                data = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
    if isinstance(data, dict) and "answer" in data:
        got = _coerce_answer_value(data["answer"])
        if got:
            return got

    # 2) 仅截取 "answer" : [...] 再 json.loads
    m = _RE_ANSWER_KEY_ARRAY.search(t)
    if m:
        arr = _json_loads_array_loose(m.group(1))
        if arr is not None:
            got = _coerce_answer_value(arr)
            if got:
                return got

    # 3) 任意 [...] 内出现 "A" "B"
    for bm in _RE_BRACKET_LETTERS.finditer(t):
        inner = bm.group(1)
        letters = _RE_QUOTED_LETTER.findall(inner)
        if letters:
            got = _coerce_answer_value([x.upper() for x in letters])
            if got:
                return got

    # 4) 单行「答案：A」或「答案：A,B」
    lm = _RE_LOOSE_ANSWER.search(t.replace("\n", " "))
    if lm:
        chunk = lm.group(1)
        letters = [c.upper() for c in chunk if c.upper() in "ABCD"]
        if letters:
            return letters

    return None


class LLMBrain:
    """
    与 DeepSeek 交互（``https://api.deepseek.com``，OpenAI 兼容 ``/v1/chat/completions``）。

    **API Key 优先级**：构造函数参数 ``api_key`` > 环境变量 ``DEEPSEEK_API_KEY`` > ``config.yaml`` 中 ``deepseek.api_key``。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        cfg = _load_deepseek_section()
        key = (api_key or os.environ.get(ENV_API_KEY) or cfg.get("api_key") or "").strip()
        if not key:
            raise ValueError(
                f"未配置 DeepSeek API Key：请设置环境变量 {ENV_API_KEY}，或在 config.yaml 的 deepseek.api_key 中填写"
            )
        self._api_key = key
        self._base_url = (base_url or cfg.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._model = model or cfg.get("model") or DEFAULT_MODEL
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def _chat_url(self) -> str:
        return f"{self._base_url}{CHAT_COMPLETIONS_PATH}"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        调用 Chat Completions，返回**助手**回复的纯文本（``choices[0].message.content``）。

        ``messages`` 为 OpenAI 格式，例如
        ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``。
        """
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if stop is not None:
            payload["stop"] = stop
        if extra_body:
            payload.update(extra_body)

        try:
            with httpx.Client(timeout=self._timeout) as client:
                r = client.post(self._chat_url(), headers=self._headers(), json=payload)
        except httpx.RequestError as e:
            raise LLMBrainError(f"DeepSeek 请求失败: {e}") from e

        if r.status_code >= 400:
            raise LLMBrainError(
                f"DeepSeek HTTP {r.status_code}: {r.text[:2000]}",
            )

        try:
            data = r.json()
        except json.JSONDecodeError as e:
            raise LLMBrainError(f"DeepSeek 返回非 JSON: {r.text[:500]}") from e

        try:
            choice = data["choices"][0]
            msg = choice["message"]
            content = msg.get("content")
        except (KeyError, IndexError, TypeError) as e:
            raise LLMBrainError(f"DeepSeek 响应结构异常: {data!r}") from e

        if content is None:
            return ""
        if isinstance(content, str):
            return content
        # 少数实现可能返回多段 content
        return str(content)

    def chat_json(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        在对话结束后将助手回复解析为 JSON（自动去除 ```json 代码块``）。
        """
        text = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        raw = _strip_json_fences(text)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMBrainError(f"助手回复不是合法 JSON: {raw[:800]}") from e

    def simple_ask(self, user_text: str, *, system: Optional[str] = None, **kwargs: Any) -> str:
        """单轮问答：可选 ``system``，一条 ``user``。"""
        msgs: List[ChatMessage] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user_text})
        return self.chat(msgs, **kwargs)

    def _exam_chat_answer(
        self,
        *,
        user_content: str,
        system_prompt: str = EXAM_SYSTEM_PROMPT,
        temperature: float = 0.2,
        max_tokens: Optional[int] = 2048,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        messages: List[ChatMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        last_raw = ""
        for attempt in range(max_retries):
            raw = self.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            last_raw = raw
            letters = _extract_answer_from_text(raw)
            if letters:
                return {"answer": letters}

            if attempt >= max_retries - 1:
                break
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": EXAM_RETRY_USER_PROMPT})

        raise LLMBrainError(
            f"经 {max_retries} 次尝试仍无法从模型回复中解析出合法 answer。"
            f"\n最后一次回复片段:\n{last_raw[:2000]}"
        )

    def exam_answer_from_ocr(
        self,
        ocr_text: str,
        *,
        system_prompt: str = EXAM_SYSTEM_PROMPT,
        temperature: float = 0.2,
        max_tokens: Optional[int] = 2048,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        根据 OCR 题目全文调用 DeepSeek，返回 ``{"answer": ["A", ...]}``。

        对模型回复先用 JSON 与正则多层提取；若仍无法得到合法选项，则自动追加用户消息**重新询问**（最多 ``max_retries`` 轮完整请求链）。
        """
        user_content = (
            "以下是通过 OCR 提取的题目全文，请严格按照系统提示只输出 JSON：\n\n" + ocr_text.strip()
        )
        return self._exam_chat_answer(
            user_content=user_content,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    def exam_answer_from_dom(
        self,
        page_text: str,
        *,
        system_prompt: str = EXAM_SYSTEM_PROMPT,
        temperature: float = 0.2,
        max_tokens: Optional[int] = 2048,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """根据 ``document.body`` 可见文本调用 DeepSeek，返回 ``{"answer": ["A", ...]}``。"""
        user_content = (
            "以下是通过页面 DOM（inner_text）提取的可见文本，可能含页眉页脚；请聚焦题干与选项作答，"
            "并严格按照系统提示只输出 JSON：\n\n" + page_text.strip()
        )
        return self._exam_chat_answer(
            user_content=user_content,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    def get_answer(
        self,
        question_text: str,
        *,
        from_ocr: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        主流程统一入口：根据题目全文返回 ``{"answer": ["A", ...]}``。

        - ``from_ocr=False``（默认）：按 DOM 提取的常见排版调用 :meth:`exam_answer_from_dom`。
        - ``from_ocr=True``：按 OCR 提取文本调用 :meth:`exam_answer_from_ocr`。
        """
        t = (question_text or "").strip()
        if not t:
            raise LLMBrainError("题目文本为空，无法作答")
        if from_ocr:
            return self.exam_answer_from_ocr(t, **kwargs)
        return self.exam_answer_from_dom(t, **kwargs)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from utils.paths import cache_dir

    brain = LLMBrain()
    ocr_path = cache_dir() / "viewport_ocr.txt"
    if ocr_path.is_file():
        ocr_body = ocr_path.read_text(encoding="utf-8")
        print("=== OCR 来源:", ocr_path.resolve(), "===")
    else:
        ocr_body = (
            "1.【单选题】(2分) 示例：2+2等于？ A. 3 B. 4 C. 5 D. 6"
        )
        print("=== 未找到 cache/viewport_ocr.txt，使用内置示例题 ===")

    result = brain.exam_answer_from_ocr(ocr_body)
    print(json.dumps(result, ensure_ascii=False, indent=2))
