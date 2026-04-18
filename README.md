# Deforest-ZHS

**项目 / 仓库名称：Deforest-ZHS**（以下文档中与路径、克隆命令中的目录名均以此为准。）

**Deforest-ZHS** 是基于 **Playwright (CDP) + PaddleOCR + DeepSeek** 的 Python 自动答题脚本：

- 从「作业/考试列表」扫描**可作答入口**（「开始答题 / 去作答 / 继续答题」等）；
- 点击后自动切换到**新开的考试标签页**，OCR 截图识别题干与选项坐标、DeepSeek 给出答案、Playwright 按钮点选；
- 答完最后一题后自动**提交作业**（含二次确认与 iframe 内按钮），并**关闭考试标签**回到列表刷新；
- 进入考试页时检测「当前是否已提交/已结束」，若已完成可仅走提交流程；
- 本地 `cache/exam_completion.json` 留存**已完成考试名称**，避免重复答题。

> **免责声明**：本项目仅用于个人学习与自动化技术研究，请遵守所在平台的服务条款，勿用于违反考试诚信、破坏平台秩序或涉嫌作弊的用途；由此产生的任何后果与作者无关。

---

## 目录结构

```
Deforest-ZHS/
|-- main.py                 # 入口：CLI / 批量 / 单场 / 列表抓取测试
|-- config.example.yaml     # 配置样例（复制为 config.yaml 再填值）
|-- requirements.txt        # Python 依赖
|-- core/
|   |-- browser_agent.py    # Playwright + CDP 的封装（截图 / 点选 / 标签切换）
|   |-- ocr_engine.py       # PaddleOCR 封装：整段文本 + A/B/C/D 坐标
|   |-- llm_brain.py        # DeepSeek（OpenAI 兼容）封装 + JSON 解析
|-- utils/
|   |-- image_utils.py
|   |-- logger.py
|   |-- paths.py            # project_root() / cache_dir() 等
|-- cache/                  # 运行缓存（.gitignore 已忽略）
|-- click_test_b.py         # 选项点击调试用
```

---



## 快速开始（Windows / macOS / Linux）

建议顺序：**Python 环境 -> 克隆与依赖 ->（可选）Playwright -> 配置 -> 启动 Chrome（浏览器目标）-> 运行脚本**。

### 1. 准备 Python

项目基于 **PaddleOCR / paddlepaddle**，官方发行包目前仅对 **Python 3.8 ~ 3.12** 提供 wheel；**Python 3.14** 通常尚无对应 wheel，请避免使用。

- **推荐 Python 3.10**。
- **Windows 用户**：若 `python` 指向 3.13 / 3.14，请改用 3.10 的绝对路径或通过 `py -3.10` 调用。

### 2. 获取代码并安装依赖

```bash
git clone https://github.com/LZOI-ZJX/Deforest-ZHS.git
cd Deforest-ZHS
```

创建虚拟环境（建议）：

- **Windows PowerShell**

  ```powershell
  py -3.10 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  ```

- **macOS / Linux**

  ```bash
  python3.10 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  ```

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

> Windows 下若 `pip` 读 `requirements.txt` 报编码错误（如 cp932），先执行：`$env:PYTHONUTF8 = '1'`（PowerShell）再安装。

### 3. 安装 Playwright 浏览器（可选）

主流程**不**由 Playwright 拉起浏览器进程，而是通过 **CDP** 连接你已打开的 Chrome，因此 **不必**安装 Playwright 自带的 Chromium。若你在其它脚本里会 `chromium.launch()`，再执行：

```bash
python -m playwright install chromium
```

### 4. 准备配置

```bash
# Windows PowerShell
Copy-Item config.example.yaml config.yaml
# macOS / Linux
cp config.example.yaml config.yaml
```

编辑 `config.yaml`（对照 `config.example.yaml`）：

- 默认仅对接 **DeepSeek**（见 `core/llm_brain.py`）；换模型需自行改代码。
- `deepseek.api_key`：DeepSeek API Key（https://platform.deepseek.com ）。
- **`browser.cdp_port`**：必须与下面「浏览器目标」中的 **`--remote-debugging-port`** 一致（默认 **9222**）。
- `exam.submit`：是否答完后自动提交；`false` 时也可用命令行 `--no-submit` 强制不提交。
- `exam.default_total_questions`：DOM 读不到总题数时的兜底题量。

`config.example.yaml` 的 `browser:` 下有 Windows 启动示例注释；**不要把 `config.yaml` 提交到 Git**（已在 `.gitignore` 中忽略）。

### 5. 浏览器目标（Chrome 启动参数与 CDP）

脚本通过 **`BrowserAgent.connect(port=...)`** 连接本机 Chrome 的 **Chrome DevTools Protocol（CDP）**。因此你启动 Chrome 时必须带上**固定的远程调试端口**，并指定**独立用户数据目录**（与日常 Chrome 配置隔离，避免冲突与误关数据）。

#### 5.1 两个必须理解的启动参数

| 参数 | 作用 |
| ---- | ---- |
| `--remote-debugging-port=9222` | 在本机打开 CDP 调试端口，供 Playwright 附着；端口号须与 `config.yaml` 里 **`browser.cdp_port`** 一致。 |
| `--user-data-dir=...` | Chrome **单独一套**用户数据（登录态、Cookie、扩展等）。**不要**与日常使用的 Chrome 用户目录相同。本仓库约定放在项目根下的 **`chomepic`** 文件夹（已在 `.gitignore` 中忽略）。 |

#### 5.2 Windows 路径写法（`user-data-dir`）

- 使用 **盘符 + 反斜杠**，例如：`D:\tools\fuckxjp\Deforest-ZHS\chomepic`。
- 路径放在 **英文双引号**内。
- 若克隆位置不是 `D:\tools\fuckxjp\Deforest-ZHS`，请把 **`Deforest-ZHS` 所在路径** 与 `\chomepic` 拼接成你自己的目录；不存在时 Chrome 会创建。

#### 5.3 推荐：命令行启动（Windows）

Chrome 默认安装路径示例（按本机安装位置修改）：

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" `
  --remote-debugging-port=9222 `
  --user-data-dir="D:\tools\fuckxjp\Deforest-ZHS\chomepic"
```

等价一行（CMD）：

```bat
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="D:\tools\fuckxjp\Deforest-ZHS\chomepic"
```

#### 5.4 可选：桌面快捷方式「目标」附加参数（Windows）

适合固定一套参数、双击启动：

1. 右键桌面上的 **Google Chrome** 快捷方式，选择 **属性**。
2. 在 **快捷方式** 选项卡的 **目标(T)** 中，在**原有 `chrome.exe` 路径及结束引号之后**先输入**一个空格**，再粘贴：

   ```text
   --remote-debugging-port=9222 --user-data-dir="D:\tools\fuckxjp\Deforest-ZHS\chomepic"
   ```

3. 若仓库不在 `D:` 盘，请把引号内路径改成你的 **`...\Deforest-ZHS\chomepic`**。

**示例（目标框整行形态）**：

```text
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="D:\tools\fuckxjp\Deforest-ZHS\chomepic"
```

#### 5.5 验证 CDP 是否可用

在**已用上述参数启动**的 Chrome 外，用**另一个**普通浏览器窗口打开：

```text
http://127.0.0.1:9222/json/version
```

若能返回 JSON（含 `Browser` 等字段），说明 **浏览器目标（端口）就绪**。脚本连接时使用 **127.0.0.1**，避免个别环境下 `localhost` 解析到 IPv6 导致失败。

#### 5.6 登录与列表页

在上述 Chrome 中完成课程平台登录，打开 **作业/考试列表**，并将该标签**置于当前窗口最前**（脚本附着后取默认上下文中的活动页）。

#### 5.7 macOS / Linux（自行替换路径）

同样使用两个参数；`user-data-dir` 建议指向本仓库旁的独立目录，例如：

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/Deforest-ZHS/chomepic"
```

确保 `config.yaml` 里 `cdp_port` 与 `--remote-debugging-port` 一致。

### 6. 跑起来

确保当前焦点标签 = 作业/考试列表，然后：

```bash
python main.py
```

观察日志直到出现「**所有考试已完成！**」为止。

---

## CLI 参数

```text
python main.py [--no-submit] [--list-url URL] [--test-list-scrape]
```

| 参数 | 含义 |
| ---- | ---- |
| `--no-submit` | 答完不自动提交（仅答题与翻页，方便调试） |
| `--list-url URL` | 列表页备用 URL；当 `go_back` 失败时用于 `page.goto` 恢复扫描 |
| `--test-list-scrape` | 仅连接 CDP 并打印**当前标签页**上扫描到的考试/作业条目，不调用 LLM，用于验证识别覆盖 |

---

## 运行时会生成哪些缓存？

所有缓存都在 **Deforest-ZHS** 项目根目录下的 `cache/`（相对路径，已被 `.gitignore` 忽略）：

- `viewport.png` / `_test_screenshot_ocr.png`：最新视口截图；
- `viewport_ocr.json` / `viewport_ocr.txt`：OCR 结构化结果；
- `exam_session_YYYYMMDD_HHMMSS.md / .txt`：每次运行的会话日志；
- `exam_completion.json`：**本地已完成考试的名称集合**（OCR 出来的标题）。删除它即可清空「已完成」记忆。

---

## 常见问题

### Q1：跑起来一直卡在「连接 CDP 端口 9222 ...」

- 按 **「5. 浏览器目标」** 检查 Chrome 是否已用 **`--remote-debugging-port`** 启动，且端口与 **`config.yaml` 中 `browser.cdp_port`** 一致。
- 用浏览器打开 `http://127.0.0.1:9222/json/version`，应能返回 JSON；若不能，说明 CDP 未监听或未用正确参数启动。
- 连接地址使用 **127.0.0.1**；若仅用 `localhost` 失败，多为 IPv6 解析问题，脚本内已优先 IPv4。

### Q2：`paddlepaddle` 装不上 / `paddleocr` 找不到模块

- 确认解释器是 **Python 3.10 ~ 3.12**（3.13/3.14 暂无 wheel）；
- `pip install -r requirements.txt` 失败可改手动：
  ```bash
  pip install "paddlepaddle>=2.5.0"
  pip install "paddleocr>=2.7.0"
  ```
- 国内网络可走镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`。

### Q3：OCR 模型首次会下载

PaddleOCR 会把 `PP-OCRv5` 等模型下载到用户主目录的 `~/.paddlex/official_models/`。首次启动稍慢属于正常现象；之后会走缓存。

### Q4：列表里明明有「开始答题」但被跳过

这是 `list_entry_row_looks_completed` 的本地白名单太激进。若贵平台用了特殊词（例如「已提交卷」「已阅卷」），可以在 `main.py` 的同名函数里把正则扩一下。你也可以先用：

```bash
python main.py --test-list-scrape
```

查看当前能抓到的行文本，再决定正则要不要改。

### Q5：最后一题作答了但没点到「提交作业」

- `submit_exam_with_confirm` 会遍历主文档 + 所有 iframe，按「提交作业 / 交卷 / 保存并提交」等关键字点击，并接受原生 `confirm` 弹窗；
- 若你们平台用了完全不同的提交控件文案，请在日志里找 `[提交]` 行，把真实按钮名发给脚本作者或自行追加到 `submit_labels` 元组。

### Q6：是否支持多选题？

支持。`LLMBrain.get_answer` 的 `answer` 字段是数组，脚本按字母顺序依次 DOM 点击。

---

## 开发者提示

- **代码风格**：类型标注 + `from __future__ import annotations`，三方 API 只通过 `core/*.py` 统一封装。
- **日志**：脚本侧用 `_log()` 函数集中打印；会话结束时会把所有行写入 `cache/exam_session_*.md`。
- **Playwright 接管**：`BrowserAgent.close()` 仅解开 Playwright session，**不会**关闭你真实的 Chrome 窗口。
- **不要**提交 `config.yaml`；`config.example.yaml` 才是公共模板。
- **修改代码后**快速自检：
  ```bash
  python -m py_compile main.py core/*.py utils/*.py
  ```

欢迎二次开发（更多平台入口文案、多标签窗口兼容、可选 UI 监控等）。

---

## License

建议上传前在仓库根目录再新增一份许可证（如 MIT），本项目示例文件未附带 License 以避免误导。
