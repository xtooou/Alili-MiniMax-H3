"""Alili API · ComfyUI 独立模块（单文件版）"""
import asyncio
import json
import os
import urllib.error
import urllib.request

from aiohttp import web
from server import PromptServer

# ==============================================================================
# 配置定义与读写
# ==============================================================================

PLATFORMS = {
    'deepseek':       {'name': 'DeepSeek (深度求索)',      'url': 'https://api.deepseek.com',                             'defaultModel': 'deepseek-flash'},
    'siliconflow':    {'name': 'SiliconFlow (硅基流动)',  'url': 'https://api.siliconflow.cn/v1',                        'defaultModel': 'zai-org/GLM-5.3-Flash'},
    'gemini':         {'name': 'Google AI Studio (Gemini)', 'url': 'https://generativelanguage.googleapis.com/v1beta/openai', 'defaultModel': 'gemini-2.5-flash'},
    'dashscope':      {'name': '阿里云百炼 (通义千问)',    'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',  'defaultModel': 'qwen-vl-max'},
    'openai':         {'name': 'OpenAI 官方',            'url': 'https://api.openai.com/v1',                            'defaultModel': 'gpt-4o'},
    'openrouter':     {'name': 'OpenRouter (海外聚合)',   'url': 'https://openrouter.ai/api/v1',                        'defaultModel': 'anthropic/claude-3.5-sonnet'},
    'zhipu':          {'name': '智谱 GLM',               'url': 'https://open.bigmodel.cn/api/paas/v4',                 'defaultModel': 'glm-4v'},
    'ollama':         {'name': 'Ollama (本地离线)',       'url': 'http://localhost:11434/v1',                           'defaultModel': 'qwen2.5:latest'},
    'runninghub_cn':  {'name': 'RunningHub (国内版)',     'url': 'https://api.runninghub.cn/v1',                         'defaultModel': 'openai/gpt-4o'},
    'runninghub_ai':  {'name': 'RunningHub (海外版)',     'url': 'https://api.runninghub.ai/v1',                         'defaultModel': 'openai/gpt-4o'},
    'custom':         {'name': '自定义平台 (OpenAI 兼容)', 'url': '',                                                     'defaultModel': ''},
}

KEY_OPTIONAL = {'ollama', 'custom'}
DEFAULT_PLATFORM = 'siliconflow'
DEFAULT_MAX_TOKENS = 4096
CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".ai_cache", "ai_api_config.json")
PREFIX = "/alili/h3/api"
_REGISTERED = False


def normalize_config(raw):
    """补全默认值"""
    if not isinstance(raw, dict):
        raw = {}

    saved = raw.get("platforms") if isinstance(raw.get("platforms"), dict) else {}

    platforms = {}
    for pid, preset in PLATFORMS.items():
        s = saved.get(pid) if isinstance(saved.get(pid), dict) else {}
        platforms[pid] = {
            "apiKey": str(s.get("apiKey") or ""),
            "baseUrl": str(s.get("baseUrl") or "") if pid == "custom" else preset["url"],
            "model": str(s.get("model") or preset["defaultModel"]),
        }

    try:
        max_tokens = max(1, int(raw.get("maxTokens", DEFAULT_MAX_TOKENS)))
    except (TypeError, ValueError):
        max_tokens = DEFAULT_MAX_TOKENS
    pid = raw.get("platform") if raw.get("platform") in PLATFORMS else DEFAULT_PLATFORM
    return {"platform": pid, "maxTokens": max_tokens, "platforms": platforms}


def read_config():
    """读取配置文件并补全默认值。"""
    raw = {}
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
    except Exception:
        raw = {}
    return normalize_config(raw)


def save_config(cfg):
    """持久化保存配置，成功返回 True。"""
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        try:
            os.chmod(CONFIG_FILE, 0o600)
        except Exception:
            pass
        return True
    except Exception:
        return False


def get_active_settings():
    """取当前选中厂家的最终调用参数。"""
    cfg = read_config()
    pid = cfg["platform"]
    p = cfg["platforms"][pid]
    return {
        "platform": pid,
        "baseUrl": p["baseUrl"].strip().rstrip("/"),
        "apiKey": p["apiKey"].strip(),
        "model": p["model"].strip(),
        "maxTokens": cfg["maxTokens"],
    }


# ==============================================================================
# 请求层 (Client)
# ==============================================================================

class ApiError(RuntimeError):
    """所有 API 调用错误统一抛这个，message 可直接展示给用户。"""


def build_headers(api_key):
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _error_text(data: bytes):
    """从错误响应体里取出可读信息。"""
    try:
        body = json.loads(data.decode("utf-8", "replace"))
        err = body.get("error", body) if isinstance(body, dict) else body
        if isinstance(err, dict):
            return str(err.get("message") or err)
        return str(err)
    except Exception:
        return data.decode("utf-8", "replace")[:300] or "未知错误"


def check_settings(settings=None):
    """返回 (是否可用, 提示信息)。"""
    s = settings or get_active_settings()
    if not s["apiKey"] and s["platform"] not in KEY_OPTIONAL:
        return False, "未检测到 API Key，请先点击设置齿轮配置！"
    if not s["baseUrl"] or not s["model"]:
        return False, "接口地址或模型名称未设置，请在设置中配置！"
    return True, ""


def _chat_url(base_url):
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def _extract_text(res_data):
    choices = (res_data.get("choices") if isinstance(res_data, dict) else None) or []
    if not choices:
        raise ApiError("接口响应成功，但未返回 choices 内容")
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    content = str(content).strip()
    if content:
        return content
    finish = str(choice.get("finish_reason") or "unknown")
    if str(message.get("reasoning_content") or "").strip():
        raise ApiError(f"接口只返回了思考过程，未输出正文（finish_reason={finish}），请在设置里调大最大 Token")
    raise ApiError(f"接口返回了空内容（finish_reason={finish}）")


def chat(messages, *, settings=None, max_tokens=None, temperature=None, timeout=120):
    """调用当前厂家的 chat/completions，返回回复文本；失败抛 ApiError。"""
    s = settings or get_active_settings()
    ok, msg = check_settings(s)
    if not ok:
        raise ApiError(msg)

    payload = {
        "model": s["model"],
        "messages": messages,
        "max_tokens": int(max_tokens or s["maxTokens"]),
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    req = urllib.request.Request(_chat_url(s["baseUrl"]), data=json.dumps(payload).encode("utf-8"),
                                 headers=build_headers(s["apiKey"]), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            raw = res.read()
    except urllib.error.HTTPError as e:
        raise ApiError(f"API 请求失败 ({e.code}): {_error_text(e.read())}")
    except TimeoutError:
        raise ApiError(f"网络请求超时（{timeout} 秒无响应）")
    except urllib.error.URLError as e:
        raise ApiError(f"网络请求异常: {e.reason}")
    except Exception as e:
        raise ApiError(f"网络请求异常: {e}")

    try:
        res_data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise ApiError(f"接口返回内容无法解析: {e}")
    return _extract_text(res_data)


def fetch_models(base_url, api_key, timeout=15):
    """GET {baseUrl}/models，返回排序去重后的模型 id 列表；失败抛 ApiError。"""
    req = urllib.request.Request(f"{base_url.rstrip('/')}/models",
                                 headers=build_headers(api_key), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            data = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise ApiError(f"HTTP {e.code}: {_error_text(e.read())}")
    except Exception as e:
        raise ApiError(str(getattr(e, "reason", None) or e))
    raw_list = data.get("data", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    models = [item if isinstance(item, str) else item.get("id", "") for item in raw_list]
    return sorted({m for m in models if m})


# ==============================================================================
# 前端 JS 脚本内容
# ==============================================================================

SETTINGS_JS = r"""import { api } from "/scripts/api.js";

const ROUTE = "/alili/h3/api";
const COLOR_ERR = "#e5737f", COLOR_OK = "#6fcf97", COLOR_MUTED = "#8294a2";

async function request(path, options) {
    const res = await api.fetchApi(`${ROUTE}${path}`, options);
    let data = {};
    try { data = await res.json(); } catch {}
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
}

const postJson = (path, body) => request(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const loadApiConfig = () => request("/config");

export async function getApiLabel() {
    try {
        const cfg = await loadApiConfig();
        const preset = cfg.presets[cfg.platform];
        const p = cfg.platforms[cfg.platform];
        if (!p.model || (!p.apiKey.trim() && !preset.keyOptional)) return "";
        return `${preset.name.split(" (")[0]}: ${p.model.split("/").pop()}`;
    } catch {
        return "";
    }
}

function injectStyle() {
    if (document.getElementById("alili-h3-api-style")) return;
    const style = document.createElement("style");
    style.id = "alili-h3-api-style";
    style.textContent = `
    .alili-api-overlay{position:fixed;inset:0;z-index:10100;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.58);font:13px Arial,sans-serif}
    .alili-api-dialog{width:min(460px,calc(100vw - 30px));box-sizing:border-box;padding:18px 20px;background:#17222c;color:#d7e3ec;border:1px solid #344958;border-radius:10px;box-shadow:0 18px 50px rgba(0,0,0,.5)}
    .alili-api-head{display:flex;align-items:center;gap:8px;margin-bottom:14px;font-size:15px;font-weight:bold}
    .alili-api-head .grow{flex:1}
    .alili-api-row{display:grid;grid-template-columns:84px minmax(0,1fr);align-items:center;gap:8px;margin-bottom:10px}
    .alili-api-row.hidden{display:none}
    .alili-api-field{display:flex;gap:6px;min-width:0;position:relative}
    .alili-api-dialog input,.alili-api-dialog select{flex:1;min-width:0;box-sizing:border-box;height:30px;padding:0 8px;background:#1d2b36;color:#dce7ee;border:1px solid #3a4d5b;border-radius:5px;font:inherit}
    .alili-api-dialog input:focus,.alili-api-dialog select:focus{outline:none;border-color:#4c8fa0}
    .alili-api-btn{height:30px;min-width:30px;padding:0 10px;background:#1d2b36;color:#cbd8e0;border:1px solid #3a4d5b;border-radius:5px;cursor:pointer;font:inherit}
    .alili-api-btn:hover{background:#263b49;border-color:#4c6879}
    .alili-api-btn:disabled{opacity:.6;cursor:default}
    .alili-api-btn.primary{background:#17535a;border-color:#26717a;color:#e0f2f2}
    .alili-api-btn.spin{animation:alili-api-spin 1s linear infinite}
    .alili-api-hint{margin-left:8px;color:${COLOR_MUTED};font-size:12px;align-self:center;white-space:nowrap}
    .alili-api-menu{display:none;position:absolute;left:0;right:0;top:34px;z-index:5;max-height:240px;overflow:auto;background:#1d2b36;border:1px solid #3a4d5b;border-radius:5px;box-shadow:0 8px 24px rgba(0,0,0,.45)}
    .alili-api-menu.open{display:block}
    .alili-api-item{padding:6px 9px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .alili-api-item:hover,.alili-api-item.selected{background:#274252}
    .alili-api-foot{display:flex;align-items:center;gap:8px;margin-top:16px}
    .alili-api-status{flex:1;min-height:34px;display:flex;align-items:center;font-size:12px;line-height:1.35;word-break:break-all}
    @keyframes alili-api-spin{to{transform:rotate(360deg)}}
    .alili-api-h3-field{justify-content:flex-end}
    .alili-api-lang{display:flex;align-items:center;justify-content:flex-end;gap:16px;flex:1;min-width:0}
    .alili-api-lang label{display:flex;align-items:center;gap:5px;cursor:pointer;white-space:nowrap}
    .alili-api-dialog .alili-api-h3-field input[type="radio"]{flex:0 0 auto;width:13px;height:13px;min-width:0;padding:0;margin:0;background:none;border:0;border-radius:0;font:inherit;cursor:pointer;accent-color:#4c8fa0}
    `;
    document.head.append(style);
}

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
}

function row(label, ...children) {
    const wrap = el("div", "alili-api-row");
    wrap.append(el("span", "", label));
    const field = el("div", "alili-api-field");
    field.append(...children);
    wrap.append(field);
    return { wrap, field };
}

export async function openApiSettings({ onSaved, h3 } = {}) {
    injectStyle();
    let cfg;
    try {
        cfg = await loadApiConfig();
    } catch (err) {
        alert("读取 API 配置失败：" + err.message);
        return false;
    }
    const presets = cfg.presets;
    let cur = cfg.platform;
    const modelCache = {};
    let fetchToken = 0;

    const overlay = el("div", "alili-api-overlay");
    const dialog = el("div", "alili-api-dialog");
    overlay.append(dialog);

    const close = el("button", "alili-api-btn", "✕");
    const head = el("div", "alili-api-head");
    head.append(el("span", "", "⚙"), el("span", "grow", "API 设置"), close);

    const platform = document.createElement("select");
    for (const [pid, p] of Object.entries(presets)) platform.append(new Option(p.name, pid));

    const url = document.createElement("input");
    url.placeholder = "https://example.com/v1";

    const key = document.createElement("input");
    key.type = "password";
    key.autocomplete = "off";
    const eye = el("button", "alili-api-btn", "显示");
    eye.title = "显示 / 隐藏 Key";

    const model = document.createElement("input");
    model.placeholder = "选择或直接输入模型名称";
    const dropBtn = el("button", "alili-api-btn", "▾");
    const refresh = el("button", "alili-api-btn", "⟳");
    refresh.title = "拉取模型列表";
    const menu = el("div", "alili-api-menu");

    const tokens = document.createElement("input");
    tokens.type = "number";
    tokens.min = "1";
    tokens.style.cssText = "flex:0 0 110px";
    tokens.value = String(cfg.maxTokens);

    const status = el("div", "alili-api-status");
    const btnCancel = el("button", "alili-api-btn", "取消");
    const btnSave = el("button", "alili-api-btn primary", "保存");

    const rowPlatform = row("厂家", platform);
    const rowUrl = row("接口地址", url);
    const rowKey = row("API Key", key, eye);
    const rowModel = row("模型", model, dropBtn, refresh, menu);
    const rowTokens = row("最大 Token", tokens, el("span", "alili-api-hint", "单次回复上限，如 4096"));

    // H3 专属：输出语言。值由 H3 自己持久化，不写入共享配置文件。
    const h3Values = { output_language: "中文", ...(h3 || {}) };
    const language = el("div", "alili-api-lang");
    for (const value of ["English", "中文"]) {
        const label = el("label");
        const radio = el("input");
        radio.type = "radio";
        radio.name = "alili-h3-output-language";
        radio.value = value;
        radio.checked = String(h3Values.output_language || "中文") === value;
        label.append(radio, el("span", "", value));
        language.append(label);
    }
    const rowLanguage = row("输出语言", language);
    rowLanguage.field.classList.add("alili-api-h3-field");

    const foot = el("div", "alili-api-foot");
    foot.append(status, btnCancel, btnSave);
    dialog.append(head, rowPlatform.wrap, rowUrl.wrap, rowKey.wrap, rowModel.wrap, rowTokens.wrap,
        rowLanguage.wrap, foot);

    const setStatus = (text, color) => {
        status.style.color = color || COLOR_MUTED;
        status.textContent = text.length > 80 ? text.slice(0, 80) + "…" : text;
        status.title = text;
    };

    const setBusy = busy => {
        refresh.disabled = busy;
        refresh.classList.toggle("spin", busy);
    };

    const fillMenu = list => {
        menu.replaceChildren();
        for (const name of list) {
            const item = el("div", "alili-api-item" + (name === model.value.trim() ? " selected" : ""), name);
            item.title = name;
            item.onclick = () => { model.value = name; menu.classList.remove("open"); };
            menu.append(item);
        }
    };

    const stash = () => {
        const p = cfg.platforms[cur];
        p.apiKey = key.value.trim();
        p.model = model.value.trim();
        if (cur === "custom") p.baseUrl = url.value.trim();
    };

    const loadPlatform = pid => {
        cur = pid;
        const p = cfg.platforms[pid];
        const isCustom = pid === "custom";
        platform.value = pid;
        rowUrl.wrap.classList.toggle("hidden", !isCustom);
        url.value = isCustom ? p.baseUrl : "";
        key.value = p.apiKey;
        key.placeholder = pid === "ollama" ? "本地模式无需填写"
            : (isCustom ? "按需填写（可留空）" : "粘贴该平台的 API Key");
        model.value = p.model;
        fillMenu(modelCache[pid] || []);
        menu.classList.remove("open");
        setStatus("");
    };

    platform.onchange = () => {
        if (platform.value === cur) return;
        stash();
        fetchToken++;
        setBusy(false);
        loadPlatform(platform.value);
    };

    eye.onclick = () => {
        const hidden = key.type === "password";
        key.type = hidden ? "text" : "password";
        eye.textContent = hidden ? "隐藏" : "显示";
    };

    dropBtn.onclick = e => {
        e.stopPropagation();
        const list = modelCache[cur] || [];
        if (!list.length) { setStatus("还没有模型列表，点右侧 ⟳ 拉取，或直接输入模型名", COLOR_MUTED); return; }
        fillMenu(list);
        menu.classList.toggle("open");
    };
    dialog.addEventListener("click", () => menu.classList.remove("open"));

    refresh.onclick = async () => {
        if (refresh.disabled) return;
        const base = cur === "custom" ? url.value.trim() : presets[cur].url;
        const apiKey = key.value.trim();
        if (!base) { setStatus("请先填写接口地址", COLOR_ERR); return; }
        if (!apiKey && !presets[cur].keyOptional) { setStatus("请先填写 API Key", COLOR_ERR); return; }
        const token = ++fetchToken;
        const pid = cur;
        setBusy(true);
        setStatus("正在获取模型列表…");
        try {
            const data = await postJson("/models", { baseUrl: base, apiKey });
            if (token !== fetchToken) return;
            if (!data.models.length) { setStatus("接口返回的模型列表为空", COLOR_ERR); return; }
            modelCache[pid] = data.models;
            fillMenu(data.models);
            setStatus(`已获取 ${data.models.length} 个模型，点 ▾ 选择`, COLOR_OK);
        } catch (err) {
            if (token === fetchToken) setStatus("获取失败：" + err.message, COLOR_ERR);
        } finally {
            if (token === fetchToken) setBusy(false);
        }
    };

    for (const type of ["keydown", "keyup", "keypress"]) {
        overlay.addEventListener(type, e => e.stopPropagation());
    }

    return new Promise(resolve => {
        const finish = saved => {
            fetchToken++;
            document.removeEventListener("keydown", onEsc, true);
            overlay.remove();
            resolve(saved);
        };
        const onEsc = e => { if (e.key === "Escape") { e.stopPropagation(); finish(false); } };
        document.addEventListener("keydown", onEsc, true);

        close.onclick = btnCancel.onclick = () => finish(false);

        btnSave.onclick = async () => {
            stash();
            if (cur === "custom" && !cfg.platforms.custom.baseUrl) { setStatus("请先填写接口地址", COLOR_ERR); return; }
            cfg.platform = cur;
            cfg.maxTokens = Math.max(64, parseInt(tokens.value, 10) || cfg.defaultMaxTokens || 4096);
            btnSave.disabled = true;
            try {
                const saved = await postJson("/config", {
                    platform: cfg.platform, maxTokens: cfg.maxTokens, platforms: cfg.platforms,
                });
                onSaved?.({
                    output_language: language.querySelector("input:checked")?.value || "中文",
                }, saved);
                finish(true);
            } catch (err) {
                btnSave.disabled = false;
                setStatus(err.message, COLOR_ERR);
            }
        };

        loadPlatform(cur);
        document.body.append(overlay);
    });
}
"""

# ==============================================================================
# 路由注册 (Routes)
# ==============================================================================

def _presets():
    return {pid: {"name": p["name"], "url": p["url"], "defaultModel": p["defaultModel"],
                  "keyOptional": pid in KEY_OPTIONAL} for pid, p in PLATFORMS.items()}


async def get_config(_request):
    return web.json_response({**read_config(), "presets": _presets(), "defaultMaxTokens": DEFAULT_MAX_TOKENS})


async def post_config(request):
    try:
        cfg = normalize_config(await request.json())
    except Exception:
        return web.json_response({"error": "请求内容无效"}, status=400)
    if cfg["platform"] == "custom" and not cfg["platforms"]["custom"]["baseUrl"].strip():
        return web.json_response({"error": "请先填写接口地址"}, status=400)
    if not save_config(cfg):
        return web.json_response({"error": "保存失败：无法写入配置文件"}, status=500)
    return web.json_response(cfg)


async def post_models(request):
    try:
        body = await request.json()
        models = await asyncio.to_thread(fetch_models, str(body.get("baseUrl") or ""), str(body.get("apiKey") or ""))
        return web.json_response({"models": models})
    except ApiError as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


async def get_settings_js(_request):
    return web.Response(text=SETTINGS_JS, content_type="application/javascript", charset="utf-8",
                        headers={"Cache-Control": "no-cache"})


def register_routes() -> bool:
    global _REGISTERED
    if _REGISTERED:
        return True
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return False
    routes = instance.routes
    routes.get(f"{PREFIX}/config")(get_config)
    routes.post(f"{PREFIX}/config")(post_config)
    routes.post(f"{PREFIX}/models")(post_models)
    routes.get(f"{PREFIX}/settings.js")(get_settings_js)
    _REGISTERED = True
    return True


register_routes()

__all__ = ["ApiError", "chat", "check_settings", "fetch_models",
           "PLATFORMS", "get_active_settings", "read_config", "save_config"]