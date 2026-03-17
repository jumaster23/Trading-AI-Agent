"""
server.py — FastAPI + HTMX frontend para ORB AI Agent
======================================================
Corre: python server.py
Abre:  http://localhost:7860
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carga el .env desde la misma carpeta que server.py, con encoding explícito
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, encoding="utf-8", override=True)

# Verificar que la key existe
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "\n❌ GROQ_API_KEY no encontrada.\n"
        "Crea un archivo .env en esta carpeta con:\n"
        "GROQ_API_KEY=sk-tu-key\n"
    )

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import uvicorn

from agent import crear_agente, ML_AVAILABLE, DB_AVAILABLE

app   = FastAPI()
agent = crear_agente()

# ── Caché para no gastar API ──────────────────────────────────────────────────
import time, hashlib
_CACHE: dict = {}        # {hash: (timestamp, respuesta)}
_CACHE_TTL   = 300       # 5 min — misma pregunta no hace nuevo request a Groq
_REQ_LOG: list = []      # para rate limiting
_MAX_RPM     = 10        # max 10 requests/minuto

def _key(msg):
    return hashlib.md5(msg.lower().strip().encode()).hexdigest()

def _from_cache(key):
    if key in _CACHE:
        ts, resp = _CACHE[key]
        if time.time() - ts < _CACHE_TTL:
            return resp
    return None

def _rate_ok():
    now = time.time()
    _REQ_LOG[:] = [t for t in _REQ_LOG if now - t < 60]
    if len(_REQ_LOG) >= _MAX_RPM:
        return False
    _REQ_LOG.append(now)
    return True

# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ORB AI Agent</title>
<script src="https://unpkg.com/htmx.org@1.9.10/dist/htmx.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: #111318;
  color: #e8ecf2;
  font-family: 'Inter', system-ui, sans-serif;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* ── TOPBAR ── */
.topbar {
  background: #1c1f27;
  border-bottom: 1px solid #2e3340;
  padding: 14px 28px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}
.brand {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.05rem;
  font-weight: 700;
  color: #5b8af5;
  letter-spacing: .05em;
}
.sub {
  font-size: .62rem;
  color: #636b7a;
  text-transform: uppercase;
  letter-spacing: .1em;
  margin-top: 2px;
}
.badge {
  padding: 3px 10px;
  border-radius: 20px;
  font-size: .62rem;
  font-weight: 700;
  border: 1px solid;
}
.badge-green { background: rgba(52,209,122,.12); color: #34d17a; border-color: rgba(52,209,122,.3); }
.badge-blue  { background: rgba(91,138,245,.12);  color: #5b8af5;  border-color: rgba(91,138,245,.3); }
.badge-gray  { background: rgba(99,107,122,.12);  color: #636b7a;  border-color: rgba(99,107,122,.3); }

/* ── CHAT ── */
.chat-wrap {
  flex: 1;
  overflow-y: auto;
  padding: 24px 20px;
}
#chat-log {
  max-width: 900px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.msg {
  padding: 14px 18px;
  border-radius: 12px;
  font-size: .875rem;
  line-height: 1.75;
  max-width: 90%;
  border: 1px solid #2e3340;
}
.msg.user {
  background: #1c1f27;
  align-self: flex-end;
  color: #e8ecf2;
  max-width: 72%;
}
.msg.agent {
  background: #16191f;
  align-self: flex-start;
  color: #9ba3b2;
  white-space: pre-wrap;
}
.msg.agent b,
.msg.agent strong { color: #e8ecf2; }
.msg.agent code {
  background: #1c1f27;
  padding: 1px 6px;
  border-radius: 3px;
  font-family: monospace;
  font-size: .8rem;
  color: #5b8af5;
}
.msg.agent .tag-up   { color: #34d17a; font-weight: 600; }
.msg.agent .tag-down { color: #f25c6e; font-weight: 600; }

/* Thinking animation */
.thinking {
  color: #636b7a;
  font-style: italic;
}
.dot-anim::after {
  content: '';
  animation: dots 1.4s steps(3, end) infinite;
}
@keyframes dots {
  0%   { content: ''; }
  33%  { content: '.'; }
  66%  { content: '..'; }
  100% { content: '...'; }
}

/* ── INPUT BAR ── */
.input-bar {
  background: #111318;
  border-top: 1px solid #2e3340;
  padding: 14px 20px;
  flex-shrink: 0;
}

.quick-btns {
  max-width: 900px;
  margin: 0 auto 10px;
  display: flex;
  gap: 7px;
  flex-wrap: wrap;
}
.q-btn {
  background: #1c1f27;
  border: 1px solid #2e3340;
  color: #9ba3b2;
  border-radius: 16px;
  padding: 5px 13px;
  font-size: .7rem;
  cursor: pointer;
  transition: all .15s;
}
.q-btn:hover {
  border-color: #5b8af5;
  color: #e8ecf2;
}

.input-row {
  max-width: 900px;
  margin: 0 auto;
  display: flex;
  gap: 10px;
}
textarea {
  flex: 1;
  background: #1c1f27;
  border: 1px solid #363c4a;
  color: #e8ecf2;
  border-radius: 8px;
  padding: 11px 15px;
  font-size: .875rem;
  resize: none;
  font-family: inherit;
  outline: none;
  transition: border-color .2s;
}
textarea:focus { border-color: #5b8af5; }
textarea::placeholder { color: #636b7a; }

.send-btn {
  background: #5b8af5;
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 0 22px;
  font-weight: 600;
  font-size: .82rem;
  cursor: pointer;
  transition: background .15s;
  white-space: nowrap;
}
.send-btn:hover    { background: #4a79e8; }
.send-btn:disabled { opacity: .5; cursor: not-allowed; }

/* HTMX loading indicator */
#loader {
  display: none;
  height: 2px;
  background: linear-gradient(90deg, #5b8af5, #a78bfa, #5b8af5);
  background-size: 200%;
  animation: shimmer 1.2s linear infinite;
}
.htmx-request #loader { display: block; }
@keyframes shimmer {
  0%   { background-position: 100%; }
  100% { background-position: 0%; }
}
</style>
</head>
<body>

<div id="loader"></div>

<div class="topbar">
  <div>
    <div class="brand">ORB AI Agent</div>
    <div class="sub">FastAPI · HTMX · phidata · GPT-4o-mini</div>
  </div>
  <span class="badge badge-green">● Live</span>
  <span class="badge {ml_cls}">🧠 ML {ml_txt}</span>
  <span class="badge {db_cls}">🗄 DB {db_txt}</span>
</div>

<div class="chat-wrap">
  <div id="chat-log">
    <div class="msg agent">
👋 <b>ORB Trading AI Agent</b> — conectado a datos reales de mercado.

Prueba estos comandos:
  • <code>Dame un plan para SPY hoy</code>
  • <code>¿Se llenará el gap de QQQ?</code>
  • <code>Análisis ML de AAPL</code>
  • <code>Compara SPY y QQQ</code>
  • <code>Noticias de NVDA</code>
  • <code>Stats de la base de datos para SPY</code>
    </div>
  </div>
</div>

<div class="input-bar">
  <div class="quick-btns">
    <button class="q-btn" onclick="ask('Dame un plan completo para SPY hoy')">📊 Plan SPY</button>
    <button class="q-btn" onclick="ask('Análisis ML de QQQ')">🧠 ML QQQ</button>
    <button class="q-btn" onclick="ask('¿Cuál es el gap de SPY hoy y se llenará?')">📈 Gap SPY</button>
    <button class="q-btn" onclick="ask('Estadísticas ORB de QQQ')">⏱ ORB QQQ</button>
    <button class="q-btn" onclick="ask('Compara SPY y QQQ')">⚖️ Comparar</button>
    <button class="q-btn" onclick="ask('Noticias de AAPL')">📰 AAPL</button>
    <button class="q-btn" onclick="ask('Stats de la base de datos para SPY')">🗄 DB SPY</button>
  </div>

  <div class="input-row">
    <textarea
      id="inp"
      rows="2"
      placeholder="Pregunta sobre cualquier ticker: SPY, QQQ, AAPL, NVDA..."
      onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();enviar()}"
    ></textarea>
    <button class="send-btn" id="btn" onclick="enviar()">Enviar ↵</button>
  </div>
</div>

<script>
function ask(q) {
  document.getElementById('inp').value = q;
  enviar();
}

function enviar() {
  const inp = document.getElementById('inp');
  const msg = inp.value.trim();
  if (!msg) return;
  inp.value = '';

  const log = document.getElementById('chat-log');

  // Mensaje del usuario
  log.innerHTML += `<div class="msg user">${esc(msg)}</div>`;

  // Indicador de carga
  const tid = 'think_' + Date.now();
  log.innerHTML += `
    <div class="msg agent thinking" id="${tid}">
      🤖 Analizando con datos reales<span class="dot-anim"></span>
    </div>`;
  bajar();

  document.getElementById('btn').disabled = true;

  // POST al endpoint FastAPI
  fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: 'message=' + encodeURIComponent(msg)
  })
  .then(r => r.text())
  .then(html => {
    document.getElementById(tid)?.remove();
    log.innerHTML += html;
    bajar();
  })
  .catch(err => {
    const el = document.getElementById(tid);
    if (el) el.textContent = '❌ Error: ' + err;
  })
  .finally(() => {
    document.getElementById('btn').disabled = false;
  });
}

function bajar() {
  const w = document.querySelector('.chat-wrap');
  w.scrollTop = w.scrollHeight;
}

function esc(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
</script>
</body>
</html>
"""

def build_html() -> str:
    ml_cls = "badge-blue"  if ML_AVAILABLE else "badge-gray"
    ml_txt = "✅ activo"   if ML_AVAILABLE else "❌ off"
    db_cls = "badge-blue"  if DB_AVAILABLE else "badge-gray"
    db_txt = "✅ activo"   if DB_AVAILABLE else "❌ off"
    return HTML.replace("{ml_cls}", ml_cls).replace("{ml_txt}", ml_txt) \
               .replace("{db_cls}", db_cls).replace("{db_txt}", db_txt)


def md_to_html(text: str) -> str:
    import re
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`",        r"<code>\1</code>", text)
    text = re.sub(r"^#{1,3} (.+)$",  r"<b>\1</b>", text, flags=re.MULTILINE)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS FASTAPI
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(build_html())


@app.post("/chat", response_class=HTMLResponse)
async def chat(message: str = Form(...)):
    k = _key(message)

    # 1. Revisar caché
    cached = _from_cache(k)
    if cached:
        return HTMLResponse(f'<div class="msg agent">{cached} <small style="color:#636b7a">💾 caché</small></div>')

    # 2. Rate limit
    if not _rate_ok():
        return HTMLResponse('<div class="msg agent">⏳ Demasiadas preguntas seguidas, espera 1 minuto.</div>')

    # 3. Llamar al agente
    try:
        run = agent.run(message, stream=False)

        # Extraer texto limpio de la respuesta de phidata
        if hasattr(run, "content") and run.content:
            text = run.content
        elif hasattr(run, "messages") and run.messages:
            # Buscar el último mensaje del asistente
            for m in reversed(run.messages):
                role = getattr(m, "role", "")
                content = getattr(m, "content", "")
                if role == "assistant" and content:
                    text = content
                    break
            else:
                text = str(run)
        else:
            text = str(run)

        # Limpiar artefactos de tool calls que se cuelan en la respuesta
        import re
        text = re.sub(r'<function:[^>]+>[^<]*</function[^>]*>', '', text)
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
        text = re.sub(r'\{[^}]+\}', '', text)
        text = text.strip()

        html = md_to_html(text)
        _CACHE[k] = (time.time(), html)
    except Exception as e:
        html = f"❌ Error: {e}"

    return HTMLResponse(f'<div class="msg agent">{html}</div>')


@app.get("/health")
async def health():
    return {
        "status":     "ok",
        "ml_engine":  ML_AVAILABLE,
        "trading_db": DB_AVAILABLE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    print(f"\n{'='*52}")
    print(f"  ORB AI Agent — FastAPI + HTMX")
    print(f"  http://localhost:{port}")
    print(f"  ml_engine.py : {'✅ activo' if ML_AVAILABLE else '❌ no encontrado'}")
    print(f"  trading_db.py: {'✅ activo' if DB_AVAILABLE else '❌ no encontrado'}")
    print(f"{'='*52}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")