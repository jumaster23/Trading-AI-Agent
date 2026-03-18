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
from live import router as live_router
from crystal import router as crystal_router

app   = FastAPI()
agent = crear_agente()
app.include_router(live_router)
app.include_router(crystal_router)

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
  background: #f8f7f5;
  color: #1a1a1a;
  font-family: -apple-system, BlinkMacSystemFont, 'Inter', system-ui, sans-serif;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* ── TOPBAR ── */
.topbar {
  background: #ffffff;
  border-bottom: 0.5px solid #e5e3df;
  padding: 12px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}
.logo {
  width: 32px; height: 32px;
  border-radius: 8px;
  background: linear-gradient(135deg, #f97316, #ea580c);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.brand { font-size: 14px; font-weight: 500; color: #1a1a1a; }
.sub   { font-size: 11px; color: #888; margin-top: 1px; }
.badge {
  font-size: 11px; padding: 3px 10px;
  border-radius: 20px; font-weight: 500;
}
.badge-green { background: #dcfce7; color: #15803d; }
.badge-yellow{ background: #fef9c3; color: #854d0e; }
.badge-blue  { background: #e0f2fe; color: #0369a1; }
.badge-gray  { background: #f1f1f1; color: #888; }

/* ── CHAT ── */
.chat-wrap {
  flex: 1; overflow-y: auto;
  padding: 20px 16px;
}
#chat-log {
  max-width: 820px; margin: 0 auto;
  display: flex; flex-direction: column; gap: 14px;
}
.msg-wrap { display: flex; flex-direction: column; gap: 3px; }
.msg-wrap.user { align-items: flex-end; }
.msg-wrap.agent { align-items: flex-start; }
.msg-label { font-size: 11px; color: #aaa; padding: 0 4px; }

.msg {
  padding: 11px 15px;
  font-size: 13px; line-height: 1.7;
  max-width: 82%;
  white-space: pre-wrap;
}
.msg.user {
  background: linear-gradient(135deg, #f97316, #ea580c);
  color: #fff;
  border-radius: 12px 0 12px 12px;
}
.msg.agent {
  background: #ffffff;
  color: #1a1a1a;
  border-radius: 0 12px 12px 12px;
  border: 0.5px solid #e5e3df;
}
.msg.agent b, .msg.agent strong { color: #1a1a1a; font-weight: 500; }
.msg.agent code {
  background: #f3f2ef; padding: 1px 6px;
  border-radius: 4px; font-family: monospace;
  font-size: .8rem; color: #ea580c;
}

.thinking { color: #aaa; font-style: italic; }
.dot-anim::after {
  content: '';
  animation: dots 1.4s steps(3,end) infinite;
}
@keyframes dots {
  0%  { content: ''; }
  33% { content: '.'; }
  66% { content: '..'; }
  100%{ content: '...'; }
}

/* ── INPUT BAR ── */
.input-bar {
  background: #ffffff;
  border-top: 0.5px solid #e5e3df;
  padding: 12px 16px;
  flex-shrink: 0;
}
.quick-btns {
  max-width: 820px; margin: 0 auto 10px;
  display: flex; gap: 6px; flex-wrap: wrap;
}
.q-btn {
  font-size: 11px; padding: 5px 12px;
  border-radius: 20px;
  border: 0.5px solid #e5e3df;
  background: #f8f7f5; color: #666;
  cursor: pointer; transition: all .15s;
}
.q-btn:hover { border-color: #f97316; color: #ea580c; background: #fff7ed; }

.input-row {
  max-width: 820px; margin: 0 auto;
  display: flex; gap: 8px; align-items: flex-end;
}
textarea {
  flex: 1;
  background: #f8f7f5;
  border: 0.5px solid #e5e3df;
  color: #1a1a1a;
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 13px; resize: none;
  font-family: inherit; outline: none;
  transition: border-color .2s;
  line-height: 1.5;
}
textarea:focus { border-color: #f97316; background: #fff; }
textarea::placeholder { color: #bbb; }

.send-btn {
  width: 38px; height: 38px;
  border-radius: 10px;
  background: linear-gradient(135deg, #f97316, #ea580c);
  border: none; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; transition: opacity .15s;
}
.send-btn:hover   { opacity: .85; }
.send-btn:disabled{ opacity: .4; cursor: not-allowed; }

#loader {
  display: none; height: 2px;
  background: linear-gradient(90deg, #f97316, #ea580c, #f97316);
  background-size: 200%;
  animation: shimmer 1.2s linear infinite;
}
.htmx-request #loader { display: block; }
@keyframes shimmer {
  0%  { background-position: 100%; }
  100%{ background-position: 0%; }
}
</style>
</head>
<body>

<div id="loader"></div>

<div class="topbar">
  <div class="logo">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
  </div>
  <div>
    <div class="brand">ORB AI Agent</div>
    <div class="sub">Groq · LLaMA 3.3 · ml_engine · trading_db</div>
  </div>
  <div style="margin-left:auto;display:flex;gap:8px">
  <a href="/live" style="font-size:11px;padding:5px 12px;border-radius:16px;background:rgba(249,115,22,.12);color:#f97316;border:1px solid rgba(249,115,22,.3);text-decoration:none;font-weight:500">⚡ Live</a>
  <a href="/crystal" style="font-size:11px;padding:5px 12px;border-radius:16px;background:rgba(0,212,255,.08);color:#00d4ff;border:1px solid rgba(0,212,255,.2);text-decoration:none;font-weight:500">💎 Crystal</a>
</div>
  <span class="badge {ml_cls}">ML {ml_txt}</span>
  <span class="badge {db_cls}">DB {db_txt}</span>
</div>

<div class="chat-wrap">
  <div id="chat-log">
    <div class="msg-wrap agent">
      <div class="msg-label">ORB Agent</div>
      <div class="msg agent">Hey! Pregúntame sobre cualquier ticker — analizo gaps, ORB, ML y genero planes de trading completos.<br><br>Prueba: <code>Dame un plan para SPY hoy</code></div>
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
    <button class="send-btn" id="btn" onclick="enviar()">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M22 2L11 13M22 2L15 22l-4-9-9-4 20-7z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </button>
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
  log.innerHTML += `<div class="msg-wrap user"><div class="msg-label">Tú</div><div class="msg user">${esc(msg)}</div></div>`;

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
    ml_txt = "on" if ML_AVAILABLE else "off"
    db_cls = "badge-blue"  if DB_AVAILABLE else "badge-gray"
    db_txt = "on" if DB_AVAILABLE else "off"
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
        return HTMLResponse(f'<div class="msg-wrap agent"><div class="msg-label">ORB Agent · caché</div><div class="msg agent">{cached}</div></div>')

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

    return HTMLResponse(f'<div class="msg-wrap agent"><div class="msg-label">ORB Agent</div><div class="msg agent">{html}</div></div>')


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