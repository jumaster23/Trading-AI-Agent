import Fastify from 'fastify';
import formbody from '@fastify/formbody';
import dotenv from 'dotenv';
import crypto from 'crypto';
import { z } from 'zod';

dotenv.config();

const fastify = Fastify({ logger: false });
fastify.register(formbody);

// --- Configuración y Tipos ---
const GROQ_KEY = process.env.GROQ_API_KEY;
if (!GROQ_KEY) throw new Error("❌ GROQ_API_KEY no encontrada.");

interface CacheEntry {
    timestamp: number;
    html: string;
}

const _CACHE = new Map<string, CacheEntry>();
const _CACHE_TTL = 300 * 1000; // 5 min en ms
const _REQ_LOG: number[] = [];
const _MAX_RPM = 10;

// --- Helpers ---
const getHash = (msg: string) => 
    crypto.createHash('md5').update(msg.toLowerCase().trim()).digest('hex');

const isRateOk = (): boolean => {
    const now = Date.now();
    // Limpiar logs antiguos
    while (_REQ_LOG.length > 0 && now - _REQ_LOG[0] > 60000) {
        _REQ_LOG.shift();
    }
    if (_REQ_LOG.length >= _MAX_RPM) return false;
    _REQ_LOG.push(now);
    return true;
};

const mdToHtml = (text: string): string => {
    return text
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
        .replace(/`(.+?)`/g, "<code>$1</code>");
};

const renderAgentMsg = (content: string, label = "ORB Agent") => `
    <div class="msg-wrap agent">
        <div class="msg-label">${label}</div>
        <div class="msg agent">${content}</div>
    </div>
`;

// --- Rutas ---

fastify.get('/', async (request, reply) => {
    reply.type('text/html').send(``);
});

fastify.post('/chat', async (request, reply) => {
    const body = request.body as { message?: string };
    const message = body.message || '';
    const key = getHash(message);

    // 1. Cache
    const cached = _CACHE.get(key);
    if (cached && (Date.now() - cached.timestamp) < _CACHE_TTL) {
        return reply.type('text/html').send(renderAgentMsg(cached.html, "ORB Agent · caché"));
    }

    // 2. Rate Limit
    if (!isRateOk()) {
        return reply.type('text/html').send(renderAgentMsg("⏳ Demasiadas preguntas, espera 1 minuto."));
    }

    // 3. Simulación de Agente
    try {
        const responseText = `Procesando niveles de trading para **${message}**...`;
        const html = mdToHtml(responseText);

        _CACHE.set(key, { timestamp: Date.now(), html });
        return reply.type('text/html').send(renderAgentMsg(html));
    } catch (e) {
        return reply.type('text/html').send(renderAgentMsg(`❌ Error: ${e}`));
    }
});

// --- Inicio ---
const start = async () => {
    try {
        const port = Number(process.env.PORT) || 7860;
        await fastify.listen({ port, host: '0.0.0.0' });
        console.log(`\n🚀 ORB Agent (TS) corriendo en http://localhost:${port}\n`);
    } catch (err) {
        fastify.log.error(err);
        process.exit(1);
    }
};

start();