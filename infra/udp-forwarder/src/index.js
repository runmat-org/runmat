import dgram from 'node:dgram';
import net from 'node:net';
import { fetch } from 'undici';

const UDP_PORT = Number(process.env.UDP_PORT || 7846);
const HTTP_ENDPOINT = process.env.TELEMETRY_HTTP_ENDPOINT || 'https://telemetry.runmat.com/ingest';
const TELEMETRY_KEY = process.env.TELEMETRY_INGESTION_KEY || '';
const CONCURRENCY = Number(process.env.FORWARDER_CONCURRENCY || 8);
const HEALTH_PORT = Number(process.env.HEALTH_PORT || 9000);

const queue = [];
let active = 0;

function enqueue(payload) {
  queue.push(payload);
  processQueue();
}

async function forward(payload) {
  const headers = { 'content-type': 'application/json' };
  if (TELEMETRY_KEY) {
    headers['x-telemetry-key'] = TELEMETRY_KEY;
  }
  try {
    await fetch(HTTP_ENDPOINT, {
      method: 'POST',
      headers,
      body: payload,
    });
  } catch (err) {
    console.error('telemetry forward failed', err);
  }
}

function processQueue() {
  while (active < CONCURRENCY && queue.length > 0) {
    const payload = queue.shift();
    active += 1;
    forward(payload)
      .catch((err) => console.error('forward error', err))
      .finally(() => {
        active -= 1;
        processQueue();
      });
  }
}

export function startForwarder() {
  const socket = dgram.createSocket('udp4');
  socket.on('message', (msg) => {
    enqueue(Buffer.from(msg));
  });
  socket.on('error', (err) => {
    console.error('udp socket error', err);
  });
  socket.bind(UDP_PORT, () => {
    console.log(`udp forwarder listening on ${UDP_PORT}, forwarding to ${HTTP_ENDPOINT}`);
  });
  const healthServer = net.createServer((conn) => {
    conn.end('ok');
  });
  healthServer.on('error', (err) => {
    console.error('health server error', err);
  });
  healthServer.listen(HEALTH_PORT, () => {
    console.log(`health server listening on ${HEALTH_PORT}`);
  });
  return { socket, healthServer };
}

if (process.env.NODE_ENV !== 'test') {
  startForwarder();
}
