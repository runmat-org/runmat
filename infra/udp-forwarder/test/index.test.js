import dgram from 'node:dgram';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

process.env.NODE_ENV = 'test';
process.env.UDP_PORT = '0';
process.env.TELEMETRY_HTTP_ENDPOINT = 'https://example.com/ingest';
process.env.HEALTH_PORT = '0';

vi.mock('undici', () => ({
  fetch: vi.fn(() => Promise.resolve({ ok: true })),
}));

const { fetch } = await import('undici');
const { startForwarder } = await import('../src/index.js');

describe('udp forwarder', () => {
  let handles;

  beforeEach(() => {
    fetch.mockClear();
    handles = startForwarder();
  });

  afterEach(() => {
    handles?.socket?.close();
    handles?.healthServer?.close();
  });

  it('forwards UDP datagrams to HTTP endpoint', async () => {
    await new Promise((resolve) => handles.socket.once('listening', resolve));
    const port = handles.socket.address().port;
    const client = dgram.createSocket('udp4');

    await new Promise((resolve, reject) => {
      client.send(Buffer.from(JSON.stringify({ hello: 'world' })), port, '127.0.0.1', (err) => {
        client.close();
        if (err) reject(err);
        else resolve();
      });
    });

    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(fetch).toHaveBeenCalled();
  });
});

