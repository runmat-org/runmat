import request from 'supertest';
import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';

vi.mock('undici', () => {
  return {
    fetch: vi.fn(() => Promise.resolve({ ok: true })),
  };
});

const { fetch } = await import('undici');
const { createApp } = await import('../src/server.js');

describe('telemetry worker', () => {
  beforeEach(() => {
    process.env.INGESTION_KEY = 'secret';
    process.env.TELEMETRY_COLLECTOR_ENDPOINT = 'https://api.runmat.com/v1/t';
    fetch.mockReset();
  });

  afterEach(() => {
    delete process.env.INGESTION_KEY;
    delete process.env.TELEMETRY_COLLECTOR_ENDPOINT;
  });

  it('rejects when ingestion key mismatch', async () => {
    const app = createApp();
    const res = await request(app).post('/ingest').send({});
    expect(res.status).toBe(401);
    expect(res.body.error).toBe('unauthorized');
  });

  it('forwards valid payloads', async () => {
    fetch.mockResolvedValue({ ok: true });
    const app = createApp();
    const res = await request(app)
      .post('/ingest')
      .set('x-telemetry-key', 'secret')
      .send({
        event_label: 'runtime_finished',
        cid: 'abc123',
        arch: 'arm64',
        payload: { success: true, jit_enabled: true, accelerate_enabled: true },
      });
    expect(res.status).toBe(200);
    expect(res.body.ok).toBe(true);
    expect(fetch).toHaveBeenCalled();
    const body = JSON.parse(fetch.mock.calls[0][1].body);
    expect(body.event).toBe('runtime.run.finished');
    expect(body.distinctId).toBe('abc123');
    expect(body.arch).toBe('arm64');
    expect(body.payload.arch).toBe('arm64');
    expect(body.properties).toEqual({});
    expect(body.context).toEqual({});
  });

  it('returns forward failure when collector errors', async () => {
    fetch.mockResolvedValue({ ok: false, status: 502 });
    const app = createApp();
    const res = await request(app)
      .post('/ingest')
      .set('x-telemetry-key', 'secret')
      .send({ event_label: 'runtime_finished', cid: 'abc123' });
    expect(res.status).toBe(502);
    expect(res.body.error).toBe('forward_failed');
  });

  it('rejects payloads missing distinct id', async () => {
    fetch.mockResolvedValue({ ok: true });
    const app = createApp();
    const res = await request(app)
      .post('/ingest')
      .set('x-telemetry-key', 'secret')
      .send({ event_label: 'runtime_started', session_id: 'install-session' });
    expect(res.status).toBe(400);
    expect(res.body.error).toBe('invalid_event');
  });

  it('rejects unknown event labels', async () => {
    fetch.mockResolvedValue({ ok: true });
    const app = createApp();
    const res = await request(app)
      .post('/ingest')
      .set('x-telemetry-key', 'secret')
      .send({ event_label: 'something_else', cid: 'abc123' });
    expect(res.status).toBe(400);
    expect(res.body.error).toBe('invalid_event');
  });
});
