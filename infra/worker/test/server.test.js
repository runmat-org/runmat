import request from 'supertest';
import { describe, expect, it, beforeEach, vi } from 'vitest';

vi.mock('undici', () => {
  return {
    fetch: vi.fn(() => Promise.resolve({ ok: true })),
  };
});

const { fetch } = await import('undici');
const { createApp } = await import('../src/server.js');

describe('telemetry worker', () => {
  beforeEach(() => {
    fetch.mockReset();
  });

  it('rejects when ingestion key mismatch', async () => {
    process.env.INGESTION_KEY = 'secret';
    const app = createApp();
    const res = await request(app).post('/ingest').send({});
    expect(res.status).toBe(401);
    expect(res.body.error).toBe('unauthorized');
  });

  it('rejects invalid events', async () => {
    process.env.INGESTION_KEY = '';
    const app = createApp();
    const res = await request(app).post('/ingest').send({ event_label: 'nope' });
    expect(res.status).toBe(400);
    expect(res.body.error).toBe('invalid_event');
  });

  it('forwards valid payloads', async () => {
    process.env.INGESTION_KEY = '';
    fetch.mockResolvedValue({ ok: true });
    const app = createApp();
    const res = await request(app)
      .post('/ingest')
      .send({
        event_label: 'runtime_value',
        cid: 'abc123',
        payload: { ok: true },
      });
    expect(res.status).toBe(200);
    expect(res.body.ok).toBe(true);
    expect(fetch).toHaveBeenCalled();
  });
});