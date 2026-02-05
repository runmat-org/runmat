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
    process.env.POSTHOG_API_KEY = 'phc_test';
    process.env.INGESTION_KEY = 'secret';
    fetch.mockReset();
  });

  afterEach(() => {
    delete process.env.POSTHOG_API_KEY;
    delete process.env.INGESTION_KEY;
  });

  it('rejects when ingestion key mismatch', async () => {
    const app = createApp();
    const res = await request(app).post('/ingest').send({});
    expect(res.status).toBe(401);
    expect(res.body.error).toBe('unauthorized');
  });

  it('rejects invalid events', async () => {
    const app = createApp();
    const res = await request(app)
      .post('/ingest')
      .set('x-telemetry-key', 'secret')
      .send({ event_label: 'nope' });
    expect(res.status).toBe(400);
    expect(res.body.error).toBe('invalid_event');
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
        session_id: 'session-123',
        run_kind: 'script',
        payload: { success: true, jit_enabled: true, accelerate_enabled: true },
      });
    expect(res.status).toBe(200);
    expect(res.body.ok).toBe(true);
    expect(fetch).toHaveBeenCalled();
    const body = JSON.parse(fetch.mock.calls[0][1].body);
    expect(body.event).toBe('runtime_finished');
    expect(body.properties.event_label).toBe('Runtime execution finished');
    expect(body.properties.summary).toContain('jit=on');
    expect(body.properties.success).toBe(true);
  });

  it('formats installer payload without runtime flags', async () => {
    fetch.mockResolvedValue({ ok: true });
    const app = createApp();
    const res = await request(app)
      .post('/ingest')
      .set('x-telemetry-key', 'secret')
      .send({
        event_label: 'install_complete',
        session_id: 'install-session',
        run_kind: 'install',
        os: 'macos',
        arch: 'arm64',
        platform: 'macos-aarch64',
        method: 'shell',
      });
    expect(res.status).toBe(200);
    const body = JSON.parse(fetch.mock.calls[0][1].body);
    expect(body.properties.summary).toContain('platform=macos-aarch64');
    expect(body.properties.summary).not.toContain('jit=');
    expect(body.properties.$current_url).toBe('runmat://install.complete?platform=macos-aarch64&method=shell&arch=arm64&status=ok');
  });

  it('prefers provided current_url when present', async () => {
    fetch.mockResolvedValue({ ok: true });
    const app = createApp();
    const res = await request(app)
      .post('/ingest')
      .set('x-telemetry-key', 'secret')
      .send({
        event_label: 'runtime_started',
        cid: 'abc123',
        session_id: 'session-xyz',
        run_kind: 'repl',
        current_url: 'https://runmat.com/sandbox?it=off',
        payload: { jit_enabled: true, accelerate_enabled: true },
      });
    expect(res.status).toBe(200);
    const body = JSON.parse(fetch.mock.calls[0][1].body);
    expect(body.properties.$current_url).toBe('https://runmat.com/sandbox?it=off');
  });
});
