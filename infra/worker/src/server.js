import express from 'express';
import morgan from 'morgan';
import crypto from 'node:crypto';
import { fetch } from 'undici';

const DEFAULT_COLLECTOR_ENDPOINT = 'https://api.runmat.com/v1/t';

export function createApp() {
  const config = {
    ingressKey: process.env.INGESTION_KEY || '',
    collectorEndpoint: process.env.TELEMETRY_COLLECTOR_ENDPOINT || DEFAULT_COLLECTOR_ENDPOINT,
    collectorKey: process.env.TELEMETRY_COLLECTOR_KEY || '',
  };

  const app = express();
  app.set('trust proxy', true);
  app.use(morgan('tiny'));
  app.use(express.json({ limit: '1mb' }));

  app.post('/ingest', async (req, res) => {
    if (config.ingressKey) {
      const provided = req.get('x-telemetry-key');
      if (!timingSafeEqual(provided, config.ingressKey)) {
        return res.status(401).json({ error: 'unauthorized' });
      }
    }

    const payload = req.body ?? {};
    const normalized = normalizeLegacyPayload(payload);
    if (!normalized) {
      return res.status(400).json({ error: 'invalid_event' });
    }
    const headers = {
      'content-type': 'application/json',
    };
    if (config.collectorKey) {
      headers['x-telemetry-key'] = config.collectorKey;
    }

    try {
      const upstream = await fetch(config.collectorEndpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify(normalized),
      });
      if (!upstream.ok) {
        return res.status(upstream.status).json({ error: 'forward_failed' });
      }
      return res.json({ ok: true });
    } catch {
      return res.status(502).json({ error: 'forward_failed' });
    }
  });

  app.use((req, res) => {
    res.status(404).send('not found');
  });

  // eslint-disable-next-line no-unused-vars
  app.use((err, req, res, _next) => {
    if (err?.type === 'entity.too.large') {
      return res.status(413).json({ error: 'payload_too_large' });
    }
    if (err instanceof SyntaxError) {
      return res.status(400).json({ error: 'invalid_json' });
    }
    return res.status(500).json({ error: 'internal_error' });
  });

  return app;
}

function normalizeLegacyPayload(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return null;
  }
  const eventRaw =
    typeof payload.event === 'string'
      ? payload.event
      : typeof payload.event_label === 'string'
        ? payload.event_label
        : '';
  const event = mapLegacyEvent(eventRaw);
  if (!event) {
    return null;
  }
  const distinctId = typeof payload.cid === 'string' ? payload.cid : undefined;
  if (!distinctId) {
    return null;
  }
  const arch = typeof payload.arch === 'string' ? payload.arch : undefined;
  return {
    event,
    distinctId,
    uuid: typeof payload.uuid === 'string' ? payload.uuid : crypto.randomUUID(),
    arch,
    payload: arch ? { arch } : {},
    properties: {},
    context: {},
    source: 'telemetry-compat-worker',
  };
}

function mapLegacyEvent(value) {
  switch ((value || '').trim()) {
    case 'runtime_started':
      return 'runtime.run.started';
    case 'runtime_finished':
      return 'runtime.run.finished';
    case 'runtime.run.started':
      return 'runtime.run.started';
    case 'runtime.run.finished':
      return 'runtime.run.finished';
    default:
      return null;
  }
}

function timingSafeEqual(provided, expected) {
  if (!provided || typeof provided !== 'string') {
    return false;
  }
  const providedBuf = Buffer.from(provided);
  const expectedBuf = Buffer.from(expected || '');
  if (providedBuf.length !== expectedBuf.length) {
    return false;
  }
  return crypto.timingSafeEqual(providedBuf, expectedBuf);
}

if (process.env.NODE_ENV !== 'test') {
  const port = Number(process.env.PORT || 8080);
  const app = createApp();
  app.listen(port, () => {
    console.log(`Telemetry worker listening on ${port}`);
  });
}
