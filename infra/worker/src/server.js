import express from 'express';
import morgan from 'morgan';
import crypto from 'node:crypto';
import { fetch } from 'undici';

const allowedEvents = new Set([
  'install_start',
  'install_complete',
  'install_failed',
  'runtime_session_start',
  'runtime_value',
]);

const gaAllowedKeys = [
  'os',
  'arch',
  'platform',
  'release',
  'method',
  'run_kind',
];

export function createApp() {
  const config = {
    posthogKey: process.env.POSTHOG_API_KEY || '',
    posthogHost: (process.env.POSTHOG_HOST || 'https://us.i.posthog.com').replace(/\/$/, ''),
    ingestionKey: process.env.INGESTION_KEY || '',
    gaMeasurementId: process.env.GA_MEASUREMENT_ID || '',
    gaApiSecret: process.env.GA_API_SECRET || '',
  };

  const app = express();
  app.use(morgan('tiny'));
  app.use(express.json({ limit: '1mb' }));

  app.post('/ingest', async (req, res) => {
    if (!config.posthogKey) {
      return res.status(500).json({ error: 'posthog_not_configured' });
    }

    if (config.ingestionKey) {
      const provided = req.get('x-telemetry-key');
      if (provided !== config.ingestionKey) {
        return res.status(401).json({ error: 'unauthorized' });
      }
    }

    const payload = req.body ?? {};
    const eventRaw = payload.event_label || payload.event || '';
    const event = sanitize(eventRaw, 'runtime_value');
    if (!allowedEvents.has(event)) {
      return res.status(400).json({ error: 'invalid_event' });
    }

    const cid = sanitize(payload.cid || payload.session_id || crypto.randomUUID(), crypto.randomUUID());
    const meta = filterUndefined({
      os: payload.os,
      arch: payload.arch,
      platform: payload.platform,
      release: payload.release,
      method: payload.method || 'runtime',
      run_kind: payload.run_kind,
      session_id: payload.session_id,
      cid,
      source: 'runmat-telemetry-worker',
    });

    const posthogBody = {
      api_key: config.posthogKey,
      event,
      distinct_id: cid,
      properties: {
        ...meta,
        payload: payload.payload,
      },
    };

    const tasks = [
      fetch(`${config.posthogHost}/capture/`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(posthogBody),
      }),
    ];

    if (config.gaMeasurementId && config.gaApiSecret) {
      const params = {};
      for (const key of gaAllowedKeys) {
        if (meta[key] !== undefined) {
          params[key] = typeof meta[key] === 'string' ? sanitize(meta[key]) : meta[key];
        }
      }
      const gaEndpoint = `https://www.google-analytics.com/mp/collect?measurement_id=${encodeURIComponent(
        config.gaMeasurementId,
      )}&api_secret=${encodeURIComponent(config.gaApiSecret)}`;
      tasks.push(
        fetch(gaEndpoint, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            client_id: cid,
            events: [{ name: event, params }],
          }),
        }),
      );
    }

    const results = await Promise.allSettled(tasks);
    const failure = results.find(
      (result) => result.status === 'rejected' || result.value?.ok === false,
    );
    if (failure) {
      return res.status(502).json({ error: 'forward_failed' });
    }

    return res.json({ ok: true });
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

function sanitize(input, fallback = 'unknown') {
  if (typeof input !== 'string') {
    return fallback;
  }
  const trimmed = input.trim();
  if (!trimmed) {
    return fallback;
  }
  return trimmed.slice(0, 64).replace(/[^a-zA-Z0-9_\-.]/g, '');
}

function filterUndefined(obj) {
  return Object.fromEntries(Object.entries(obj).filter(([, value]) => value !== undefined && value !== null));
}

if (process.env.NODE_ENV !== 'test') {
  const port = Number(process.env.PORT || 8080);
  const app = createApp();
  app.listen(port, () => {
    console.log(`Telemetry worker listening on ${port}`);
  });
}

