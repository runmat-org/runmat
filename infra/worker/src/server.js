import express from 'express';
import morgan from 'morgan';
import crypto from 'node:crypto';
import { fetch } from 'undici';

const allowedEvents = new Set(['install_start', 'install_complete', 'install_failed', 'runtime_started', 'runtime_finished']);
const allowedRunKinds = new Set(['script', 'repl', 'benchmark']);
const gaAllowedKeys = ['os', 'arch', 'platform', 'release', 'method', 'run_kind'];
const MAX_DETAIL_BYTES = 16 * 1024;

export function createApp() {
  const config = {
    posthogKey: process.env.POSTHOG_API_KEY || '',
    posthogHost: (process.env.POSTHOG_HOST || 'https://us.i.posthog.com').replace(/\/$/, ''),
    ingestionKey: process.env.INGESTION_KEY || '',
    gaMeasurementId: process.env.GA_MEASUREMENT_ID || '',
    gaApiSecret: process.env.GA_API_SECRET || '',
  };

  const app = express();
  app.set('trust proxy', true);
  app.use(morgan('tiny'));
  app.use(express.json({ limit: '1mb' }));

  app.post('/ingest', async (req, res) => {
    if (!config.ingestionKey) {
      console.error('telemetry worker misconfigured: missing INGESTION_KEY');
      return res.status(503).json({ error: 'telemetry_disabled' });
    }
    if (!config.posthogKey) {
      return res.status(500).json({ error: 'posthog_not_configured' });
    }

    const provided = req.get('x-telemetry-key');
    if (!timingSafeEqual(provided, config.ingestionKey)) {
      return res.status(401).json({ error: 'unauthorized' });
    }

    const payload = req.body ?? {};
    const eventRaw = payload.event_label || payload.event || '';
    const event = sanitize(eventRaw, 'runtime_finished');
    if (!allowedEvents.has(event)) {
      return res.status(400).json({ error: 'invalid_event' });
    }

    if (!payload.session_id || typeof payload.session_id !== 'string') {
      return res.status(400).json({ error: 'missing_session' });
    }
    if (!payload.run_kind || typeof payload.run_kind !== 'string' || !allowedRunKinds.has(payload.run_kind)) {
      return res.status(400).json({ error: 'invalid_run_kind' });
    }

    const sessionId = sanitize(payload.session_id, null);
    const runKind = sanitize(payload.run_kind, 'script');
    const cid = deriveCid(payload.cid, sessionId);
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

    const detail = normalizeDetail(payload.payload);
    const successFlag = coerceBoolean(detail.success ?? payload.success);
    const summary = summarizeEvent(event, runKind, detail, successFlag);
    const currentUrl = buildSyntheticUrl(event, runKind, detail, successFlag);

    const eventName = friendlyEventName(event);
    const friendlyLabel = friendlyEventLabel(event);
    const clientIp = req.ip || req.headers['x-forwarded-for'];
    const posthogBody = {
      api_key: config.posthogKey,
      event: eventName,
      distinct_id: cid,
      properties: {
        ...meta,
        event_label: friendlyLabel,
        client_ip: clientIp,
        $ip: clientIp,
        gpu_wall_ns: detail.gpu_wall_ns ?? null,
        gpu_ratio: detail.gpu_ratio ?? null,
        gpu_dispatches: detail.gpu_dispatches ?? null,
        gpu_upload_bytes: detail.gpu_upload_bytes ?? null,
        gpu_download_bytes: detail.gpu_download_bytes ?? null,
        fusion_cache_hits: detail.fusion_cache_hits ?? null,
        fusion_cache_misses: detail.fusion_cache_misses ?? null,
        summary,
        status: successFlag === false ? 'failed' : 'ok',
        jit_enabled: coerceBoolean(detail.jit_enabled),
        jit_used: coerceBoolean(detail.jit_used),
        accelerate_enabled: coerceBoolean(detail.accelerate_enabled),
        gpu_used: Boolean(detail.gpu_dispatches || detail.gpu_ratio || detail.provider),
        success: successFlag,
        $current_url: currentUrl,
        payload: detail,
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

function deriveCid(rawCid, sessionId) {
  if (typeof rawCid === 'string' && rawCid.trim()) {
    return sanitize(rawCid, sessionId || crypto.randomUUID());
  }
  if (sessionId) {
    return crypto.createHash('sha256').update(sessionId).digest('hex');
  }
  return crypto.randomUUID();
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function normalizeDetail(detail) {
  if (!isPlainObject(detail)) {
    return {};
  }
  const serialized = JSON.stringify(detail);
  if (Buffer.byteLength(serialized) > MAX_DETAIL_BYTES) {
    return { truncated: true };
  }
  return detail;
}

function coerceBoolean(value) {
  if (value === true) return true;
  if (value === false) return false;
  return undefined;
}

function summarizeEvent(event, runKind, detail, successFlag) {
  const jitEnabled = coerceBoolean(detail.jit_enabled);
  const jitUsed = coerceBoolean(detail.jit_used);
  const accel = coerceBoolean(detail.accelerate_enabled);
  const gpuUsed = detail.gpu_dispatches || detail.gpu_ratio || (detail.provider && Object.keys(detail.provider).length > 0);
  const pieces = [
    `event=${event}`,
    `run=${runKind}`,
    `jit=${jitEnabled ? (jitUsed ? 'used' : 'on') : 'off'}`,
    `accel=${accel ? 'on' : 'off'}`,
    `gpu=${gpuUsed ? 'on' : 'off'}`,
    `status=${successFlag === false ? 'fail' : 'ok'}`,
  ];
  return pieces.join(' • ');
}

function buildSyntheticUrl(event, runKind, detail, successFlag) {
  const params = new URLSearchParams({
    jit: coerceBoolean(detail.jit_enabled) ? 'on' : 'off',
    accel: coerceBoolean(detail.accelerate_enabled) ? 'on' : 'off',
    gpu: detail.gpu_dispatches || detail.gpu_ratio ? 'on' : 'off',
    status: successFlag === false ? 'fail' : 'ok',
  });
  return `runmat://run.${runKind}?${params.toString()}`;
}

function friendlyEventName(event) {
  switch (event) {
    case 'runtime_started':
      return 'runtime_started';
    case 'runtime_finished':
      return 'runtime_finished';
    default:
      return event;
  }
}

function friendlyEventLabel(event) {
  switch (event) {
    case 'runtime_started':
      return 'Runtime execution start';
    case 'runtime_finished':
      return 'Runtime execution finished';
    case 'install_start':
      return 'Installer start';
    case 'install_complete':
      return 'Installer complete';
    case 'install_failed':
      return 'Installer failed';
    default:
      return event.replace(/_/g, ' ');
  }
}

if (process.env.NODE_ENV !== 'test') {
  const port = Number(process.env.PORT || 8080);
  const app = createApp();
  app.listen(port, () => {
    console.log(`Telemetry worker listening on ${port}`);
  });
}

