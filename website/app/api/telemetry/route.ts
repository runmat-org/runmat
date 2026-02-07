import { NextResponse } from 'next/server';

const ALLOWED_EVENTS = new Set(['install_start', 'install_complete', 'install_failed']);
const UPSTREAM_URL = 'https://telemetry.runmat.com/ingest';

export async function POST(request: Request) {
  const ingestionKey = process.env.TELEMETRY_INGESTION_KEY;
  if (!ingestionKey) {
    return NextResponse.json({ error: 'telemetry_disabled' }, { status: 503 });
  }

  let payload: Record<string, unknown>;
  try {
    payload = (await request.json()) ?? {};
  } catch {
    return NextResponse.json({ error: 'invalid_json' }, { status: 400 });
  }

  const eventLabel = typeof payload.event_label === 'string' ? payload.event_label : '';
  if (!ALLOWED_EVENTS.has(eventLabel)) {
    return NextResponse.json({ error: 'invalid_event' }, { status: 400 });
  }

  const forwarded = request.headers.get('x-forwarded-for') ?? request.headers.get('x-real-ip') ?? undefined;
  const proxiedPayload = {
    ...payload,
    method: payload.method ?? 'installer',
    source: 'installer-proxy',
    client_ip: forwarded,
    $ip: forwarded,
  };

  const upstreamResponse = await fetch(UPSTREAM_URL, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-telemetry-key': ingestionKey,
    },
    body: JSON.stringify(proxiedPayload),
  });

  if (!upstreamResponse.ok) {
    return NextResponse.json({ error: 'forward_failed' }, { status: upstreamResponse.status });
  }

  return NextResponse.json({ ok: true });
}
