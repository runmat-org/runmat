const allowedEvents = new Set([
  "install_start",
  "install_complete",
  "install_failed",
  "runtime_session_start",
  "runtime_value",
]);

const gaAllowedKeys = [
  "os",
  "arch",
  "platform",
  "release",
  "method",
  "run_kind",
];

export default {
  async fetch(request, env) {
    if (request.method !== "POST") {
      return new Response("method not allowed", { status: 405 });
    }

    const url = new URL(request.url);
    if (url.pathname !== "/ingest") {
      return new Response("not found", { status: 404 });
    }

    if (env.INGESTION_KEY) {
      const provided = request.headers.get("x-telemetry-key");
      if (provided !== env.INGESTION_KEY) {
        return new Response("unauthorized", { status: 401 });
      }
    }

    let payload;
    try {
      payload = await request.json();
    } catch {
      return new Response("invalid json", { status: 400 });
    }

    const eventRaw = payload.event_label || payload.event || "";
    const event = sanitize(eventRaw, "runtime_value");
    if (!allowedEvents.has(event)) {
      return new Response("invalid event", { status: 400 });
    }

    const cid = sanitize(
      payload.cid || payload.session_id || crypto.randomUUID(),
      crypto.randomUUID(),
    );

    const meta = filterUndefined({
      os: payload.os,
      arch: payload.arch,
      platform: payload.platform,
      release: payload.release,
      method: payload.method || "runtime",
      run_kind: payload.run_kind,
      session_id: payload.session_id,
      cid,
      source: "runmat-telemetry-worker",
    });

    const posthogProperties = {
      ...meta,
      payload: payload.payload,
    };

    const tasks = [];

    if (!env.POSTHOG_API_KEY) {
      return new Response("missing POSTHOG_API_KEY", { status: 500 });
    }

    const posthogHost = (env.POSTHOG_HOST || "https://us.i.posthog.com").replace(
      /\/$/,
      "",
    );
    tasks.push(
      fetch(`${posthogHost}/capture/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          api_key: env.POSTHOG_API_KEY,
          event,
          distinct_id: cid,
          properties: posthogProperties,
        }),
      }),
    );

    if (env.GA_MEASUREMENT_ID && env.GA_API_SECRET) {
      const params = {};
      for (const key of gaAllowedKeys) {
        if (meta[key] !== undefined) {
          params[key] =
            typeof meta[key] === "string" ? sanitize(meta[key]) : meta[key];
        }
      }
      const gaEndpoint = `https://www.google-analytics.com/mp/collect?measurement_id=${encodeURIComponent(
        env.GA_MEASUREMENT_ID,
      )}&api_secret=${encodeURIComponent(env.GA_API_SECRET)}`;
      tasks.push(
        fetch(gaEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            client_id: cid,
            events: [{ name: event, params }],
          }),
        }),
      );
    }

    const results = await Promise.allSettled(tasks);
    const failed = results.find(
      (result) => result.status === "rejected" || result.value?.ok === false,
    );
    if (failed) {
      return new Response("telemetry forwarding failed", { status: 502 });
    }

    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  },
};

function sanitize(input, fallback = "unknown") {
  if (typeof input !== "string") {
    return fallback;
  }
  const trimmed = input.trim();
  if (!trimmed) {
    return fallback;
  }
  return trimmed.slice(0, 64).replace(/[^a-zA-Z0-9_\-.]/g, "");
}

function filterUndefined(obj) {
  return Object.fromEntries(
    Object.entries(obj).filter(([, value]) => value !== undefined),
  );
}