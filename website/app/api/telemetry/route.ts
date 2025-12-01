import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";
export const dynamic = "force-dynamic";
export const revalidate = 0;

const GA_MEASUREMENT_ID = process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID;
const GA_API_SECRET = process.env.GA_API_SECRET;
const POSTHOG_API_KEY = process.env.POSTHOG_API_KEY;
const POSTHOG_HOST =
  (process.env.POSTHOG_HOST && process.env.POSTHOG_HOST.trim()) ||
  "https://us.i.posthog.com";

// Expandable whitelist of event names
const ALLOWED_EVENTS = new Set([
  "install_start",
  "install_complete",
  "install_failed",
]);

// Param keys we allow to forward as-is
const ALLOWED_PARAM_KEYS = new Set([
  "os",
  "arch",
  "platform",
  "release",
  "method",
  "event_label",
]);

function sanitize(input: unknown, fallback = "unknown"): string {
  if (typeof input !== "string") return fallback;
  const trimmed = input.trim();
  if (!trimmed) return fallback;
  return trimmed.slice(0, 64).replace(/[^a-zA-Z0-9_\-\.]/g, "");
}

function uuid(): string {
  const random = crypto.getRandomValues(new Uint8Array(16));
  random[6] = (random[6] & 0x0f) | 0x40;
  random[8] = (random[8] & 0x3f) | 0x80;
  const toHex = (n: number) => n.toString(16).padStart(2, "0");
  const hex = Array.from(random, toHex).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json().catch(() => ({}))) as Record<string, unknown>;

    const event = sanitize(body.event || body.event_label);
    if (!ALLOWED_EVENTS.has(event)) {
      console.error(`Invalid event: ${event}`);
      return NextResponse.json({ ok: false, error: "invalid_event" }, { status: 400 });
    }

    const cid = sanitize(body.cid, uuid());

    // Collect allowed params
    const params: Record<string, unknown> = {};
    for (const key of ALLOWED_PARAM_KEYS) {
      if (key in body) {
        const value = body[key];
        params[key] = typeof value === "string" ? sanitize(value) : value;
      }
    }

    const forwarders: Array<Promise<Response | undefined>> = [];
    let forwardedGA = false;
    let forwardedPostHog = false;

    // Forward to Google Analytics (Measurement Protocol) if configured
    if (GA_MEASUREMENT_ID && GA_API_SECRET) {
      const gaPayload = {
        client_id: cid,
        events: [
          {
            name: event,
            params,
          },
        ],
      };
      const gaEndpoint = `https://www.google-analytics.com/mp/collect?measurement_id=${encodeURIComponent(
        GA_MEASUREMENT_ID
      )}&api_secret=${encodeURIComponent(GA_API_SECRET)}`;
      forwarders.push(
        fetch(gaEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(gaPayload),
          cache: "no-store",
        })
          .then((r) => {
            if (!r.ok) {
              console.error(`GA forwarding failed for event: ${event}`);
            } else {
              forwardedGA = true;
            }
            return r;
          })
          .catch((err) => {
            console.error(`GA forwarding error for event: ${event}`, err);
            return undefined;
          })
      );
    }

    // Forward to PostHog ingestion API if configured
    if (POSTHOG_API_KEY) {
      const posthogEndpoint = `${POSTHOG_HOST.replace(/\/$/, "")}/capture/`;
      const phPayload = {
        api_key: POSTHOG_API_KEY,
        event,
        distinct_id: cid,
        properties: params,
      };
      forwarders.push(
        fetch(posthogEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(phPayload),
          cache: "no-store",
        })
          .then((r) => {
            if (!r.ok) {
              console.error(`PostHog forwarding failed for event: ${event}`);
            } else {
              forwardedPostHog = true;
            }
            return r;
          })
          .catch((err) => {
            console.error(`PostHog forwarding error for event: ${event}`, err);
            return undefined;
          })
      );
    }

    // Execute any configured forwarders; do not fail the request
    if (forwarders.length > 0) {
      await Promise.allSettled(forwarders);
    } else {
      // Nothing configured; keep endpoint no-op but successful to caller
      return NextResponse.json({ ok: true, forwarded: { ga: false, posthog: false }, disabled: true }, { status: 200 });
    }

    return NextResponse.json(
      { ok: true, forwarded: { ga: forwardedGA, posthog: forwardedPostHog } },
      { status: 200 }
    );
  } catch {
    return NextResponse.json({ ok: true }, { status: 200 });
  }
}