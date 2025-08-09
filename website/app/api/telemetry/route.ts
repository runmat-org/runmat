import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

const GA_MEASUREMENT_ID = process.env.GA_MEASUREMENT_ID;
const GA_API_SECRET = process.env.GA_API_SECRET;

// Expandable whitelist of event names
const ALLOWED_EVENTS = new Set([
  "install_start",
  "install_complete",
  "install_failed",
  "copy_install_command",
  "select_os",
  "page_interaction",
  "kernel_install",
]);

// Param keys we allow to forward as-is
const ALLOWED_PARAM_KEYS = new Set([
  "os",
  "arch",
  "platform",
  "release",
  "method",
  "event_category",
  "event_label",
  "value",
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
    if (!GA_MEASUREMENT_ID || !GA_API_SECRET) {
      return NextResponse.json({ ok: true, disabled: true }, { status: 200 });
    }

    const body = (await req.json().catch(() => ({}))) as Record<string, unknown>;

    const event = sanitize(body.event);
    if (!ALLOWED_EVENTS.has(event)) {
      return NextResponse.json({ ok: false, error: "invalid_event" }, { status: 400 });
    }

    const cid = sanitize(body.cid, uuid());

    // Collect allowed params
    const params: Record<string, unknown> = { anonymize_ip: true };
    for (const key of ALLOWED_PARAM_KEYS) {
      if (key in body) {
        const value = body[key];
        params[key] = typeof value === "string" ? sanitize(value) : value;
      }
    }

    const payload = {
      client_id: cid,
      events: [
        {
          name: event,
          params,
        },
      ],
    };

    const endpoint = `https://www.google-analytics.com/mp/collect?measurement_id=${encodeURIComponent(
      GA_MEASUREMENT_ID
    )}&api_secret=${encodeURIComponent(GA_API_SECRET)}`;

    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
    }).catch(() => undefined);

    if (!resp || !resp.ok) {
      return NextResponse.json({ ok: true, forwarded: false }, { status: 200 });
    }

    return NextResponse.json({ ok: true, forwarded: true }, { status: 200 });
  } catch {
    return NextResponse.json({ ok: true }, { status: 200 });
  }
}