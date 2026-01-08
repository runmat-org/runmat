import { NextRequest, NextResponse } from "next/server";

// Node runtime to avoid Edge CORS peculiarities and allow future secret usage
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const revalidate = 0;

const HUBSPOT_PORTAL_ID = process.env.HUBSPOT_PORTAL_ID;
const HUBSPOT_NEWSLETTER_FORM_ID = process.env.HUBSPOT_NEWSLETTER_FORM_ID;
const HUBSPOT_DESKTOP_FORM_ID = process.env.HUBSPOT_DESKTOP_FORM_ID;

function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

export async function POST(req: NextRequest) {
  try {
    const { email, pageUri, pageName, source } = (await req.json()) as {
      email?: string;
      pageUri?: string;
      pageName?: string;
      source?: string;
    };

    if (!email || !isValidEmail(email)) {
      return NextResponse.json({ ok: false, error: "invalid_email" }, { status: 400 });
    }

    if (!HUBSPOT_PORTAL_ID || !HUBSPOT_NEWSLETTER_FORM_ID || !HUBSPOT_DESKTOP_FORM_ID) {
      return NextResponse.json({ ok: false, error: "not_configured" }, { status: 503 });
    }
    const formId = source === "desktop" ? HUBSPOT_DESKTOP_FORM_ID : HUBSPOT_NEWSLETTER_FORM_ID;
    const endpoint = `https://api.hsforms.com/submissions/v3/integration/submit/${HUBSPOT_PORTAL_ID}/${formId}`;

    const payload = {
      fields: [{ name: "email", value: email }],
      context: {
        pageUri: pageUri || req.headers.get("referer") || "https://runmat.org",
        pageName: pageName || "RunMat Subscribe",
      },
    };

    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).catch(() => undefined);

    if (!resp) {
      return NextResponse.json({ ok: false, error: "network_error" }, { status: 502 });
    }

    // HubSpot returns 200 OK or 204 No Content on success; treat other codes as soft-fail
    if (resp.ok) {
      return NextResponse.json({ ok: true }, { status: 200 });
    }

    const text = await resp.text().catch(() => "");
    return NextResponse.json({ ok: false, error: "hubspot_error", detail: text.slice(0, 500) }, { status: 502 });
  } catch {
    return NextResponse.json({ ok: false, error: "invalid_request" }, { status: 400 });
  }
}


