import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const revalidate = 0;

const HUBSPOT_PORTAL_ID = process.env.HUBSPOT_PORTAL_ID;
const HUBSPOT_CONTACT_FORM_ID = process.env.HUBSPOT_CONTACT_FORM_ID;

function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

export async function POST(req: NextRequest) {
  try {
    const { firstname, lastname, email, company, message, pageUri, pageName } =
      (await req.json()) as {
        firstname?: string;
        lastname?: string;
        email?: string;
        company?: string;
        message?: string;
        pageUri?: string;
        pageName?: string;
      };

    if (!firstname || !firstname.trim()) {
      return NextResponse.json({ ok: false, error: "missing_firstname" }, { status: 400 });
    }
    if (!lastname || !lastname.trim()) {
      return NextResponse.json({ ok: false, error: "missing_lastname" }, { status: 400 });
    }
    if (!email || !isValidEmail(email)) {
      return NextResponse.json({ ok: false, error: "invalid_email" }, { status: 400 });
    }
    if (!message || !message.trim()) {
      return NextResponse.json({ ok: false, error: "missing_message" }, { status: 400 });
    }

    if (!HUBSPOT_PORTAL_ID || !HUBSPOT_CONTACT_FORM_ID) {
      return NextResponse.json({ ok: false, error: "not_configured" }, { status: 503 });
    }

    const endpoint = `https://api.hsforms.com/submissions/v3/integration/submit/${HUBSPOT_PORTAL_ID}/${HUBSPOT_CONTACT_FORM_ID}`;

    const fields = [
      { name: "firstname", value: firstname.trim() },
      { name: "lastname", value: lastname.trim() },
      { name: "email", value: email.trim() },
      { name: "message", value: message.trim() },
    ];

    if (company?.trim()) {
      fields.push({ name: "company", value: company.trim() });
    }

    const payload = {
      fields,
      context: {
        pageUri: pageUri || req.headers.get("referer") || "https://runmat.com/contact",
        pageName: pageName || "RunMat Contact",
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

    if (resp.ok) {
      return NextResponse.json({ ok: true }, { status: 200 });
    }

    const text = await resp.text().catch(() => "");
    return NextResponse.json(
      { ok: false, error: "hubspot_error", detail: text.slice(0, 500) },
      { status: 502 },
    );
  } catch {
    return NextResponse.json({ ok: false, error: "invalid_request" }, { status: 400 });
  }
}
