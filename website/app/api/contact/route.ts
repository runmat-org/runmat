import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const revalidate = 0;

const HUBSPOT_PORTAL_ID = process.env.HUBSPOT_PORTAL_ID;
const HUBSPOT_CONTACT_FORM_ID = process.env.HUBSPOT_CONTACT_FORM_ID;
const RUNMAT_SERVER_API_BASE_URL = process.env.RUNMAT_SERVER_API_BASE_URL || "https://api.runmat.com";

type AttributionPayload = {
  utmSource?: string;
  utmMedium?: string;
  utmCampaign?: string;
  utmTerm?: string;
  utmContent?: string;
  gclid?: string;
  gbraid?: string;
  wbraid?: string;
  msclkid?: string;
  fbclid?: string;
  ttclid?: string;
  liFatId?: string;
  landingPageUrl?: string;
  pageReferrer?: string;
  capturedAt?: string;
  gaClientId?: string;
};

function isValidEmail(email: string): boolean {
  const trimmed = email.trim();
  if (!trimmed || trimmed.length > 320) {
    return false;
  }

  let atIndex = -1;
  for (let i = 0; i < trimmed.length; i += 1) {
    const code = trimmed.charCodeAt(i);
    if (code <= 32 || code === 127) {
      return false;
    }
    if (trimmed[i] === "@") {
      if (atIndex !== -1) {
        return false;
      }
      atIndex = i;
    }
  }

  if (atIndex <= 0 || atIndex === trimmed.length - 1) {
    return false;
  }

  const localPartLength = atIndex;
  const domain = trimmed.slice(atIndex + 1);
  if (localPartLength > 64 || domain.length > 255) {
    return false;
  }
  if (domain.startsWith(".") || domain.endsWith(".") || domain.includes("..")) {
    return false;
  }

  const dotIndex = domain.indexOf(".");
  return dotIndex > 0 && dotIndex < domain.length - 1;
}

export async function POST(req: NextRequest) {
  try {
    const { firstname, lastname, email, company, message, source, pageUri, pageName, attribution } =
      (await req.json()) as {
        firstname?: string;
        lastname?: string;
        email?: string;
        company?: string;
        message?: string;
        source?: string;
        pageUri?: string;
        pageName?: string;
        attribution?: AttributionPayload;
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

    fields.push({ name: "runmat_source", value: source?.trim() || "website_contact_page" });

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
      const campaign = attribution?.utmCampaign?.trim() || undefined;
      const intakePayload = {
        kind: "contact",
        email: email.trim().toLowerCase(),
        source: source?.trim() || "website_contact_page",
        campaign,
        attribution: attribution
          ? {
              utmSource: attribution.utmSource,
              utmMedium: attribution.utmMedium,
              utmCampaign: attribution.utmCampaign,
              utmTerm: attribution.utmTerm,
              utmContent: attribution.utmContent,
              gclid: attribution.gclid,
              gbraid: attribution.gbraid,
              wbraid: attribution.wbraid,
              msclkid: attribution.msclkid,
              fbclid: attribution.fbclid,
              ttclid: attribution.ttclid,
              liFatId: attribution.liFatId,
              landingPageUrl: attribution.landingPageUrl,
              pageReferrer: attribution.pageReferrer,
              capturedAt: attribution.capturedAt,
              gaClientId: attribution.gaClientId,
            }
          : undefined,
      };
      void fetch(`${RUNMAT_SERVER_API_BASE_URL.replace(/\/$/, "")}/v1/lifecycle/intake`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(intakePayload),
      }).catch(() => undefined);
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
