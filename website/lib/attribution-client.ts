"use client";

export type AttributionPayload = {
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

const STORAGE_KEY = "runmat_attribution_v1";

const QUERY_FIELD_MAP: Record<string, keyof AttributionPayload> = {
  utm_source: "utmSource",
  utm_medium: "utmMedium",
  utm_campaign: "utmCampaign",
  utm_term: "utmTerm",
  utm_content: "utmContent",
  gclid: "gclid",
  gbraid: "gbraid",
  wbraid: "wbraid",
  msclkid: "msclkid",
  fbclid: "fbclid",
  ttclid: "ttclid",
  li_fat_id: "liFatId",
};

function readStoredAttribution(): AttributionPayload {
  if (typeof window === "undefined") {
    return {};
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw) as AttributionPayload;
    return parsed ?? {};
  } catch {
    return {};
  }
}

function writeStoredAttribution(value: AttributionPayload): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
  } catch {
    // no-op
  }
}

function mergeDefined(base: AttributionPayload, patch: AttributionPayload): AttributionPayload {
  const next: AttributionPayload = { ...base };
  for (const [key, rawValue] of Object.entries(patch)) {
    const value = typeof rawValue === "string" ? rawValue.trim() : rawValue;
    if (!value) {
      continue;
    }
    next[key as keyof AttributionPayload] = value;
  }
  return next;
}

export function captureAttributionFromLocation(): AttributionPayload {
  if (typeof window === "undefined") {
    return {};
  }
  const params = new URLSearchParams(window.location.search);
  const fromQuery: AttributionPayload = {};
  for (const [queryKey, targetKey] of Object.entries(QUERY_FIELD_MAP)) {
    const value = params.get(queryKey);
    if (!value) {
      continue;
    }
    fromQuery[targetKey] = value;
  }
  const captured = mergeDefined(readStoredAttribution(), {
    ...fromQuery,
    landingPageUrl: window.location.href,
    pageReferrer: typeof document !== "undefined" ? document.referrer || undefined : undefined,
    capturedAt: new Date().toISOString(),
  });
  writeStoredAttribution(captured);
  return captured;
}

export function setAttributionGaClientId(gaClientId: string | undefined): AttributionPayload {
  const existing = readStoredAttribution();
  const next = mergeDefined(existing, { gaClientId });
  writeStoredAttribution(next);
  return next;
}

export function getCurrentAttribution(): AttributionPayload {
  if (typeof window === "undefined") {
    return {};
  }
  const current = captureAttributionFromLocation();
  return current;
}
