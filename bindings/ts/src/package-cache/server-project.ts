export interface ServerProjectAcquisitionPlan {
  service: string;
  project: string;
  selector: { kind: "exact" | "tag"; value: string };
  allow_network: boolean;
  expected?: {
    service: string;
    project: string;
    snapshot: string;
    tree_digest: string;
  };
  lock_action: "preserve" | "write" | "replace";
}

export interface ServerProjectTreeInventoryWire {
  project: string;
  snapshot: string;
  treeDigest: string;
  entries: Array<{
    path: string;
    kind: "file" | "directory";
    bytes?: string;
    executable?: boolean;
  }>;
}

export interface ServerProjectSnapshotOptions {
  authenticatedOrigin?: string;
  authToken?: string | (() => string | null | Promise<string | null>);
  fetch?: typeof globalThis.fetch;
  signal?: AbortSignal;
  etag?: string;
  range?: { start: number; end?: number };
}

export type ServerProjectSnapshotFetchOutcome =
  | {
      kind: "snapshot";
      inventory: ServerProjectTreeInventoryWire;
      etag?: string;
    }
  | { kind: "not-modified"; etag?: string }
  | {
      kind: "partial";
      bytes: Uint8Array;
      contentRange: string;
      etag?: string;
    };

export async function fetchServerProjectSnapshot(
  plan: ServerProjectAcquisitionPlan,
  options: ServerProjectSnapshotOptions = {}
): Promise<ServerProjectTreeInventoryWire> {
  const outcome = await fetchServerProjectSnapshotResponse(plan, options);
  if (outcome.kind !== "snapshot") {
    throw new Error(
      `RunMat Server returned ${outcome.kind} where a complete project snapshot was required`
    );
  }
  return outcome.inventory;
}

export async function fetchServerProjectSnapshotResponse(
  plan: ServerProjectAcquisitionPlan,
  options: ServerProjectSnapshotOptions = {}
): Promise<ServerProjectSnapshotFetchOutcome> {
  if (!plan.allow_network) {
    throw new Error("Server project snapshot network access is disabled");
  }
  const fetcher = options.fetch ?? globalThis.fetch;
  if (typeof fetcher !== "function") {
    throw new Error("fetch API is unavailable for RunMat Server project snapshots");
  }
  const origin = normalizeOrigin(plan.service);
  const authenticatedOrigin = options.authenticatedOrigin
    ? normalizeOrigin(options.authenticatedOrigin)
    : null;
  const token =
    authenticatedOrigin === origin
      ? typeof options.authToken === "function"
        ? await options.authToken()
        : options.authToken
      : null;
  const url = `${origin}/v1/packages/projects/${encodeURIComponent(
    plan.project
  )}/snapshots/${encodeURIComponent(plan.selector.value)}`;
  const headers: Record<string, string> = {};
  if (token) headers.authorization = `Bearer ${token}`;
  if (options.etag) headers["if-none-match"] = options.etag;
  if (options.range) {
    if (
      !Number.isSafeInteger(options.range.start) ||
      options.range.start < 0 ||
      (options.range.end !== undefined &&
        (!Number.isSafeInteger(options.range.end) ||
          options.range.end < options.range.start))
    ) {
      throw new Error("RunMat Server project snapshot byte range is invalid");
    }
    headers.range = `bytes=${options.range.start}-${options.range.end ?? ""}`;
  }
  const response = await fetcher(url, {
    method: "GET",
    headers,
    cache: "no-store",
    credentials: "omit",
    signal: options.signal
  });
  const etag = response.headers.get("etag") ?? undefined;
  if (response.status === 304) {
    return { kind: "not-modified", etag };
  }
  if (response.status === 206) {
    const contentRange = response.headers.get("content-range");
    if (!contentRange) {
      throw new Error("RunMat Server returned a partial snapshot without Content-Range");
    }
    return {
      kind: "partial",
      bytes: new Uint8Array(await response.arrayBuffer()),
      contentRange,
      etag
    };
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(
      `RunMat Server project snapshot failed with HTTP ${response.status}${
        detail ? `: ${detail}` : ""
      }`
    );
  }
  const value: unknown = await response.json();
  if (!value || typeof value !== "object") {
    throw new Error("RunMat Server returned an invalid project snapshot payload");
  }
  return {
    kind: "snapshot",
    inventory: value as ServerProjectTreeInventoryWire,
    etag
  };
}

function normalizeOrigin(value: string): string {
  const url = new URL(value);
  if (
    url.protocol !== "https:" ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  ) {
    throw new Error("RunMat Server origin must be a credential-free HTTPS URL");
  }
  return url.toString().replace(/\/+$/, "");
}
