const METADATA_LIMIT_BYTES = 4 * 1024 * 1024;
const DEFAULT_ARTIFACT_LIMIT_BYTES = 512 * 1024 * 1024;

export interface RegistryCandidatePlan {
  source_registry: string;
  index: string;
  package: string;
  allow_network: boolean;
}

export interface RegistryAcquisitionPlan extends RegistryCandidatePlan {
  requirement: string;
  expected?: {
    release: string;
  };
  lock_action: "preserve" | "write" | "replace";
}

export interface BrowserRegistryOptions {
  authenticatedOrigin?: string;
  authToken?: string | (() => string | null | Promise<string | null>);
  fetch?: typeof globalThis.fetch;
  signal?: AbortSignal;
  maxArtifactBytes?: number;
}

export interface RegistryReleaseTransfer {
  release: unknown;
  artifactBytes: Uint8Array;
}

export async function fetchRegistryCandidates(
  plan: RegistryCandidatePlan,
  options: BrowserRegistryOptions = {}
): Promise<unknown> {
  if (!plan.allow_network) {
    throw new Error("registry candidate network access is disabled");
  }
  const index = normalizeIndex(plan.index);
  const { namespace, name } = parsePackage(plan.package);
  return fetchJson(
    endpoint(index, [
      "v1",
      "packages",
      "registry",
      namespace,
      name,
      "candidates"
    ]),
    index,
    options
  );
}

export async function fetchRegistryRelease(
  plan: RegistryAcquisitionPlan,
  options: BrowserRegistryOptions = {}
): Promise<RegistryReleaseTransfer> {
  if (!plan.allow_network) {
    throw new Error("registry release network access is disabled");
  }
  const index = normalizeIndex(plan.index);
  const releaseUrl = plan.expected
    ? endpoint(index, [
        "v1",
        "packages",
        "registry",
        "releases",
        plan.expected.release
      ])
    : (() => {
        const { namespace, name } = parsePackage(plan.package);
        const url = endpoint(index, [
          "v1",
          "packages",
          "registry",
          namespace,
          name,
          "resolve"
        ]);
        url.searchParams.set("requirement", plan.requirement);
        return url;
      })();
  const release = await fetchJson(releaseUrl, index, options);
  const artifact = releaseArtifact(release);
  const limit = options.maxArtifactBytes ?? DEFAULT_ARTIFACT_LIMIT_BYTES;
  if (!Number.isSafeInteger(limit) || limit <= 0) {
    throw new Error("registry artifact byte limit must be a positive safe integer");
  }
  if (!Number.isSafeInteger(artifact.byteLen) || artifact.byteLen < 0) {
    throw new Error("registry release contains an invalid artifact length");
  }
  if (artifact.byteLen > limit) {
    throw new Error(`registry artifact exceeds the ${limit}-byte browser limit`);
  }
  const artifactUrl = sameOriginArtifactUrl(index, artifact.downloadUrl);
  const response = await request(artifactUrl, index, options);
  if (!response.ok) {
    throw new Error(`registry artifact request failed with HTTP ${response.status}`);
  }
  const bytes = await readBounded(response, limit, artifact.byteLen);
  return { release, artifactBytes: bytes };
}

async function fetchJson(
  url: URL,
  index: URL,
  options: BrowserRegistryOptions
): Promise<unknown> {
  const response = await request(url, index, options);
  if (!response.ok) {
    throw new Error(`registry metadata request failed with HTTP ${response.status}`);
  }
  const bytes = await readBounded(response, METADATA_LIMIT_BYTES);
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new Error("registry returned invalid JSON metadata");
  }
  if (!value || typeof value !== "object") {
    throw new Error("registry returned an invalid metadata payload");
  }
  return value;
}

async function request(
  url: URL,
  index: URL,
  options: BrowserRegistryOptions
): Promise<Response> {
  const fetcher = options.fetch ?? globalThis.fetch;
  if (typeof fetcher !== "function") {
    throw new Error("fetch API is unavailable for registry requests");
  }
  const trusted = options.authenticatedOrigin
    ? normalizeIndex(options.authenticatedOrigin)
    : null;
  const token =
    trusted?.origin === index.origin
      ? typeof options.authToken === "function"
        ? await options.authToken()
        : options.authToken
      : null;
  const headers: Record<string, string> = {};
  if (token) headers.authorization = `Bearer ${token}`;
  return fetcher(url, {
    method: "GET",
    headers,
    cache: "no-store",
    credentials: "omit",
    redirect: "manual",
    signal: options.signal
  });
}

async function readBounded(
  response: Response,
  limit: number,
  expectedLength?: number
): Promise<Uint8Array> {
  const declared = response.headers.get("content-length");
  if (declared !== null) {
    const length = Number(declared);
    if (!Number.isSafeInteger(length) || length < 0 || length > limit) {
      throw new Error(`registry response exceeds the ${limit}-byte transfer limit`);
    }
    if (expectedLength !== undefined && length !== expectedLength) {
      throw new Error("registry artifact length differs from signed metadata");
    }
  }
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    validateLength(bytes.byteLength, limit, expectedLength);
    return bytes;
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.byteLength;
      validateLength(length, limit);
      chunks.push(value);
    }
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  }
  validateLength(length, limit, expectedLength);
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function validateLength(length: number, limit: number, expectedLength?: number): void {
  if (length > limit) {
    throw new Error(`registry response exceeds the ${limit}-byte transfer limit`);
  }
  if (expectedLength !== undefined && length !== expectedLength) {
    throw new Error("registry artifact length differs from signed metadata");
  }
}

function normalizeIndex(value: string): URL {
  const url = new URL(value);
  if (
    url.protocol !== "https:" ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  ) {
    throw new Error("registry index must be a credential-free HTTPS URL");
  }
  url.pathname = url.pathname.replace(/\/+$/, "");
  return url;
}

function endpoint(index: URL, segments: string[]): URL {
  const url = new URL(index);
  const base = url.pathname.replace(/\/+$/, "");
  url.pathname = `${base}/${segments.map(encodeURIComponent).join("/")}`;
  return url;
}

function sameOriginArtifactUrl(index: URL, value: string): URL {
  if (!value.startsWith("/") || value.startsWith("//")) {
    throw new Error("registry returned an unsafe artifact URL");
  }
  const url = new URL(value, index);
  if (
    url.origin !== index.origin ||
    url.username ||
    url.password ||
    url.hash
  ) {
    throw new Error("registry returned an unsafe artifact URL");
  }
  return url;
}

function parsePackage(value: string): { namespace: string; name: string } {
  const colon = value.indexOf(":");
  const slash = value.indexOf("/", colon + 1);
  if (colon <= 0 || slash <= colon + 1 || value.indexOf("/", slash + 1) !== -1) {
    throw new Error("registry plan contains an invalid canonical package ID");
  }
  return {
    namespace: value.slice(colon + 1, slash),
    name: value.slice(slash + 1)
  };
}

function releaseArtifact(value: unknown): { byteLen: number; downloadUrl: string } {
  const artifact = (value as { artifact?: unknown }).artifact;
  if (!artifact || typeof artifact !== "object") {
    throw new Error("registry release is missing artifact metadata");
  }
  const { byteLen, downloadUrl } = artifact as {
    byteLen?: unknown;
    downloadUrl?: unknown;
  };
  if (typeof byteLen !== "number" || typeof downloadUrl !== "string") {
    throw new Error("registry release contains invalid artifact metadata");
  }
  return { byteLen, downloadUrl };
}
