export type GitGatewaySelector =
  | { kind: "rev"; value: string }
  | { kind: "tag"; value: string }
  | { kind: "branch"; value: string };

export interface GitGatewayRequest {
  repository: string;
  selector: GitGatewaySelector;
  subdir?: string;
}

export interface GitSnapshotWire {
  source: unknown;
  tree: unknown;
  blobs: Array<{
    digest: string;
    bytes: string;
  }>;
}

export type GitAcquisitionIntent = "execute" | "fetch" | "update";

export interface GitSourceWire {
  repository: string;
  commit: {
    algorithm: "sha1" | "sha256";
    hex: string;
  };
  subdir: string;
  tree_digest: string;
}

export interface GitAcquisitionPolicy {
  locked?: boolean;
  frozen?: boolean;
  offline?: boolean;
}

export interface GitAcquisitionPlanRequest extends GitGatewayRequest {
  lockedSource?: GitSourceWire;
  intent: GitAcquisitionIntent;
  policy?: GitAcquisitionPolicy;
}

export interface GitAcquisitionPlan {
  repository: string;
  selector: GitGatewaySelector;
  subdir: string;
  allow_network: boolean;
  expected?: GitSourceWire;
  lock_action: "preserve" | "write" | "replace";
}

export interface GitTreeInventoryWire {
  commit: string;
  entries: Array<{
    path: string;
    kind: "file" | "directory" | "symlink";
    bytes?: string;
    executable?: boolean;
    linkTarget?: string;
  }>;
}

export interface ServerGitGatewayOptions {
  baseUrl: string;
  authToken?: string | (() => string | null | Promise<string | null>);
  fetch?: typeof globalThis.fetch;
  signal?: AbortSignal;
}

function trimTrailingSlashes(value: string): string {
  let end = value.length;
  while (end > 0 && value.charCodeAt(end - 1) === 47) {
    end -= 1;
  }
  return end === value.length ? value : value.slice(0, end);
}

export async function fetchGitTreeInventoryWire(
  request: GitGatewayRequest,
  options: ServerGitGatewayOptions
): Promise<GitTreeInventoryWire> {
  const fetcher = options.fetch ?? globalThis.fetch;
  if (typeof fetcher !== "function") {
    throw new Error("fetch API is unavailable for the RunMat Git gateway");
  }
  const token =
    typeof options.authToken === "function" ? await options.authToken() : options.authToken;
  const response = await fetcher(
    `${trimTrailingSlashes(options.baseUrl)}/v1/packages/git/snapshot`,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(token ? { authorization: `Bearer ${token}` } : {})
      },
      body: JSON.stringify({
        repository: request.repository,
        selector: request.selector,
        subdir: request.subdir ?? "."
      }),
      cache: "no-store",
      credentials: "omit",
      signal: options.signal
    }
  );
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(
      `RunMat Git gateway failed with HTTP ${response.status}${detail ? `: ${detail}` : ""}`
    );
  }
  const value: unknown = await response.json();
  if (!value || typeof value !== "object") {
    throw new Error("RunMat Git gateway returned an invalid tree inventory payload");
  }
  return value as GitTreeInventoryWire;
}
