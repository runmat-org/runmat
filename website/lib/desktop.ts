const WORKSPACE_PARAM = "workspace";
const STORAGE_PREFIX = "runmat:workspace:";
const DEFAULT_TTL_MS = 10 * 60 * 1000; // 10 minutes
const MAX_PAYLOAD_CHARS = 1_000_000; // ~2MB of UTF-16 payload data

/**
 * Usage example:
 *
 * ```ts
 * import { openWorkspace } from "@/lib/desktop";
 *
 * openWorkspace(
 *   [{ path: "/demo.m", content: "disp('hello from docs');" }],
 *   { targetPath: "/sandbox", metadata: { source: "example-page" } }
 * );
 * ```
 */

export interface WorkspaceFile {
  path: string;
  content: string;
}

export interface WorkspaceLaunchMetadata {
  source?: string;
  label?: string;
  [key: string]: unknown;
}

export interface WorkspaceLaunchPayload {
  id: string;
  files: WorkspaceFile[];
  openPath?: string;
  metadata?: WorkspaceLaunchMetadata;
  createdAt: number;
  expiresAt: number;
}

export interface WorkspaceLaunchOptions {
  targetPath?: string;
  openPath?: string;
  metadata?: WorkspaceLaunchMetadata;
  ttlMs?: number;
  useHash?: boolean;
  baseUrl?: string;
}

export interface WorkspaceOpenOptions extends WorkspaceLaunchOptions {
  newTab?: boolean;
  windowName?: string;
}

export interface WorkspaceLaunchResult {
  id: string;
  url: string;
  payload: WorkspaceLaunchPayload;
}

export function prepareWorkspaceLaunch(files: WorkspaceFile[], options?: WorkspaceLaunchOptions): WorkspaceLaunchResult {
  const storage = getStorageOrThrow();
  pruneExpired(storage);

  const sanitizedFiles = normalizeFiles(files);
  const id = createWorkspaceId();
  const createdAt = Date.now();
  const ttl = Math.max(options?.ttlMs ?? DEFAULT_TTL_MS, 1_000);

  const payload: WorkspaceLaunchPayload = {
    id,
    files: sanitizedFiles,
    openPath: options?.openPath,
    metadata: options?.metadata,
    createdAt,
    expiresAt: createdAt + ttl
  };

  const serialized = JSON.stringify(payload);
  if (serialized.length > MAX_PAYLOAD_CHARS) {
    throw new Error("Workspace snapshot is too large to open in the browser sandbox.");
  }

  storage.setItem(buildStorageKey(id), serialized);

  return {
    id,
    url: buildWorkspaceUrl({
      workspaceId: id,
      targetPath: options?.targetPath,
      useHash: options?.useHash,
      baseUrl: options?.baseUrl
    }),
    payload
  };
}

export function openWorkspace(files: WorkspaceFile[], options?: WorkspaceOpenOptions): string {
  const { url } = prepareWorkspaceLaunch(files, options);
  const newTab = options?.newTab ?? true;
  if (typeof window === "undefined") {
    return url;
  }

  if (newTab) {
    window.open(url, options?.windowName ?? "_blank", "noopener,noreferrer");
  } else {
    window.location.assign(url);
  }
  return url;
}

export function buildWorkspaceUrl({
  workspaceId,
  targetPath = "/sandbox",
  useHash,
  baseUrl
}: {
  workspaceId: string;
  targetPath?: string;
  useHash?: boolean;
  baseUrl?: string;
}): string {
  const normalizedPath = normalizeTargetPath(targetPath);
  const base = baseUrl ?? (typeof window !== "undefined" ? window.location.origin : "https://runmat.com");
  const url = new URL(normalizedPath, base);

  if (useHash) {
    const hashParams = new URLSearchParams(url.hash.replace(/^#/, ""));
    hashParams.set(WORKSPACE_PARAM, workspaceId);
    url.hash = hashParams.toString();
  } else {
    url.searchParams.set(WORKSPACE_PARAM, workspaceId);
  }

  return url.toString();
}

export function clearWorkspacePayload(workspaceId: string): void {
  const storage = getStorage();
  if (!storage) {
    return;
  }
  storage.removeItem(buildStorageKey(workspaceId));
}

function normalizeFiles(files: WorkspaceFile[]): WorkspaceFile[] {
  return files.map((file, index) => ({
    path: normalizePath(file.path ?? `/unnamed-${index}.m`),
    content: file.content ?? ""
  }));
}

function normalizePath(path: string): string {
  if (!path) {
    return "/untitled.m";
  }
  return path.startsWith("/") ? path : `/${path}`;
}

function normalizeTargetPath(targetPath: string): string {
  if (targetPath.startsWith("http://") || targetPath.startsWith("https://")) {
    return targetPath;
  }
  return targetPath.startsWith("/") ? targetPath : `/${targetPath}`;
}

function pruneExpired(storage: Storage): void {
  const now = Date.now();
  const toDelete: string[] = [];
  for (let i = 0; i < storage.length; i += 1) {
    const key = storage.key(i);
    if (!key || !key.startsWith(STORAGE_PREFIX)) {
      continue;
    }
    const raw = storage.getItem(key);
    if (!raw) {
      toDelete.push(key);
      continue;
    }
    try {
      const payload = JSON.parse(raw) as WorkspaceLaunchPayload;
      if (!payload.expiresAt || payload.expiresAt < now) {
        toDelete.push(key);
      }
    } catch {
      toDelete.push(key);
    }
  }
  toDelete.forEach((key) => storage.removeItem(key));
}

function createWorkspaceId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function getStorage(): Storage | null {
  if (typeof window === "undefined" || typeof window.localStorage === "undefined") {
    return null;
  }
  return window.localStorage;
}

function getStorageOrThrow(): Storage {
  const storage = getStorage();
  if (!storage) {
    throw new Error("Workspace launches must be triggered in a browser environment.");
  }
  return storage;
}

function buildStorageKey(workspaceId: string): string {
  return `${STORAGE_PREFIX}${workspaceId}`;
}

export { WORKSPACE_PARAM, STORAGE_PREFIX as WORKSPACE_STORAGE_PREFIX, DEFAULT_TTL_MS as WORKSPACE_DEFAULT_TTL_MS };
