import { dirname, normalizePath } from "../fs/path.js";
import type { RunMatFilesystemProvider } from "../fs/provider-types.js";
import {
  BrowserProjectResolver,
  type BrowserProjectResolveRequest,
  type BrowserProjectResolverConfig,
  type BrowserProjectResolverNative,
  type BrowserResolvedProject
} from "./browser-resolver.js";
import type {
  PackageCacheGcPlan,
  PackageCacheStatus,
  RunMatPackageCacheProvider
} from "./provider-types.js";

const LOCK_FILE_NAME = "runmat.lock";

export interface BrowserProjectSessionHandle {
  setFsProvider?(provider: RunMatFilesystemProvider): Promise<void> | void;
  installProjectHandoff(handoff: unknown): Promise<unknown>;
  clearProjectHandoff(): Promise<void>;
  projectRevision(): Promise<unknown | null>;
}

export interface BrowserProjectSessionNative extends BrowserProjectResolverNative {
  decodePackageLock(input: string): unknown;
  encodePackageLock(lock: unknown): string;
  handoffFromFrozenProject(project: unknown): unknown;
  packageCacheStatus(provider: RunMatPackageCacheProvider): Promise<PackageCacheStatus>;
  packageCacheGc(
    provider: RunMatPackageCacheProvider,
    targetBytes: bigint,
    retainRecentMs: bigint
  ): Promise<PackageCacheGcPlan>;
}

export interface BrowserProjectSessionConfig
  extends Omit<BrowserProjectResolverConfig, "native"> {
  native: BrowserProjectSessionNative;
  session: BrowserProjectSessionHandle;
}

export interface BrowserProjectSessionResolveRequest
  extends BrowserProjectResolveRequest {
  /**
   * Defaults to `runmat.lock` beside the manifest. Set to `null` only for an
   * intentionally ephemeral embedding that must neither read nor write a lock.
   */
  lockPath?: string | null;
}

export interface BrowserProjectSessionResolution {
  project: BrowserResolvedProject;
  handoff: unknown;
  revision: unknown;
  lockPath: string | null;
  lockWritten: boolean;
}

/**
 * Owns one resolved browser project and its immutable cache lease/mount table.
 *
 * Rust remains authoritative for lock parsing, solving, graph construction,
 * handoff validation, and cache policy. This class only coordinates host ports
 * and makes teardown explicit.
 */
export class BrowserProjectSession {
  private readonly resolver: BrowserProjectResolver;
  private active: BrowserProjectSessionResolution | null = null;
  private disposed = false;

  constructor(private readonly config: BrowserProjectSessionConfig) {
    this.resolver = new BrowserProjectResolver(config);
  }

  get resolution(): BrowserProjectSessionResolution | null {
    return this.active;
  }

  get filesystem(): RunMatFilesystemProvider {
    return this.resolver.filesystem;
  }

  get cacheLeaseError(): unknown {
    return this.resolver.cacheLeaseError;
  }

  async resolve(
    request: BrowserProjectSessionResolveRequest
  ): Promise<BrowserProjectSessionResolution> {
    if (this.disposed) {
      throw new Error("Browser project session has been disposed");
    }
    if (this.active) {
      throw new Error("Dispose the active browser project before resolving another project");
    }

    const lockPath =
      request.lockPath === undefined
        ? normalizePath(`${dirname(request.manifestPath)}/${LOCK_FILE_NAME}`)
        : request.lockPath;
    const existingLock =
      request.existingLock ??
      (lockPath === null ? undefined : await this.readExistingLock(lockPath));
    const project = await this.resolver.resolve({
      manifestPath: request.manifestPath,
      existingLock,
      options: request.options
    });

    try {
      const handoff = await this.config.native.handoffFromFrozenProject(project.frozen);
      await this.config.session.setFsProvider?.(this.resolver.filesystem);
      const revision = await this.config.session.installProjectHandoff(handoff);
      const lockWritten =
        lockPath !== null && project.lock_decision === "write-generated"
          ? await this.writeLock(lockPath, project.lock)
          : false;
      this.active = {
        project,
        handoff,
        revision,
        lockPath,
        lockWritten
      };
      return this.active;
    } catch (error) {
      this.disposed = true;
      await this.rollback().catch(() => {});
      throw error;
    }
  }

  async sourceRevision(): Promise<unknown | null> {
    return this.config.session.projectRevision();
  }

  async cacheStatus(): Promise<PackageCacheStatus> {
    return this.config.native.packageCacheStatus(this.config.packageCache);
  }

  async collectCache(
    targetBytes: bigint,
    retainRecentMs = 0n
  ): Promise<PackageCacheGcPlan> {
    if (targetBytes < 0n || retainRecentMs < 0n) {
      throw new Error("Browser package cache GC values must be non-negative");
    }
    return this.config.native.packageCacheGc(
      this.config.packageCache,
      targetBytes,
      retainRecentMs
    );
  }

  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    await this.rollback();
  }

  private async readExistingLock(lockPath: string): Promise<unknown | undefined> {
    try {
      const metadata = await this.config.filesystem.metadata(lockPath);
      if (metadata.fileType !== "file") {
        throw new Error(`Project lock path is not a file: ${lockPath}`);
      }
    } catch (error) {
      if (isNotFound(error)) {
        return undefined;
      }
      throw error;
    }
    const bytes = await this.config.filesystem.readFile(lockPath);
    return this.config.native.decodePackageLock(
      new TextDecoder("utf-8", { fatal: true }).decode(normalizeBytes(bytes))
    );
  }

  private async writeLock(lockPath: string, lock: unknown): Promise<boolean> {
    const bytes = new TextEncoder().encode(
      await this.config.native.encodePackageLock(lock)
    );
    const exists = await this.pathExists(lockPath);
    if (exists || !this.config.filesystem.rename) {
      await this.config.filesystem.writeFile(lockPath, bytes);
      return true;
    }

    const suffix =
      globalThis.crypto?.randomUUID?.() ??
      `${Date.now()}-${Math.floor(Math.random() * Number.MAX_SAFE_INTEGER)}`;
    const temporaryPath = `${lockPath}.runmat-${suffix}.tmp`;
    await this.config.filesystem.writeFile(temporaryPath, bytes);
    try {
      await this.config.filesystem.rename(temporaryPath, lockPath);
    } catch (error) {
      try {
        await this.config.filesystem.removeFile(temporaryPath);
      } catch {
        // Preserve the atomic rename failure; a best-effort temp cleanup is secondary.
      }
      throw error;
    }
    return true;
  }

  private async pathExists(path: string): Promise<boolean> {
    try {
      await this.config.filesystem.metadata(path);
      return true;
    } catch (error) {
      if (isNotFound(error)) {
        return false;
      }
      throw error;
    }
  }

  private async rollback(): Promise<void> {
    this.active = null;
    await this.config.session.clearProjectHandoff().catch(() => {});
    await this.config.session.setFsProvider?.(this.config.filesystem);
    await this.resolver.dispose();
  }
}

function normalizeBytes(value: Uint8Array | ArrayBuffer): Uint8Array {
  return value instanceof Uint8Array ? value : new Uint8Array(value);
}

function isNotFound(error: unknown): boolean {
  if (typeof error === "object" && error !== null && "code" in error) {
    if ((error as { code?: unknown }).code === "ENOENT") {
      return true;
    }
  }
  const message = error instanceof Error ? error.message : String(error);
  const normalized = message.toLowerCase();
  return normalized.includes("not found") || normalized.includes("(404)");
}
