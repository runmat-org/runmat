import type { RunMatFilesystemProvider } from "../fs/provider-types.js";
import type {
  GitAcquisitionIntent,
  GitAcquisitionPlan,
  GitAcquisitionPolicy,
  GitSnapshotWire,
  ServerGitGatewayOptions
} from "./git-gateway.js";
import { fetchGitTreeInventoryWire } from "./git-gateway.js";
import type {
  ServerProjectAcquisitionPlan,
  ServerProjectSnapshotOptions
} from "./server-project.js";
import { fetchServerProjectSnapshot } from "./server-project.js";
import type {
  BrowserRegistryOptions,
  RegistryAcquisitionPlan,
  RegistryCandidatePlan
} from "./registry.js";
import {
  fetchRegistryCandidates,
  fetchRegistryRelease
} from "./registry.js";
import { BrowserPackageMountFilesystem } from "./mount.js";
import type { PackageSnapshotMountInput } from "./mount.js";
import type { RunMatPackageCacheProvider } from "./provider-types.js";
import type { PackageCacheLease } from "./provider-types.js";

const LEASE_TTL_MS = 120_000n;
const LEASE_RENEW_MS = 40_000;
let nextResolverOwner = 1;

export interface BrowserProjectResolveOptions {
  target: string;
  default_server_origin: string;
  default_registry_index: string;
  groups: Array<"runtime" | "development" | "test">;
  root_features: string[];
  host_capabilities: string[];
  source_intent: GitAcquisitionIntent;
  source_policy: GitAcquisitionPolicy;
}

export interface BrowserProjectResolveRequest {
  manifestPath: string;
  existingLock?: unknown;
  options: BrowserProjectResolveOptions;
}

export interface BrowserResolvedProject {
  frozen: unknown;
  lock: unknown;
  lock_decision: "use-existing" | "write-generated";
  acquired_git_sources: unknown[];
  acquired_server_sources: unknown[];
  acquired_registry_sources: unknown[];
  source_inventories: unknown[];
}

interface BrowserResolveWireResult extends BrowserResolvedProject {
  cache_lease?: PackageCacheLease | null;
}

export interface BrowserProjectResolverNative {
  resolveProject(
    request: BrowserProjectResolveRequest,
    provider: {
      packageCache: RunMatPackageCacheProvider;
      leaseOwner: string;
      fetchGitInventory(plan: GitAcquisitionPlan): Promise<unknown>;
      fetchServerSnapshot(plan: ServerProjectAcquisitionPlan): Promise<unknown>;
      fetchRegistryCandidates(plan: RegistryCandidatePlan): Promise<unknown>;
      fetchRegistryRelease(plan: RegistryAcquisitionPlan): Promise<unknown>;
      mountPackageSnapshot(snapshot: GitSnapshotWire): string;
    },
    filesystem: RunMatFilesystemProvider
  ): Promise<BrowserResolveWireResult>;
  packageCacheRenewLease(
    provider: RunMatPackageCacheProvider,
    lease: PackageCacheLease,
    ttlMs: bigint
  ): Promise<PackageCacheLease>;
  packageCacheReleaseLease(
    provider: RunMatPackageCacheProvider,
    lease: PackageCacheLease
  ): Promise<void>;
}

export interface BrowserProjectResolverConfig {
  native: BrowserProjectResolverNative;
  filesystem: RunMatFilesystemProvider;
  packageCache: RunMatPackageCacheProvider;
  gitGateway: ServerGitGatewayOptions;
  serverSnapshots?: ServerProjectSnapshotOptions;
  registry?: BrowserRegistryOptions;
}

/**
 * Browser composition root for the portable Rust package resolver.
 *
 * JavaScript supplies storage, authenticated transport, and immutable mount mechanics;
 * selector/lock/network policy, snapshot verification, CAS publication, graph building,
 * and lock reconciliation remain in Rust.
 */
export class BrowserProjectResolver {
  public readonly filesystem: BrowserPackageMountFilesystem;
  private readonly leaseOwner = createLeaseOwner();
  private cacheLease: PackageCacheLease | null = null;
  private renewalTimer: ReturnType<typeof setInterval> | null = null;
  private renewalInFlight: Promise<void> | null = null;
  private resolving = false;
  private disposed = false;
  private leaseFailure: unknown = null;

  constructor(private readonly config: BrowserProjectResolverConfig) {
    this.filesystem = new BrowserPackageMountFilesystem(config.filesystem);
  }

  async resolve(request: BrowserProjectResolveRequest): Promise<BrowserResolvedProject> {
    if (this.disposed) {
      throw new Error("Browser project resolver has been disposed");
    }
    if (this.resolving || this.cacheLease) {
      throw new Error("Dispose the active browser project before resolving another project");
    }
    this.resolving = true;
    try {
      const { cache_lease: cacheLease, ...resolved } =
        await this.config.native.resolveProject(
          request,
          {
            packageCache: this.config.packageCache,
            leaseOwner: this.leaseOwner,
            fetchGitInventory: (plan) =>
              fetchGitTreeInventoryWire(
                {
                  repository: plan.repository,
                  selector: plan.selector,
                  subdir: plan.subdir
                },
                this.config.gitGateway
              ),
            fetchServerSnapshot: (plan) =>
              fetchServerProjectSnapshot(plan, this.config.serverSnapshots),
            fetchRegistryCandidates: (plan) =>
              fetchRegistryCandidates(plan, this.config.registry),
            fetchRegistryRelease: (plan) =>
              fetchRegistryRelease(plan, this.config.registry),
            mountPackageSnapshot: (snapshot) =>
              this.filesystem.register(
                snapshot as unknown as PackageSnapshotMountInput,
                this.config.packageCache
              )
          },
          this.filesystem
        );
      this.cacheLease = cacheLease ?? null;
      if (this.cacheLease) {
        this.startRenewal();
      }
      return resolved;
    } catch (error) {
      this.filesystem.clear();
      throw error;
    } finally {
      this.resolving = false;
    }
  }

  get cacheLeaseError(): unknown {
    return this.leaseFailure;
  }

  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    if (this.renewalTimer) {
      clearInterval(this.renewalTimer);
      this.renewalTimer = null;
    }
    await this.renewalInFlight;
    const lease = this.cacheLease;
    this.cacheLease = null;
    this.filesystem.clear();
    if (lease) {
      await this.config.native.packageCacheReleaseLease(this.config.packageCache, lease);
    }
  }

  private startRenewal(): void {
    this.renewalTimer = setInterval(() => {
      if (!this.renewalInFlight) {
        this.renewalInFlight = this.renewLease().finally(() => {
          this.renewalInFlight = null;
        });
      }
    }, LEASE_RENEW_MS);
  }

  private async renewLease(): Promise<void> {
    const lease = this.cacheLease;
    if (!lease || this.disposed) {
      return;
    }
    try {
      this.cacheLease = await this.config.native.packageCacheRenewLease(
        this.config.packageCache,
        lease,
        LEASE_TTL_MS
      );
      this.leaseFailure = null;
    } catch (error) {
      this.leaseFailure = error;
    }
  }
}

function createLeaseOwner(): string {
  const random = globalThis.crypto?.randomUUID?.();
  return random
    ? `browser-${random}`
    : `browser-${Date.now()}-${nextResolverOwner++}`;
}
