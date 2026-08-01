import type { RunMatFilesystemProvider } from "../fs/provider-types.js";
import type {
  GitAcquisitionIntent,
  GitAcquisitionPlan,
  GitAcquisitionPolicy,
  GitSnapshotWire,
  ServerGitGatewayOptions
} from "./git-gateway.js";
import { fetchGitTreeInventoryWire } from "./git-gateway.js";
import { BrowserPackageMountFilesystem } from "./mount.js";
import type { GitSnapshotMountInput } from "./mount.js";
import type { RunMatPackageCacheProvider } from "./provider-types.js";

export interface BrowserProjectResolveOptions {
  target: string;
  groups: Array<"runtime" | "development" | "test">;
  root_features: string[];
  host_capabilities: string[];
  git_intent: GitAcquisitionIntent;
  git_policy: GitAcquisitionPolicy;
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
  source_inventories: unknown[];
}

export interface BrowserProjectResolverNative {
  resolveProject(
    request: BrowserProjectResolveRequest,
    provider: {
      packageCache: RunMatPackageCacheProvider;
      fetchGitInventory(plan: GitAcquisitionPlan): Promise<unknown>;
      mountGitSnapshot(snapshot: GitSnapshotWire): string;
    },
    filesystem: RunMatFilesystemProvider
  ): Promise<BrowserResolvedProject>;
}

export interface BrowserProjectResolverConfig {
  native: BrowserProjectResolverNative;
  filesystem: RunMatFilesystemProvider;
  packageCache: RunMatPackageCacheProvider;
  gitGateway: ServerGitGatewayOptions;
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

  constructor(private readonly config: BrowserProjectResolverConfig) {
    this.filesystem = new BrowserPackageMountFilesystem(config.filesystem);
  }

  resolve(request: BrowserProjectResolveRequest): Promise<BrowserResolvedProject> {
    return this.config.native.resolveProject(
      request,
      {
        packageCache: this.config.packageCache,
        fetchGitInventory: (plan) =>
          fetchGitTreeInventoryWire(
            {
              repository: plan.repository,
              selector: plan.selector,
              subdir: plan.subdir
            },
            this.config.gitGateway
          ),
        mountGitSnapshot: (snapshot) =>
          this.filesystem.register(
            snapshot as unknown as GitSnapshotMountInput,
            this.config.packageCache
          )
      },
      this.filesystem
    );
  }
}
