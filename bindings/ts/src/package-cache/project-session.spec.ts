import { describe, expect, it, vi } from "vitest";
import { createInMemoryFsProvider } from "../fs/memory.js";
import {
  BrowserProjectSession,
  type BrowserProjectSessionNative
} from "./project-session.js";
import type {
  PackageCacheSnapshot,
  RunMatPackageCacheProvider
} from "./provider-types.js";

describe("BrowserProjectSession", () => {
  it("replays the lock, installs the Rust handoff, and writes a generated lock", async () => {
    const filesystem = workspaceFilesystem({
      "/workspace/runmat.toml": '[package]\nname = "demo"\n',
      "/workspace/runmat.lock": "existing-lock"
    });
    const installProjectHandoff = vi.fn(async () => ({ graph_digest: "sha256:graph" }));
    const clearProjectHandoff = vi.fn(async () => {});
    const setFsProvider = vi.fn(async () => {});
    const native = nativeBindings();
    native.decodePackageLock = vi.fn(() => ({ revision: "existing" }));
    native.encodePackageLock = vi.fn(() => "canonical-lock\n");
    native.resolveProject = vi.fn(async (request) => {
      expect(request.existingLock).toEqual({ revision: "existing" });
      return {
        frozen: { graph: "frozen" },
        lock: { revision: "generated" },
        lock_decision: "write-generated",
        acquired_git_sources: [],
        acquired_server_sources: [],
        acquired_registry_sources: [],
        source_inventories: []
      };
    });
    const project = new BrowserProjectSession({
      native,
      session: {
        setFsProvider,
        installProjectHandoff,
        clearProjectHandoff,
        async projectRevision() {
          return { graph_digest: "sha256:graph" };
        }
      },
      filesystem,
      packageCache: cacheProvider(),
      gitGateway: { baseUrl: "https://api.runmat.test" }
    });

    const resolution = await project.resolve(request());

    expect(native.handoffFromFrozenProject).toHaveBeenCalledWith({ graph: "frozen" });
    expect(setFsProvider).toHaveBeenCalledWith(project.filesystem);
    expect(installProjectHandoff).toHaveBeenCalledWith({ handoff: "validated" });
    expect(new TextDecoder().decode(await filesystem.readFile("/workspace/runmat.lock")))
      .toBe("canonical-lock\n");
    expect(resolution).toMatchObject({
      revision: { graph_digest: "sha256:graph" },
      lockPath: "/workspace/runmat.lock",
      lockWritten: true
    });
    expect(await project.sourceRevision()).toEqual({ graph_digest: "sha256:graph" });

    await project.dispose();
    expect(clearProjectHandoff).toHaveBeenCalledOnce();
    expect(setFsProvider).toHaveBeenLastCalledWith(filesystem);
  });

  it("supports locked offline reload without writing the lock", async () => {
    const filesystem = workspaceFilesystem({
      "/workspace/runmat.toml": '[package]\nname = "demo"\n',
      "/workspace/runmat.lock": "cached-lock"
    });
    const native = nativeBindings();
    native.resolveProject = vi.fn(async (request) => {
      expect(request.options.source_policy).toEqual({
        locked: true,
        offline: true
      });
      expect(request.existingLock).toEqual({ decoded: "cached-lock" });
      return resolved("use-existing");
    });
    const project = projectSession(filesystem, native);

    const resolution = await project.resolve({
      ...request(),
      options: {
        ...request().options,
        source_policy: { locked: true, offline: true }
      }
    });

    expect(resolution.lockWritten).toBe(false);
    expect(native.encodePackageLock).not.toHaveBeenCalled();
    expect(new TextDecoder().decode(await filesystem.readFile("/workspace/runmat.lock")))
      .toBe("cached-lock");
    await project.dispose();
  });

  it("surfaces cache status and GC through portable Rust policy", async () => {
    const filesystem = workspaceFilesystem({
      "/workspace/runmat.toml": '[package]\nname = "demo"\n'
    });
    const native = nativeBindings();
    const project = projectSession(filesystem, native);

    await expect(project.cacheStatus()).resolves.toMatchObject({ object_count: 3 });
    await expect(project.collectCache(1024n, 60_000n)).resolves.toEqual({
      delete: ["sha256:stale"],
      reclaim_bytes: 128
    });
    await expect(project.collectCache(-1n)).rejects.toThrow("non-negative");
    await project.dispose();
  });

  it("clears the handoff, mounts, and lease when lock persistence fails", async () => {
    const filesystem = workspaceFilesystem({
      "/workspace/runmat.toml": '[package]\nname = "demo"\n'
    });
    filesystem.writeFile = vi.fn(async () => {
      throw new Error("quota exceeded");
    });
    const native = nativeBindings();
    native.resolveProject = vi.fn(async () => ({
      ...resolved("write-generated"),
      cache_lease: {
        id: "lease",
        owner: "browser",
        objects: [],
        acquired_at_ms: 1,
        expires_at_ms: 2,
        generation: 0
      }
    }));
    const release = native.packageCacheReleaseLease as ReturnType<typeof vi.fn>;
    const clearProjectHandoff = vi.fn(async () => {});
    const project = new BrowserProjectSession({
      native,
      session: {
        installProjectHandoff: vi.fn(async () => ({})),
        clearProjectHandoff,
        async projectRevision() {
          return null;
        }
      },
      filesystem,
      packageCache: cacheProvider(),
      gitGateway: { baseUrl: "https://api.runmat.test" }
    });

    await expect(project.resolve(request())).rejects.toThrow("quota exceeded");
    expect(clearProjectHandoff).toHaveBeenCalledOnce();
    expect(release).toHaveBeenCalledOnce();
    await project.dispose();
    expect(release).toHaveBeenCalledOnce();
  });
});

function projectSession(
  filesystem: ReturnType<typeof createInMemoryFsProvider>,
  native: BrowserProjectSessionNative
): BrowserProjectSession {
  return new BrowserProjectSession({
    native,
    session: {
      async installProjectHandoff() {
        return {};
      },
      async clearProjectHandoff() {},
      async projectRevision() {
        return {};
      }
    },
    filesystem,
    packageCache: cacheProvider(),
    gitGateway: { baseUrl: "https://api.runmat.test" }
  });
}

function workspaceFilesystem(
  files: Record<string, string>
): ReturnType<typeof createInMemoryFsProvider> {
  const filesystem = createInMemoryFsProvider();
  filesystem.createDir?.("/workspace");
  for (const [path, value] of Object.entries(files)) {
    filesystem.writeFile(path, new TextEncoder().encode(value));
  }
  return filesystem;
}

function request() {
  return {
    manifestPath: "/workspace/runmat.toml",
    options: {
      target: "wasm32-unknown-unknown",
      default_server_origin: "https://api.runmat.test",
      default_registry_index: "https://api.runmat.test",
      groups: ["runtime" as const],
      root_features: [],
      host_capabilities: ["browser-filesystem", "network", "worker"],
      source_intent: "execute" as const,
      source_policy: {}
    }
  };
}

function resolved(lockDecision: "use-existing" | "write-generated") {
  return {
    frozen: { graph: "frozen" },
    lock: { lock: "canonical" },
    lock_decision: lockDecision,
    acquired_git_sources: [],
    acquired_server_sources: [],
    acquired_registry_sources: [],
    source_inventories: []
  };
}

function nativeBindings(): BrowserProjectSessionNative {
  return {
    resolveProject: vi.fn(async () => resolved("use-existing")),
    packageCacheRenewLease: vi.fn(async (_provider, lease) => lease),
    packageCacheReleaseLease: vi.fn(async () => {}),
    decodePackageLock: vi.fn((input) => ({ decoded: input })),
    encodePackageLock: vi.fn(() => "canonical-lock\n"),
    handoffFromFrozenProject: vi.fn(() => ({ handoff: "validated" })),
    packageCacheStatus: vi.fn(async () => ({
      schema_version: 1,
      object_count: 3,
      objects_by_kind: { blob: 1, tree: 1, "source-index": 1 },
      logical_bytes: 256,
      stored_payload_bytes: 128,
      pin_count: 0,
      lease_count: 0,
      corruption_count: 0,
      materialization_count: 0
    })),
    packageCacheGc: vi.fn(async () => ({
      delete: ["sha256:stale"],
      reclaim_bytes: 128
    }))
  };
}

function cacheProvider(): RunMatPackageCacheProvider {
  let snapshot: PackageCacheSnapshot | null = null;
  return {
    async snapshot() {
      return snapshot;
    },
    async initialize(initial) {
      snapshot = initial;
      return initial;
    },
    async commit(transaction) {
      snapshot = {
        revision: Number(transaction.expected_revision) + 1,
        state: transaction.next_state
      };
      return { outcome: "committed", revision: snapshot.revision };
    },
    async readObjectBytes() {
      return null;
    }
  };
}
