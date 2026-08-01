import { describe, expect, it, vi } from "vitest";
import { createInMemoryFsProvider } from "../fs/memory.js";
import { BrowserProjectResolver } from "./browser-resolver.js";
import { invalidateBrowserPrivatePackageArtifacts } from "./private-artifact-lifecycle.js";
import type { RunMatPackageCacheProvider } from "./provider-types.js";

describe("BrowserProjectResolver", () => {
  it("composes authenticated transport, cache, and immutable filesystem mounts", async () => {
    vi.useFakeTimers();
    const cache = cacheProvider();
    const fetcher = vi.fn(async () =>
      new Response(
        JSON.stringify({
          commit: "0123456789abcdef0123456789abcdef01234567",
          entries: []
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      )
    );
    let mountedRoot = "";
    const releaseLease = vi.fn(async () => {});
    const renewLease = vi.fn(async (_provider, lease) => ({
      ...lease,
      expires_at_ms: 240_001
    }));
    const resolver = new BrowserProjectResolver({
      native: {
        async resolveProject(request, provider, filesystem) {
          const inventory = await provider.fetchGitInventory({
            repository: "https://example.com/acme/helper.git",
            selector: { kind: "rev", value: "0123456789abcdef0123456789abcdef01234567" },
            subdir: ".",
            allow_network: true,
            lock_action: "write"
          });
          expect(inventory).toMatchObject({ commit: expect.any(String) });
          mountedRoot = provider.mountPackageSnapshot({
            source: {
              repository: "https://example.com/acme/helper.git",
              commit: {
                algorithm: "sha1",
                hex: "0123456789abcdef0123456789abcdef01234567"
              },
              subdir: ".",
              tree_digest: `sha256:${"a".repeat(64)}`
            },
            tree: {
              digest: `sha256:${"a".repeat(64)}`,
              entries: [
                {
                  path: "src",
                  kind: "directory",
                  byte_len: 0,
                  executable: false
                },
                {
                  path: "src/helper.m",
                  kind: "file",
                  digest: `sha256:${"b".repeat(64)}`,
                  byte_len: 4,
                  executable: false
                }
              ]
            },
            blobs: []
          });
          expect((await filesystem.metadata(mountedRoot)).readonly).toBe(true);
          expect(await filesystem.readDir(`${mountedRoot}/src`)).toHaveLength(1);
          return {
            frozen: {},
            lock: {},
            lock_decision: "write-generated",
            acquired_git_sources: [],
            acquired_server_sources: [],
            acquired_registry_sources: [],
            source_inventories: [],
            cache_lease: {
              id: "browser-test-graph",
              owner: "browser-test",
              objects: [`sha256:${"a".repeat(64)}`],
              acquired_at_ms: 1,
              expires_at_ms: 120_001,
              generation: 0
            }
          };
        },
        packageCacheRenewLease: renewLease,
        packageCacheReleaseLease: releaseLease
      },
      filesystem: createInMemoryFsProvider(),
      packageCache: cache,
      gitGateway: {
        baseUrl: "https://api.runmat.test",
        authToken: "secret",
        fetch: fetcher as typeof fetch
      }
    });

    const result = await resolver.resolve({
      manifestPath: "/workspace/runmat.toml",
      options: {
        target: "wasm32-unknown-unknown",
        default_server_origin: "https://api.runmat.com",
        default_registry_index: "https://api.runmat.com",
        groups: ["runtime"],
        root_features: [],
        host_capabilities: ["browser-filesystem", "worker"],
        source_intent: "execute",
        source_policy: {}
      }
    });
    expect(result.lock_decision).toBe("write-generated");
    expect(mountedRoot).toMatch(/^\/__runmat\/packages\/sha256_/);
    expect(fetcher).toHaveBeenCalledOnce();
    const request = fetcher.mock.calls[0][1] as RequestInit;
    expect(request.headers).toMatchObject({ authorization: "Bearer secret" });
    await vi.advanceTimersByTimeAsync(40_000);
    expect(renewLease).toHaveBeenCalledOnce();
    await resolver.dispose();
    expect(releaseLease).toHaveBeenCalledOnce();
    vi.useRealTimers();
  });

  it("keeps decrypted private package bytes volatile and removes the mount on dispose", async () => {
    const bytes = new TextEncoder().encode("secret_function");
    const digest = await sha256(bytes);
    const commit = vi.fn(async () => {
      throw new Error("private plaintext must not enter the persistent cache");
    });
    let root = "";
    const resolver = new BrowserProjectResolver({
      native: {
        async resolveProject(_request, provider) {
          root = provider.mountPrivatePackageSnapshot({
            source: {
              repository: "registry:acme/private",
              commit: {
                algorithm: "sha1",
                hex: "0123456789abcdef0123456789abcdef01234567"
              },
              subdir: ".",
              tree_digest: `sha256:${"a".repeat(64)}`
            },
            tree: {
              digest: `sha256:${"a".repeat(64)}`,
              entries: [
                {
                  path: "secret.m",
                  kind: "file",
                  digest,
                  byte_len: bytes.byteLength,
                  executable: false
                }
              ]
            },
            blobs: [{ digest, bytes }]
          });
          return resolvedWithoutLease();
        },
        async packageCacheRenewLease(_provider, lease) {
          return lease;
        },
        async packageCacheReleaseLease() {}
      },
      filesystem: createInMemoryFsProvider(),
      packageCache: { ...cacheProvider(), commit },
      gitGateway: { baseUrl: "https://api.runmat.test" }
    });

    await resolver.resolve(request());
    expect(Array.from(bytes)).toEqual(Array(bytes.byteLength).fill(0));
    expect(new TextDecoder().decode(await resolver.filesystem.readFile(`${root}/secret.m`)))
      .toBe("secret_function");
    expect(commit).not.toHaveBeenCalled();
    invalidateBrowserPrivatePackageArtifacts();
    await expect(resolver.filesystem.readFile(`${root}/secret.m`)).rejects.toThrow();
    await resolver.dispose();
    await expect(
      resolver.filesystem.readFile(`/__runmat/packages/sha256_${"a".repeat(64)}/secret.m`)
    ).rejects.toThrow();
  });

  it("removes a partially composed private mount when resolution fails", async () => {
    const bytes = new TextEncoder().encode("secret_function");
    const digest = await sha256(bytes);
    let root = "";
    const resolver = new BrowserProjectResolver({
      native: {
        async resolveProject(_request, provider) {
          root = provider.mountPrivatePackageSnapshot({
            source: { tree_digest: `sha256:${"c".repeat(64)}` },
            tree: {
              digest: `sha256:${"c".repeat(64)}`,
              entries: [
                {
                  path: "secret.m",
                  kind: "file",
                  digest,
                  byte_len: bytes.byteLength,
                  executable: false
                }
              ]
            },
            blobs: [{ digest, bytes }]
          });
          throw new Error("verification failed after acquisition");
        },
        async packageCacheRenewLease(_provider, lease) {
          return lease;
        },
        async packageCacheReleaseLease() {}
      },
      filesystem: createInMemoryFsProvider(),
      packageCache: cacheProvider(),
      gitGateway: { baseUrl: "https://api.runmat.test" }
    });

    await expect(resolver.resolve(request())).rejects.toThrow("verification failed");
    await expect(resolver.filesystem.readFile(`${root}/secret.m`)).rejects.toThrow();
    await resolver.dispose();
  });
});

function request() {
  return {
    manifestPath: "/workspace/runmat.toml",
    options: {
      target: "wasm32-unknown-unknown",
      default_server_origin: "https://api.runmat.com",
      default_registry_index: "https://api.runmat.com",
      groups: ["runtime" as const],
      root_features: [],
      host_capabilities: ["browser-filesystem", "worker"],
      source_intent: "execute" as const,
      source_policy: {}
    }
  };
}

function resolvedWithoutLease() {
  return {
    frozen: {},
    lock: {},
    lock_decision: "write-generated" as const,
    acquired_git_sources: [],
    acquired_server_sources: [],
    acquired_registry_sources: [],
    source_inventories: []
  };
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const value = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return `sha256:${Array.from(value, (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

function cacheProvider(): RunMatPackageCacheProvider {
  return {
    async snapshot() {
      return null;
    },
    async initialize(initial) {
      return initial;
    },
    async commit(transaction) {
      return {
        outcome: "committed",
        revision: Number(transaction.expected_revision) + 1
      };
    },
    async readObjectBytes() {
      return null;
    }
  };
}
