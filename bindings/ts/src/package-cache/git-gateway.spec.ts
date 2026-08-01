import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchGitTreeInventoryWire } from "./git-gateway.js";
import { __internals, fetchGitSnapshot } from "../index.js";

describe("Git snapshot gateway", () => {
  afterEach(() => {
    __internals.setNativeModuleOverride(null);
  });

  it("sends only the normalized transport request and optional bearer token", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        JSON.stringify({
          commit: "0123456789abcdef0123456789abcdef01234567",
          entries: []
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      )
    );
    const inventory = await fetchGitTreeInventoryWire(
      {
        repository: "https://github.com/runmat-org/runmat",
        selector: { kind: "branch", value: "main" }
      },
      {
        baseUrl: "https://api.runmat.com/",
        authToken: async () => "token",
        fetch: fetcher
      }
    );

    expect(inventory.entries).toEqual([]);
    expect(fetcher).toHaveBeenCalledOnce();
    const [url, init] = fetcher.mock.calls[0]!;
    expect(url).toBe("https://api.runmat.com/v1/packages/git/snapshot");
    expect(init).toMatchObject({
      method: "POST",
      cache: "no-store",
      credentials: "omit",
      headers: {
        authorization: "Bearer token",
        "content-type": "application/json"
      }
    });
    expect(JSON.parse(String(init?.body))).toEqual({
      repository: "https://github.com/runmat-org/runmat",
      selector: { kind: "branch", value: "main" },
      subdir: "."
    });
  });

  it("surfaces status and server detail without accepting credentials in URLs", async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response("unsupported host", { status: 400 }));
    await expect(
      fetchGitTreeInventoryWire(
        {
          repository: "https://user:secret@example.com/repo",
          selector: { kind: "rev", value: "0123456789abcdef0123456789abcdef01234567" }
        },
        { baseUrl: "https://api.runmat.com", fetch: fetcher }
      )
    ).rejects.toThrow("HTTP 400: unsupported host");
  });

  it("hands the untrusted inventory to the portable Rust snapshot builder", async () => {
    const inventory = {
      commit: "0123456789abcdef0123456789abcdef01234567",
      entries: [
        {
          path: "main.m",
          kind: "file" as const,
          bytes: "YW5zd2VyID0gNDI7Cg==",
          executable: false
        }
      ]
    };
    const snapshot = { source: {}, tree: {}, blobs: [] };
    const buildGitSnapshot = vi.fn().mockReturnValue(snapshot);
    __internals.setNativeModuleOverride({
      default: async () => {},
      initRunMat: vi.fn(),
      buildGitSnapshot
    } as never);
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(JSON.stringify(inventory), { status: 200 }));
    const result = await fetchGitSnapshot(
      {
        repository: "https://github.com/runmat-org/runmat",
        selector: { kind: "rev", value: inventory.commit },
        subdir: "package"
      },
      { baseUrl: "https://api.runmat.com", fetch: fetcher }
    );
    expect(result).toBe(snapshot);
    expect(buildGitSnapshot).toHaveBeenCalledWith(
      "https://github.com/runmat-org/runmat",
      "package",
      inventory
    );
  });
});
