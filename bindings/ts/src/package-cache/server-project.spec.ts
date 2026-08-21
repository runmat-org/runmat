import { describe, expect, it, vi } from "vitest";
import {
  fetchServerProjectSnapshot,
  fetchServerProjectSnapshotResponse
} from "./server-project.js";
import type { ServerProjectAcquisitionPlan } from "./server-project.js";

const plan: ServerProjectAcquisitionPlan = {
  service: "https://api.runmat.test",
  project: "proj_0123456789abcdef0123456789abcdef",
  selector: { kind: "tag", value: "main channel" },
  allow_network: true,
  lock_action: "write"
};

describe("Server project snapshot transport", () => {
  it("encodes selectors, omits ambient credentials, and scopes bearer tokens to one origin", async () => {
    const fetcher = vi.fn(async () =>
      new Response(
        JSON.stringify({
          project: plan.project,
          snapshot: "snap_0123456789abcdef0123456789abcdef",
          treeDigest: `sha256:${"a".repeat(64)}`,
          entries: []
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      )
    );
    await fetchServerProjectSnapshot(plan, {
      authenticatedOrigin: "https://api.runmat.test/",
      authToken: "secret",
      fetch: fetcher as typeof fetch
    });
    expect(fetcher).toHaveBeenCalledWith(
      expect.stringContaining("/snapshots/main%20channel"),
      expect.objectContaining({
        credentials: "omit",
        cache: "no-store",
        headers: { authorization: "Bearer secret" }
      })
    );

    await fetchServerProjectSnapshot(
      { ...plan, service: "https://other.runmat.test" },
      {
        authenticatedOrigin: "https://api.runmat.test",
        authToken: "secret",
        fetch: fetcher as typeof fetch
      }
    );
    expect(fetcher.mock.calls[1][1]).toMatchObject({ headers: {} });
  });

  it("refuses insecure origins and disabled network plans before fetching", async () => {
    const fetcher = vi.fn();
    await expect(
      fetchServerProjectSnapshot(
        { ...plan, service: "http://api.runmat.test" },
        { fetch: fetcher as typeof fetch }
      )
    ).rejects.toThrow("HTTPS");
    await expect(
      fetchServerProjectSnapshot(
        { ...plan, allow_network: false },
        { fetch: fetcher as typeof fetch }
      )
    ).rejects.toThrow("disabled");
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("supports conditional and byte-range requests without conflating their outcomes", async () => {
    const notModified = vi.fn(async () =>
      new Response(null, { status: 304, headers: { etag: '"tree"' } })
    );
    await expect(
      fetchServerProjectSnapshotResponse(plan, {
        fetch: notModified as typeof fetch,
        etag: '"tree"'
      })
    ).resolves.toEqual({ kind: "not-modified", etag: '"tree"' });
    expect(notModified.mock.calls[0][1]).toMatchObject({
      headers: { "if-none-match": '"tree"' }
    });

    const partial = vi.fn(async () =>
      new Response(new Uint8Array([1, 2, 3]), {
        status: 206,
        headers: {
          "content-range": "bytes 4-6/20",
          etag: '"tree"'
        }
      })
    );
    const outcome = await fetchServerProjectSnapshotResponse(plan, {
      fetch: partial as typeof fetch,
      range: { start: 4, end: 6 }
    });
    expect(outcome).toMatchObject({
      kind: "partial",
      contentRange: "bytes 4-6/20",
      etag: '"tree"'
    });
    expect(partial.mock.calls[0][1]).toMatchObject({
      headers: { range: "bytes=4-6" }
    });
  });
});
