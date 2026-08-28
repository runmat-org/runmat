import { describe, expect, it, vi } from "vitest";
import {
  fetchRegistryCandidates,
  fetchRegistryRelease,
  type RegistryAcquisitionPlan
} from "./registry.js";

const packageId = "default:acme/tools";

describe("browser registry transport", () => {
  it("scopes credentials to the configured trusted origin", async () => {
    const fetcher = vi.fn(async () =>
      jsonResponse({ candidates: [] })
    );
    await fetchRegistryCandidates(
      {
        source_registry: "default",
        index: "https://packages.runmat.test/index",
        package: packageId,
        allow_network: true
      },
      {
        authenticatedOrigin: "https://packages.runmat.test",
        authToken: "secret",
        fetch: fetcher as typeof fetch
      }
    );
    expect(fetcher).toHaveBeenCalledOnce();
    expect((fetcher.mock.calls[0][1] as RequestInit).headers).toMatchObject({
      authorization: "Bearer secret"
    });

    fetcher.mockClear();
    await fetchRegistryCandidates(
      {
        source_registry: "mirror",
        index: "https://mirror.example.test",
        package: packageId,
        allow_network: true
      },
      {
        authenticatedOrigin: "https://packages.runmat.test",
        authToken: "secret",
        fetch: fetcher as typeof fetch
      }
    );
    expect((fetcher.mock.calls[0][1] as RequestInit).headers).not.toHaveProperty(
      "authorization"
    );
  });

  it("rejects cross-origin artifact URLs before making the artifact request", async () => {
    const fetcher = vi.fn(async () =>
      jsonResponse({
        artifact: {
          byteLen: 1,
          downloadUrl: "https://evil.test/artifact"
        }
      })
    );
    await expect(
      fetchRegistryRelease(plan(), { fetch: fetcher as typeof fetch })
    ).rejects.toThrow("unsafe artifact URL");
    expect(fetcher).toHaveBeenCalledOnce();
  });

  it("enforces signed artifact length while streaming", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          artifact: {
            byteLen: 2,
            downloadUrl: "/artifact"
          }
        })
      )
      .mockResolvedValueOnce(new Response(new Uint8Array([1]), { status: 200 }));
    await expect(
      fetchRegistryRelease(plan(), { fetch: fetcher as typeof fetch })
    ).rejects.toThrow("length differs");
  });
});

function plan(): RegistryAcquisitionPlan {
  return {
    source_registry: "default",
    index: "https://packages.runmat.test",
    package: packageId,
    requirement: "=1.0.0",
    allow_network: true,
    expected: {
      release: "rel_0123456789abcdef0123456789abcdef"
    },
    lock_action: "preserve"
  };
}

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { "content-type": "application/json" }
  });
}
