import { afterEach, describe, expect, it, vi } from "vitest";
import { createRemoteFsProvider } from "./remote.js";
import { encodeWorkspace, decodeWorkspace } from "./workspace-codec.js";
import type { WorkspaceCaptureRequest } from "../generated/fea-workspace-contracts.js";
const capture: WorkspaceCaptureRequest = { root: "/", excluded_subtrees: [], max_files: 100, max_bytes: 10000 };
const revision = new Array(32).fill(1);
afterEach(() => vi.unstubAllGlobals());
describe("remote workspace transport", () => {
  it("reads pinned versions without forwarding project credentials to signed storage", async () => {
    const fetch = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ downloadUrl: "https://storage.example/immutable" }), { headers: { "content-type": "application/json" } }))
      .mockResolvedValueOnce(new Response(new Uint8Array([1, 2, 3])));
    vi.stubGlobal("fetch", fetch);
    const provider = createRemoteFsProvider({ baseUrl: "https://api.example/v1/projects/project", authToken: "token" });
    await expect(provider.readVersion!("/main.m", "version", 3, "snapshot")).resolves.toEqual(new Uint8Array([1, 2, 3]));
    const request = new URL(fetch.mock.calls[0]![0]);
    expect(request.pathname).toBe("/v1/projects/project/fs/read");
    expect(Object.fromEntries(request.searchParams)).toEqual({ path: "/main.m", versionId: "version", snapshotId: "snapshot", offset: "0", length: "3" });
    expect(fetch.mock.calls[0]![1].headers.Authorization).toBe("Bearer token");
    expect(fetch.mock.calls[1]![1].headers).toEqual({});
  });
  it("rejects missing and oversized pinned inputs without a live-read fallback", async () => {
    const cancel = vi.fn();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(new Uint8Array([1, 2, 3, 4])); }, cancel,
    });
    const fetch = vi.fn()
      .mockResolvedValueOnce(new Response("missing", { status: 404 }))
      .mockResolvedValueOnce(new Response(stream));
    vi.stubGlobal("fetch", fetch);
    const provider = createRemoteFsProvider({ baseUrl: "https://api.example/v1/projects/project" });
    await expect(provider.readVersion!("/main.m", "missing", 3, "snapshot")).rejects.toThrow(/404/);
    await expect(provider.readVersion!("/main.m", "oversized", 3, "snapshot")).rejects.toThrow(/length/);
    expect(cancel).toHaveBeenCalledOnce();
    expect(fetch).toHaveBeenCalledTimes(2);
  });
  it("does not mistake an authorization failure containing 'not found' for a missing file", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("authorization not found", { status: 403 })));
    const provider = createRemoteFsProvider({ baseUrl: "https://api.example/v1/projects/project" });
    await expect(provider.readMany!(["/main.m"])).rejects.toThrow(/403/);
  });
  it("uses the existing protected project transport and closed binary contract", async () => {
    const fetch = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      expect(init?.method).toBe("POST");
      expect(init?.headers).toEqual(expect.objectContaining({ Authorization: "Bearer token", "X-RunMat-Client": "remote-fs", "X-Project-Policy": "policy", "Content-Type": "application/cbor" }));
      expect(decodeWorkspace(new Uint8Array(init?.body as ArrayBuffer))).toEqual(capture);
      return new Response(encodeWorkspace({ root: "/", excluded_subtrees: [], revision,
        files: [{ path: "main.fea", version: "v", bytes: new Uint8Array([128,255]), readonly: true }] }));
    });
    vi.stubGlobal("fetch", fetch);
    const provider = createRemoteFsProvider({ baseUrl: "https://api.example/v1/projects/project", authToken: "token", headers: { "X-Project-Policy": "policy" } });
    const snapshot = await provider.captureWorkspace!(capture);
    expect(fetch.mock.calls[0]?.[0]).toBe("https://api.example/v1/projects/project/collaboration/workspace/capture");
    expect(snapshot.files[0]?.bytes).toEqual(new Uint8Array([128,255]));
    expect(snapshot.files[0]?.readonly).toBe(true);
  });
  it("propagates authorization failures and rejects unknown response variants", async () => {
    const fetch = vi.fn().mockResolvedValueOnce(new Response("denied", { status: 403 })).mockResolvedValueOnce(new Response(encodeWorkspace({ type: "overwrite_without_check" })));
    vi.stubGlobal("fetch", fetch);
    const provider = createRemoteFsProvider({ baseUrl: "https://api.example/v1/projects/project" });
    await expect(provider.captureWorkspace!(capture)).rejects.toThrow(/403/);
    const id = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] as const;
    const digest = [...revision] as [number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number,number];
    await expect(provider.publishWorkspace!({ id: [...id], root: "/", excluded_subtrees: [], expected_revision: digest, edits: [] })).rejects.toThrow();
    expect(fetch).toHaveBeenCalledTimes(2);
  });
});
