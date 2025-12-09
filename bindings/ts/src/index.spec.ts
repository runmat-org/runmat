import { describe, expect, it, vi } from "vitest";
import { __internals, type RunMatSnapshotSource } from "./index.js";

async function readStream(stream: ReadableStream<Uint8Array> | undefined): Promise<number[]> {
  if (!stream) {
    return [];
  }
  const reader = stream.getReader();
  const chunks: number[] = [];
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    if (value) {
      chunks.push(...value);
    }
  }
  return chunks;
}

describe("resolveSnapshotSource", () => {
  it("prefers inline bytes", async () => {
    const source: RunMatSnapshotSource = { bytes: new Uint8Array([1, 2, 3]) };
    const resolved = await __internals.resolveSnapshotSource(source);
    expect(Array.from(resolved.bytes ?? [])).toEqual([1, 2, 3]);
    expect(resolved.stream).toBeUndefined();
  });

  it("invokes custom fetcher", async () => {
    const fetcher = vi.fn(async () => new Uint8Array([9, 9, 9]));
    const source: RunMatSnapshotSource = { fetcher };
    const resolved = await __internals.resolveSnapshotSource(source);
    expect(fetcher).toHaveBeenCalledOnce();
    expect(Array.from(resolved.bytes ?? [])).toEqual([9, 9, 9]);
  });

  it("passes through readable streams from fetcher responses", async () => {
    const payload = new Uint8Array([5, 6, 7]);
    const response = new Response(payload);
    const source: RunMatSnapshotSource = {
      fetcher: vi.fn(async () => response)
    };
    const resolved = await __internals.resolveSnapshotSource(source);
    expect(resolved.stream).toBeDefined();
    const values = await readStream(resolved.stream as ReadableStream<Uint8Array>);
    expect(values).toEqual([5, 6, 7]);
  });

  it("fetches from URL when provided", async () => {
    const payload = new Uint8Array([4, 5, 6]);
    const mockFetch = vi.fn(async () => new Response(payload));
    // @ts-expect-error override global fetch for test
    globalThis.fetch = mockFetch;
    const result = await __internals.fetchSnapshotFromUrl("https://example.com/snapshot.bin");
    expect(mockFetch).toHaveBeenCalledOnce();
    expect(Array.from(result)).toEqual([4, 5, 6]);
  });
});

describe("coerceFigureError", () => {
  it("wraps structured payloads", () => {
    const payload = {
      code: "InvalidHandle",
      message: "figure missing",
      handle: 7
    };
    const err = __internals.coerceFigureError(payload);
    expect(err.code).toBe("InvalidHandle");
    expect(err.handle).toBe(7);
    expect(err.message).toBe("figure missing");
  });

  it("defaults unknown errors to code 'Unknown'", () => {
    const original = new Error("boom");
    const err = __internals.coerceFigureError(original);
    expect(err.code).toBe("Unknown");
    expect(err.message).toBe("boom");
  });
});

describe("normalizeResumeInputValue", () => {
  it("coerces scalar inputs into line payloads", () => {
    expect(__internals.normalizeResumeInputValue("abc")).toEqual({
      kind: "line",
      value: "abc"
    });
    expect(__internals.normalizeResumeInputValue(42)).toEqual({
      kind: "line",
      value: "42"
    });
    expect(__internals.normalizeResumeInputValue(null)).toEqual({
      kind: "line",
      value: ""
    });
  });

  it("honors keyPress payloads", () => {
    expect(
      __internals.normalizeResumeInputValue({ kind: "keyPress" })
    ).toEqual({ kind: "keyPress" });
  });

  it("propagates error payloads", () => {
    expect(
      __internals.normalizeResumeInputValue({ error: "cancelled" })
    ).toEqual({ error: "cancelled" });
  });
});
