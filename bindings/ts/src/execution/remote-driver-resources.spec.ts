import { describe, expect, it } from "vitest";
import {
  remoteDriverMemoryBytes,
  REMOTE_DRIVER_COMPLETION_CONTROL_BYTES,
} from "./remote-driver-resources.js";

describe("remote driver parent capacity", () => {
  it("preserves the exact child budget and refuses inexact or overflowing requests", () => {
    const child = 1024 * 1024 * 1024;
    expect(remoteDriverMemoryBytes(child) - REMOTE_DRIVER_COMPLETION_CONTROL_BYTES).toBe(child);
    expect(remoteDriverMemoryBytes(0)).toBe(REMOTE_DRIVER_COMPLETION_CONTROL_BYTES);
    const largest = Number.MAX_SAFE_INTEGER - REMOTE_DRIVER_COMPLETION_CONTROL_BYTES;
    expect(remoteDriverMemoryBytes(largest)).toBe(Number.MAX_SAFE_INTEGER);
    for (const invalid of [-1, 0.5, NaN, Infinity, largest + 1]) {
      expect(() => remoteDriverMemoryBytes(invalid)).toThrow(RangeError);
    }
  });
});
