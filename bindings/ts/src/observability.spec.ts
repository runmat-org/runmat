import { afterEach, describe, expect, it, vi } from "vitest";
import {
  __internals, initRunMat, sendOpenTelemetryTest, flushOpenTelemetry,
  shutdownOpenTelemetry, openTelemetryStats,
} from "./index.js";
import type { OpenTelemetryExportStats } from "./observability.js";

type NativeModule = NonNullable<Parameters<typeof __internals.setNativeModuleOverride>[0]>;
const receipt: OpenTelemetryExportStats = {
  droppedRecords: 0, failedBatches: 0, exportedSpans: 1, exportedLogs: 1, collectorWarnings: 0, queuedBytes: 0,
};

afterEach(() => __internals.setNativeModuleOverride(null));

describe("public OpenTelemetry SDK boundary", () => {
  it("awaits acknowledged test, flush and shutdown, and exposes actual counters", async () => {
    const test = vi.fn(async () => receipt);
    const flush = vi.fn(async () => receipt);
    const shutdown = vi.fn(async () => receipt);
    const stats = vi.fn(() => receipt);
    __internals.setNativeModuleOverride({
      default: async () => {}, stringElementWireSchemaVersion: () => 1,
      sendOpenTelemetryTest: test, flushOpenTelemetry: flush,
      shutdownOpenTelemetry: shutdown, openTelemetryStats: stats,
    } as unknown as NativeModule);
    await expect(sendOpenTelemetryTest()).resolves.toEqual(receipt);
    await expect(flushOpenTelemetry()).resolves.toEqual(receipt);
    await expect(shutdownOpenTelemetry()).resolves.toEqual(receipt);
    await expect(openTelemetryStats()).resolves.toEqual(receipt);
    for (const operation of [test, flush, shutdown, stats]) expect(operation).toHaveBeenCalledOnce();
  });

  it("preserves collector rejection instead of synthesizing queued success", async () => {
    const failure = new Error("OTLP browser request failed; check collector CORS, network and TLS");
    __internals.setNativeModuleOverride({
      default: async () => {}, stringElementWireSchemaVersion: () => 1,
      sendOpenTelemetryTest: async () => { throw failure; },
    } as unknown as NativeModule);
    await expect(sendOpenTelemetryTest()).rejects.toBe(failure);
    await expect(flushOpenTelemetry()).rejects.toThrow("does not support flushOpenTelemetry");
  });

  it("passes startup-only headers separately and initializes policy before plot surfaces", async () => {
    const order: string[] = [];
    const session = {
      executeRequest: async () => { throw new Error("not exercised"); }, resetSession() {},
      stats: () => ({}), clearWorkspace() {}, gpuStatus: () => ({ requested: false, active: false }),
    };
    const initialize = vi.fn(async (_options: unknown) => { order.push("session-policy"); return session; });
    __internals.setNativeModuleOverride({
      default: async () => {}, stringElementWireSchemaVersion: () => 1,
      initRunMat: initialize, createPlotSurface: async () => { order.push("plot"); return 1; },
    } as unknown as NativeModule);
    const headers = { authorization: "Bearer runtime-only-test-value" };
    await initRunMat({ enableGpu: false, openTelemetryHeaders: headers, plotCanvas: {} as HTMLCanvasElement,
      fsProvider: { readFile: () => new Uint8Array(), writeFile() {}, removeFile() {},
        metadata: () => ({ fileType: "file", len: 0, readonly: false }), readDir: () => [] } });
    expect(order).toEqual(["session-policy", "plot"]);
    expect(initialize.mock.calls[0]?.[0]).toMatchObject({ openTelemetryHeaders: headers });
  });
});
