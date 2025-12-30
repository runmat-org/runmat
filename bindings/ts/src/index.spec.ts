import { describe, expect, it, vi, afterEach } from "vitest";
import * as defaultFs from "./fs/default.js";
import {
  __internals,
  initRunMat,
  deregisterFigureCanvas,
  deregisterPlotCanvas,
  renderFigureImage,
  createWorkspaceHoverProvider,
  createFusionPlanAdapter,
  type RunMatSessionHandle,
  type RunMatFilesystemProvider,
  type RunMatSnapshotSource,
  type ExecuteResult,
  type GpuStatus,
  type SessionStats,
  type PendingStdinRequest,
  type InputRequest
} from "./index.js";

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

const defaultStats: SessionStats = {
  totalExecutions: 0,
  jitCompiled: 0,
  interpreterFallback: 0,
  totalExecutionTimeMs: 0,
  averageExecutionTimeMs: 0
};

function createExecuteResult(overrides: Partial<ExecuteResult> = {}): ExecuteResult {
  return {
    ...baseExecuteResult,
    ...overrides,
    workspace: overrides.workspace ?? baseExecuteResult.workspace
  };
}

function createSessionHandleMock(
  overrides: Partial<RunMatSessionHandle> = {}
): RunMatSessionHandle {
  const stub: Partial<RunMatSessionHandle> = {
    execute: vi.fn(async () => baseExecuteResult),
    resetSession: vi.fn(async () => {}),
    stats: vi.fn(async () => ({ ...defaultStats })),
    clearWorkspace: vi.fn(),
    dispose: vi.fn(),
    telemetryConsent: vi.fn(() => true),
    memoryUsage: vi.fn(async () => ({ bytes: 0, pages: 0 })),
    telemetryClientId: vi.fn(() => undefined),
    gpuStatus: vi.fn(() => ({ requested: false, active: false })),
    cancelExecution: vi.fn(),
    setInputHandler: vi.fn(async () => {}),
    resumeInput: vi.fn(async () => baseExecuteResult),
    pendingStdinRequests: vi.fn(async () => []),
    materializeVariable: vi.fn(async () => undefined),
    setFusionPlanEnabled: vi.fn(),
    ...overrides
  };
  return stub as RunMatSessionHandle;
}

function createHoverHarness(
  options: { session?: RunMatSessionHandle; word?: string } = {}
): {
  controller: ReturnType<typeof createWorkspaceHoverProvider>;
  provider: { provideHover: (model: any, position: any) => Promise<any> | any };
  model: { getWordAtPosition: () => { word: string } | null };
  position: { lineNumber: number; column: number };
} {
  let providerImpl: { provideHover: (model: any, position: any) => Promise<any> | any } | undefined;
  const monacoStub = {
    languages: {
      registerHoverProvider: vi.fn((_language: string, provider) => {
        providerImpl = provider;
        return { dispose: vi.fn() };
      })
    }
  };
  const controller = createWorkspaceHoverProvider({
    monaco: monacoStub,
    language: "matlab",
    session: options.session
  });
  const word = options.word ?? "A";
  const model = {
    getWordAtPosition: () => ({ word })
  };
  const position = { lineNumber: 1, column: 1 };
  if (!providerImpl) {
    throw new Error("Hover provider was not registered");
  }
  return { controller, provider: providerImpl, model, position };
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

  it("preserves render failure details", () => {
    const payload = {
      code: "RenderFailure",
      message: "snapshot failed",
      details: "adapter unavailable"
    };
    const err = __internals.coerceFigureError(payload);
    expect(err.code).toBe("RenderFailure");
    expect(err.details).toBe("adapter unavailable");
  });
});

type NativeModule = Parameters<typeof __internals.setNativeModuleOverride>[0];

interface NativeSession {
  execute(source: string): ExecuteResult;
  resetSession(): void;
  stats(): SessionStats;
  clearWorkspace(): void;
  telemetryConsent(): boolean;
  telemetryClientId?: () => string | undefined;
  memoryUsage?: () => { bytes: number; pages: number };
  gpuStatus(): GpuStatus;
  cancelExecution?: () => void;
  setInputHandler?: (handler: ((req: InputRequest) => unknown) | null) => void;
  resumeInput?: (requestId: string, value: unknown) => ExecuteResult;
  pendingStdinRequests?: () => PendingStdinRequest[];
}

const baseExecuteResult: ExecuteResult = {
  executionTimeMs: 0,
  usedJit: false,
  stdout: [],
  workspace: { full: false, version: 0, values: [] },
  figuresTouched: [],
  warnings: [],
  stdinEvents: []
};

function createMockNativeSession(overrides: Partial<NativeSession> = {}): NativeSession {
  return {
    execute: () => baseExecuteResult,
    resetSession: () => {},
    stats: () => ({
      totalExecutions: 0,
      jitCompiled: 0,
      interpreterFallback: 0,
      totalExecutionTimeMs: 0,
      averageExecutionTimeMs: 0
    }),
    clearWorkspace: () => {},
    telemetryConsent: () => true,
    telemetryClientId: () => undefined,
    memoryUsage: () => ({ bytes: 0, pages: 0 }),
    gpuStatus: () => ({ requested: false, active: false }),
    pendingStdinRequests: () => [],
    ...overrides
  };
}

function createFsProviderStub(): RunMatFilesystemProvider {
  return {
    readFile: () => new Uint8Array(),
    writeFile: () => {},
    removeFile: () => {},
    metadata: () => ({ fileType: "file", len: 0, readonly: false }),
    readDir: () => []
  };
}

describe("initRunMat wiring", () => {
  afterEach(() => {
    __internals.setNativeModuleOverride(null);
    vi.restoreAllMocks();
  });

  it("registers provided fs provider before native init", async () => {
    const order: string[] = [];
    const options: any[] = [];
    const fsProvider = createFsProviderStub();
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: (provider: RunMatFilesystemProvider) => {
        order.push("registerFsProvider");
        expect(provider).toBe(fsProvider);
      },
      initRunMat: async (opts: any) => {
        order.push("initRunMat");
        options.push(opts);
        return createMockNativeSession();
      }
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await initRunMat({ snapshot: { bytes: new Uint8Array([1, 2, 3]) }, fsProvider, enableGpu: false });

    expect(order).toEqual(["registerFsProvider", "initRunMat"]);
    expect(options[0].snapshotBytes).toBeDefined();
  });

  it("passes snapshot bytes and telemetry flag through to native init", async () => {
    const captured: any[] = [];
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: () => {},
      initRunMat: async (opts: any) => {
        captured.push(opts);
        return createMockNativeSession();
      }
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const snapshot = new Uint8Array([9, 9, 9]);
    await initRunMat({ snapshot: { bytes: snapshot }, telemetryConsent: false, enableGpu: false });

    expect(captured).toHaveLength(1);
    expect(captured[0].snapshotBytes).toBe(snapshot);
    expect(captured[0].telemetryConsent).toBe(false);
  });

  it("passes telemetry id and exposes telemetryClientId()", async () => {
    const captured: any[] = [];
    const nativeSession = createMockNativeSession({
      telemetryClientId: () => "cid-native"
    });
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: () => {},
      initRunMat: async (opts: any) => {
        captured.push(opts);
        return nativeSession;
      }
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({
      snapshot: { bytes: new Uint8Array([2]) },
      telemetryId: "cid-host",
      enableGpu: false
    });

    expect(captured).toHaveLength(1);
    expect(captured[0].telemetryId).toBe("cid-host");
    expect(session.telemetryClientId()).toBe("cid-native");
  });

  it("passes scatter/surface overrides to native init", async () => {
    const captured: any[] = [];
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: () => {},
      initRunMat: async (opts: any) => {
        captured.push(opts);
        return createMockNativeSession();
      }
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await initRunMat({
      snapshot: { bytes: new Uint8Array([3]) },
      scatterTargetPoints: 250_000,
      surfaceVertexBudget: 1_000_000,
      enableGpu: false
    });

    expect(captured).toHaveLength(1);
    expect(captured[0].scatterTargetPoints).toBe(250_000);
    expect(captured[0].surfaceVertexBudget).toBe(1_000_000);
  });

  it("disables GPU when navigator.gpu is unavailable", async () => {
    const captured: any[] = [];
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: () => {},
      initRunMat: async (opts: any) => {
        captured.push(opts);
        return createMockNativeSession();
      }
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const originalNavigator = (globalThis as any).navigator;
    Object.defineProperty(globalThis, "navigator", {
      value: {},
      configurable: true
    });
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    await initRunMat({ snapshot: { bytes: new Uint8Array([1]) }, enableGpu: true });

    expect(captured).toHaveLength(1);
    expect(captured[0].enableGpu).toBe(false);
    expect(warnSpy).toHaveBeenCalled();

    warnSpy.mockRestore();
    Object.defineProperty(globalThis, "navigator", {
      value: originalNavigator,
      configurable: true
    });
  });

  it("auto-registers the default filesystem provider when none is supplied", async () => {
    const autoProvider = createFsProviderStub();
    const defaultSpy = vi
      .spyOn(defaultFs, "createDefaultFsProvider")
      .mockResolvedValue(autoProvider);
    const registerSpy = vi.fn();
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: registerSpy,
      initRunMat: async (opts: any) => {
        expect(opts.snapshotBytes).toBeDefined();
        return createMockNativeSession();
      }
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await initRunMat({ snapshot: { bytes: new Uint8Array([7]) }, enableGpu: false });

    expect(defaultSpy).toHaveBeenCalledOnce();
    expect(registerSpy).toHaveBeenCalledWith(autoProvider);
  });

  it("registers plotCanvas before calling initRunMat", async () => {
    const order: string[] = [];
    const registerSpy = vi.fn(async () => {
      order.push("registerPlotCanvas");
    });
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: () => {},
      registerPlotCanvas: registerSpy,
      initRunMat: async () => {
        order.push("initRunMat");
        return createMockNativeSession();
      }
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const canvas = { id: "canvas" } as unknown as HTMLCanvasElement;
    await initRunMat({ snapshot: { bytes: new Uint8Array([1]) }, plotCanvas: canvas, enableGpu: false });

    expect(order).toEqual(["registerPlotCanvas", "initRunMat"]);
    expect(registerSpy).toHaveBeenCalledWith(canvas);
  });

  it("deregisters the default plot canvas when requested", async () => {
    const deregisterSpy = vi.fn();
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: () => {},
      initRunMat: async () => createMockNativeSession(),
      deregisterPlotCanvas: deregisterSpy
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await deregisterPlotCanvas();
    expect(deregisterSpy).toHaveBeenCalledOnce();
  });

  it("deregisters figure-specific canvases", async () => {
    const deregisterSpy = vi.fn();
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: () => {},
      initRunMat: async () => createMockNativeSession(),
      deregisterFigureCanvas: deregisterSpy
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await deregisterFigureCanvas(42);
    expect(deregisterSpy).toHaveBeenCalledWith(42);
  });

  it("surfaces structured errors from registerPlotCanvas", async () => {
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: () => {},
      registerPlotCanvas: async () => {
        const err = new Error("canvas failed") as Error & { code?: string };
        err.code = "PlotCanvas";
        throw err;
      },
      initRunMat: async () => createMockNativeSession()
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await expect(
      initRunMat({ snapshot: { bytes: new Uint8Array([1]) }, plotCanvas: {} as HTMLCanvasElement, enableGpu: false })
    ).rejects.toMatchObject({ code: "PlotCanvas" });
  });

  it("disposes the session and blocks further calls", async () => {
    const disposeSpy = vi.fn();
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: () => {},
      initRunMat: async () =>
        createMockNativeSession({
          dispose: disposeSpy
        })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({ snapshot: { bytes: new Uint8Array([1]) }, enableGpu: false });
    session.dispose();
    expect(disposeSpy).toHaveBeenCalledOnce();
    expect(() => session.telemetryConsent()).toThrow(/disposed/);
    // dispose is idempotent
    session.dispose();
    expect(disposeSpy).toHaveBeenCalledTimes(1);
  });

  it("exposes memory usage stats from the native session", async () => {
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: () => {},
      initRunMat: async () =>
        createMockNativeSession({
          memoryUsage: () => ({ bytes: 1024, pages: 16 })
        })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({ snapshot: { bytes: new Uint8Array([1]) }, enableGpu: false });
    await expect(session.memoryUsage()).resolves.toEqual({ bytes: 1024, pages: 16 });
  });
});

describe("renderFigureImage", () => {
  afterEach(() => {
    __internals.setNativeModuleOverride(null);
    vi.restoreAllMocks();
  });

  it("returns bytes from the native binding", async () => {
    const native: NativeModule = {
      default: async () => {},
      renderFigureImage: vi.fn(async (handle: number | null, width: number, height: number) => {
        expect(handle).toBeNull();
        expect(width).toBe(0);
        expect(height).toBe(0);
        return new Uint8Array([1, 2, 3]);
      })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const bytes = await renderFigureImage();
    expect(Array.from(bytes)).toEqual([1, 2, 3]);
    expect(native.renderFigureImage).toHaveBeenCalledWith(null, 0, 0);
  });

  it("forwards options and surfaces figure errors", async () => {
    const native: NativeModule = {
      default: async () => {},
      renderFigureImage: vi.fn(async () => {
        throw { code: "InvalidHandle", handle: 77 };
      })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await expect(renderFigureImage({ handle: 77, width: 640, height: 480 })).rejects.toMatchObject({
      code: "InvalidHandle",
      handle: 77
    });
    expect(native.renderFigureImage).toHaveBeenCalledWith(77, 640, 480);
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

describe("ExecuteResult passthroughs", () => {
  afterEach(() => {
    __internals.setNativeModuleOverride(null);
    vi.restoreAllMocks();
  });

  it("preserves stdinRequested.waitingMs from the native session", async () => {
    const request = {
      id: "req",
      request: { prompt: ">> ", kind: "line", echo: true },
      waitingMs: 1500
    };
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: () => {},
      initRunMat: async () =>
        createMockNativeSession({
          execute: () => ({
            ...baseExecuteResult,
            stdinRequested: request
          })
        })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({ snapshot: { bytes: new Uint8Array([1]) }, enableGpu: false });
    const result = await session.execute("disp('prompt')");
    expect(result.stdinRequested).toEqual(request);
    expect(result.stdinRequested?.waitingMs).toBe(1500);
  });
});

describe("workspace hover provider", () => {
  it("formats workspace metadata in hover tooltips", async () => {
    const harness = createHoverHarness();
    harness.controller.updateWorkspace({
      full: true,
      version: 1,
      values: [
        {
          name: "A",
          className: "double",
          dtype: "double",
          shape: [2, 2],
          isGpu: false,
          sizeBytes: 64,
          preview: { values: [1, 2, 3], truncated: false },
          residency: "cpu"
        }
      ]
    });
    const hover = await harness.provider.provideHover(harness.model, harness.position);
    expect(hover?.contents[0].value).toContain("Class: `double`");
    expect(hover?.contents[0].value).toContain("Shape: 2×2");
  });

  it("materializes truncated previews only once", async () => {
    const session = createSessionHandleMock({
      materializeVariable: vi.fn(async () => ({
        name: "B",
        className: "double",
        dtype: "double",
        shape: [1, 3],
        isGpu: true,
        residency: "gpu",
        sizeBytes: 24,
        preview: { values: [4, 5, 6], truncated: false },
        valueText: "[4 5 6]",
        valueJson: [4, 5, 6]
      }))
    });
    const harness = createHoverHarness({ session, word: "B" });
    harness.controller.updateWorkspace({
      full: true,
      version: 1,
      values: [
        {
          name: "B",
          className: "double",
          dtype: "double",
          shape: [1, 3],
          isGpu: true,
          sizeBytes: 24,
          preview: { values: [], truncated: true },
          residency: "gpu",
          previewToken: "token-b"
        }
      ]
    });
    const hover = await harness.provider.provideHover(harness.model, harness.position);
    expect(session.materializeVariable).toHaveBeenCalledTimes(1);
    expect(hover?.contents[0].value).toContain("Preview (materialized)");
    await harness.provider.provideHover(harness.model, harness.position);
    expect(session.materializeVariable).toHaveBeenCalledTimes(1);
  });
});

describe("fusion plan adapter", () => {
  it("toggles emission and notifies listeners", () => {
    const session = createSessionHandleMock();
    const onPlanChange = vi.fn();
    const adapter = createFusionPlanAdapter({ session, onPlanChange });
    adapter.setEnabled(true);
    expect(session.setFusionPlanEnabled).toHaveBeenCalledWith(true);
    const listener = vi.fn();
    const unsubscribe = adapter.subscribe(listener);
    const plan = { nodes: [], edges: [], shaders: [], decisions: [] };
    adapter.handleExecutionResult(createExecuteResult({ fusionPlan: plan }));
    expect(onPlanChange).toHaveBeenLastCalledWith(plan);
    expect(listener).toHaveBeenCalledWith(plan);
    adapter.setEnabled(false);
    expect(session.setFusionPlanEnabled).toHaveBeenCalledWith(false);
    expect(adapter.plan).toBeNull();
    expect(listener).toHaveBeenCalledTimes(2);
    unsubscribe();
    adapter.handleExecutionResult(createExecuteResult());
    expect(listener).toHaveBeenCalledTimes(2);
  });
});

describe("materializeVariable wiring", () => {
  afterEach(() => {
    __internals.setNativeModuleOverride(null);
    vi.restoreAllMocks();
  });

  it("forwards string selectors and options to the native session", async () => {
    const spy = vi.fn(() => ({
      name: "A",
      className: "double",
      dtype: "double",
      shape: [1, 1],
      isGpu: false,
      residency: "cpu",
      sizeBytes: 8,
      preview: { values: [1], truncated: false },
      valueText: "1",
      valueJson: 1
    }));
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: () => {},
      initRunMat: async () =>
        createMockNativeSession({
          materializeVariable: spy
        })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({ snapshot: { bytes: new Uint8Array([1]) }, enableGpu: false });
    const materialized = await session.materializeVariable("token-123", { limit: 64 });

    expect(spy).toHaveBeenCalledWith("token-123", { limit: 64 });
    expect(materialized.name).toBe("A");
    expect(materialized.preview?.values).toEqual([1]);
  });

  it("normalizes selector objects and clamps invalid limits", async () => {
    const spy = vi.fn(() => ({
      name: "B",
      className: "double",
      shape: [2, 2],
      dtype: "double",
      isGpu: true,
      residency: "gpu",
      preview: undefined,
      sizeBytes: undefined,
      valueText: "[]",
      valueJson: []
    }));
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: () => {},
      initRunMat: async () =>
        createMockNativeSession({
          materializeVariable: spy
        })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({ snapshot: { bytes: new Uint8Array([9]) }, enableGpu: false });
    await session.materializeVariable({ previewToken: "abc-uuid", name: "ignored" }, { limit: 0 });

    expect(spy).toHaveBeenCalledWith({ previewToken: "abc-uuid" }, {});
  });
});

describe("setFusionPlanEnabled", () => {
  afterEach(() => {
    __internals.setNativeModuleOverride(null);
  });

  it("delegates to the native session", async () => {
    const spy = vi.fn();
    const native: NativeModule = {
      default: async () => {},
      registerFsProvider: () => {},
      initRunMat: async () =>
        createMockNativeSession({
          setFusionPlanEnabled: spy
        })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({ snapshot: { bytes: new Uint8Array([1]) }, enableGpu: false });
    session.setFusionPlanEnabled(true);
    expect(spy).toHaveBeenCalledWith(true);
  });
});
