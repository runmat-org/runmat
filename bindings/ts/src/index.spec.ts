import { describe, expect, it, vi, afterEach } from "vitest";
import * as defaultFs from "./fs/default.js";
import {
  __internals,
  executeProgramArtifact,
  initRunMat,
  renderFigureImage,
  exportFigureScene,
  importFigureScene,
  importFigureSceneFromPath,
  exportWorkspaceState,
  importWorkspaceState,
  resetPlotState,
  resolveRunmatConfig,
  patchRunmatConfig,
  migrateLegacyRunmatConfig,
  migrateLegacyRunmatConfigInto,
  createWorkspaceHoverProvider,
  createFusionPlanAdapter,
  decodePackageLock,
  encodePackageLock,
  handoffFromFrozenProject,
  FEA_ANALYSIS_PROFILES,
  FEA_ANALYSIS_RUN_KINDS,
  FEA_ARTIFACT_MANIFEST_KIND,
  FEA_DATASET_ARTIFACT_KIND,
  FEA_DIAGNOSTICS_ARTIFACT_KIND,
  FEA_DIAGNOSTICS_SCHEMA_VERSION,
  FEA_FIELD_DEFAULT_MATERIALIZE_LIMIT,
  FEA_FIELD_DEFAULT_PAGE_SIZE,
  FEA_FIELD_DESCRIPTORS_ARTIFACT_KIND,
  FEA_FIELD_DESCRIPTORS_SCHEMA_VERSION,
  FEA_OBJECT_ARTIFACT_METADATA_SCHEMA_VERSION,
  FEA_RUN_DATASET_KIND,
  FEA_RUN_DATASET_SCHEMA_VERSION,
  FEA_STUDY_DOCUMENT_OPERATIONS,
  FEA_SUPPORTED_PHYSICS_PROFILES,
  type RunMatSessionHandle,
  type RunMatFilesystemProvider,
  type ExecuteRequest,
  type ExecuteResult,
  type GpuStatus,
  type SessionStats,
  type InputRequest,
  setSignalTraceHandler,
  withSignalTrace
} from "./index.js";

const defaultStats: SessionStats = {
  totalExecutions: 0,
  jitCompiled: 0,
  interpreterFallback: 0,
  totalExecutionTimeMs: 0,
  averageExecutionTimeMs: 0
};

describe("FEA study document contracts", () => {
  it("exports Rust-generated analysis artifact constants", () => {
    expect(FEA_RUN_DATASET_SCHEMA_VERSION).toBe(1);
    expect(FEA_FIELD_DESCRIPTORS_SCHEMA_VERSION).toBe(1);
    expect(FEA_DIAGNOSTICS_SCHEMA_VERSION).toBe(1);
    expect(FEA_OBJECT_ARTIFACT_METADATA_SCHEMA_VERSION).toBe(1);
    expect(FEA_RUN_DATASET_KIND).toBe("finite_element_run_dataset");
    expect(FEA_DATASET_ARTIFACT_KIND).toBe("finite_element_dataset");
    expect(FEA_FIELD_DESCRIPTORS_ARTIFACT_KIND).toBe("finite_element_field_descriptors");
    expect(FEA_DIAGNOSTICS_ARTIFACT_KIND).toBe("finite_element_diagnostics");
    expect(FEA_ARTIFACT_MANIFEST_KIND).toBe("finite_element_artifact_manifest");
    expect(FEA_FIELD_DEFAULT_PAGE_SIZE).toBe(4096);
    expect(FEA_FIELD_DEFAULT_MATERIALIZE_LIMIT).toBe(256);
  });

  it("exports Rust-generated study operation names", () => {
    expect(FEA_STUDY_DOCUMENT_OPERATIONS).toEqual([
      "get_summary",
      "create",
      "add_region",
      "update_region",
      "remove_region",
      "add_material",
      "update_material",
      "assign_material",
      "add_constraint",
      "update_constraint",
      "remove_constraint",
      "add_driving_condition",
      "update_driving_condition",
      "remove_driving_condition",
      "set_mesh",
      "set_outputs",
    ]);
  });

  it("keeps the public FEA physics catalog aligned with Rust-supported profiles", () => {
    expect(FEA_ANALYSIS_PROFILES).toEqual([
      "linear_static_structural",
      "thermo_mechanical_coupled",
      "electro_thermal_coupled",
      "thermal_standalone",
      "modal_structural",
      "acoustic_harmonic",
      "transient_structural",
      "nonlinear_structural",
      "electromagnetic_static",
      "cfd_steady_state",
      "cfd_transient",
      "cht_coupled",
      "fsi_coupled",
    ]);
    expect(FEA_ANALYSIS_RUN_KINDS).toEqual([
      "linear_static",
      "modal",
      "acoustic",
      "thermal",
      "transient",
      "cfd",
      "cht",
      "fsi",
      "nonlinear",
      "electromagnetic",
    ]);
    const catalogProfiles = FEA_SUPPORTED_PHYSICS_PROFILES.map((profile) => profile.profile);
    expect(catalogProfiles).toHaveLength(FEA_ANALYSIS_PROFILES.length);
    expect(new Set(catalogProfiles)).toEqual(new Set(FEA_ANALYSIS_PROFILES));
    expect(new Set(FEA_SUPPORTED_PHYSICS_PROFILES.map((profile) => profile.family))).toEqual(
      new Set(["structural", "modal", "coupled physics", "thermal", "acoustic", "electromagnetic", "CFD"]),
    );
  });

  it("exposes electro-thermal as a first-class coupled FEA profile", () => {
    const profile = FEA_SUPPORTED_PHYSICS_PROFILES.find(
      (entry) => entry.profile === "electro_thermal_coupled"
    );

    expect(profile).toMatchObject({
      label: "Electro-thermal",
      family: "coupled physics",
      target: "coupled electromagnetics and heat transfer",
      value: "resistive heating, temperature, and electrical fields",
    });
    expect(profile?.defaultOutputs.map((output) => output.field)).toEqual([
      "electro_thermal.temperature",
      "electro_thermal.joule_heat",
      "electro_thermal.electric_potential",
      "electro_thermal.current_density",
    ]);
  });
});

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
  executeRequest(request: ExecuteRequest): ExecuteResult;
  resetSession(): void;
  stats(): SessionStats;
  clearWorkspace(): void;
  exportWorkspaceState?: (includeVariables?: string) => Promise<Uint8Array | null>;
  importWorkspaceState?: (state: Uint8Array) => boolean;
  telemetryConsent(): boolean;
  telemetryClientId?: () => string | undefined;
  memoryUsage?: () => { bytes: number; pages: number };
  gpuStatus(): GpuStatus;
  cancelExecution?: () => void;
  setInputHandler?: (handler: ((req: InputRequest) => unknown) | null) => void;
  setLanguageCompat?: (mode: string) => void;
  setErrorNamespace?: (namespace: string) => void;
  installProjectHandoff?: (handoff: unknown) => unknown;
  clearProjectHandoff?: () => void;
  projectRevision?: () => unknown | null;
}

const baseExecuteResult: ExecuteResult = {
  flow: { kind: "no-value" },
  executionTimeMs: 0,
  usedJit: false,
  stdout: [],
  workspace: { full: false, version: 0, values: [], removals: [] },
  figuresTouched: [],
  warnings: [],
  stdinEvents: []
};

function createMockNativeSession(overrides: Partial<NativeSession> = {}): NativeSession {
  return {
    executeRequest: () => baseExecuteResult,
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

  it("forwards dynamic language policy and error namespace updates", async () => {
    const setLanguageCompat = vi.fn();
    const setErrorNamespace = vi.fn();
    __internals.setNativeModuleOverride({
      default: async () => {},
      initRunMat: async () => createMockNativeSession({
        setLanguageCompat,
        setErrorNamespace,
      }),
    } as NativeModule);

    const session = await initRunMat({ enableGpu: false });
    session.setLanguageCompat("runmat");
    session.setErrorNamespace("Acme");

    expect(setLanguageCompat).toHaveBeenCalledWith("runmat");
    expect(setErrorNamespace).toHaveBeenCalledWith("Acme");
  });

  it("passes provided fs provider into native init options", async () => {
    const options: any[] = [];
    const fsProvider = createFsProviderStub();
    const native: NativeModule = {
      default: async () => {},
      initRunMat: async (opts: any) => {
        options.push(opts);
        return createMockNativeSession();
      }
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await initRunMat({ fsProvider, enableGpu: false });

    expect(options[0].fsProvider).toBe(fsProvider);
  });

  it("passes telemetry consent through to native init", async () => {
    const captured: any[] = [];
    const native: NativeModule = {
      default: async () => {},
      initRunMat: async (opts: any) => {
        captured.push(opts);
        return createMockNativeSession();
      }
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await initRunMat({ telemetryConsent: false, enableGpu: false });

    expect(captured).toHaveLength(1);
    expect(captured[0].telemetryConsent).toBe(false);
  });

  it("passes the browser execution host through without interpreting it", async () => {
    const captured: any[] = [];
    const executionHost = {
      capabilities: { topology: "serial" as const, maxWorkers: 1 },
      launch: vi.fn(async () => ({ outcome: "success" })),
      cancel: vi.fn()
    };
    __internals.setNativeModuleOverride({
      default: async () => {},
      initRunMat: async (options: any) => {
        captured.push(options);
        return createMockNativeSession();
      }
    } as NativeModule);

    await initRunMat({ executionHost, enableGpu: false });

    expect(captured[0].executionHost).toBe(executionHost);
  });

  it("delegates exact program execution to the portable wasm export", async () => {
    const execute = vi.fn(async (request) => ({
      outcome: "success",
      request
    }));
    __internals.setNativeModuleOverride({
      default: async () => {},
      initRunMat: async () => createMockNativeSession(),
      executeProgramArtifact: execute
    } as NativeModule);

    await expect(executeProgramArtifact({ schemaVersion: 1 })).resolves.toEqual({
      outcome: "success",
      request: { schemaVersion: 1 }
    });
  });

  it("passes telemetry id and exposes telemetryClientId()", async () => {
    const captured: any[] = [];
    const nativeSession = createMockNativeSession({
      telemetryClientId: () => "cid-native"
    });
    const native: NativeModule = {
      default: async () => {},
      initRunMat: async (opts: any) => {
        captured.push(opts);
        return nativeSession;
      }
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({
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
      initRunMat: async (opts: any) => {
        captured.push(opts);
        return createMockNativeSession();
      }
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await initRunMat({
      scatterTargetPoints: 250_000,
      surfaceVertexBudget: 1_000_000,
      enableGpu: false
    });

    expect(captured).toHaveLength(1);
    expect(captured[0].scatterTargetPoints).toBe(250_000);
    expect(captured[0].surfaceVertexBudget).toBe(1_000_000);
  });

  it("passes log level preference to native init", async () => {
    const captured: any[] = [];
    const native: NativeModule = {
      default: async () => { },
      initRunMat: async (opts: any) => {
        captured.push(opts);
        return createMockNativeSession();
      }
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await initRunMat({
      logLevel: "trace",
      enableGpu: false
    });

    expect(captured).toHaveLength(1);
    expect(captured[0].logLevel).toBe("trace");
  });

  it("disables GPU when navigator.gpu is unavailable", async () => {
    const captured: any[] = [];
    const native: NativeModule = {
      default: async () => {},
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

    await initRunMat({ enableGpu: true });

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
    const captured: any[] = [];
    const native: NativeModule = {
      default: async () => {},
      initRunMat: async (opts: any) => {
        captured.push(opts);
        return createMockNativeSession();
      }
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await initRunMat({ enableGpu: false });

    expect(defaultSpy).toHaveBeenCalledOnce();
    expect(captured).toHaveLength(1);
    expect(captured[0].fsProvider).toBe(autoProvider);
  });

  it("creates a plot surface before calling initRunMat", async () => {
    const order: string[] = [];
    const createSurfaceSpy = vi.fn(async () => {
      order.push("createPlotSurface");
    });
    const native: NativeModule = {
      default: async () => {},
      createPlotSurface: createSurfaceSpy,
      initRunMat: async () => {
        order.push("initRunMat");
        return createMockNativeSession();
      }
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const canvas = { id: "canvas" } as unknown as HTMLCanvasElement;
    await initRunMat({ plotCanvas: canvas, enableGpu: false });

    expect(order).toEqual(["createPlotSurface", "initRunMat"]);
    expect(createSurfaceSpy).toHaveBeenCalledWith(canvas);
  });

  it("surfaces structured errors from createPlotSurface", async () => {
    const native: NativeModule = {
      default: async () => {},
      createPlotSurface: async () => {
        const err = new Error("canvas failed") as Error & { code?: string };
        err.code = "PlotCanvas";
        throw err;
      },
      initRunMat: async () => createMockNativeSession()
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await expect(
      initRunMat({ plotCanvas: {} as HTMLCanvasElement, enableGpu: false })
    ).rejects.toMatchObject({ code: "PlotCanvas" });
  });

  it("disposes the session and blocks further calls", async () => {
    const disposeSpy = vi.fn();
    const native: NativeModule = {
      default: async () => {},
      initRunMat: async () =>
        createMockNativeSession({
          dispose: disposeSpy
        })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({ enableGpu: false });
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
      initRunMat: async () =>
        createMockNativeSession({
          memoryUsage: () => ({ bytes: 1024, pages: 16 })
        })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({ enableGpu: false });
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

  it("throws when cameraState is requested but unsupported", async () => {
    const native: NativeModule = {
      default: async () => {},
      renderFigureImage: vi.fn(async () => new Uint8Array([1]))
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await expect(
      renderFigureImage({ cameraState: { projection: "2d", xMin: 0, xMax: 1, yMin: 0, yMax: 1 } })
    ).rejects.toThrow(/renderFigureImageWithCameraState/);
    expect(native.renderFigureImage).not.toHaveBeenCalled();
  });

  it("only uses textmark export when a textmark was explicitly requested", async () => {
    const native: NativeModule = {
      default: async () => {},
      renderFigureImage: vi.fn(async () => new Uint8Array([1, 2, 3])),
      renderFigureImageWithTextmark: vi.fn(async () => new Uint8Array([9, 9, 9]))
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const plain = await renderFigureImage();
    expect(Array.from(plain)).toEqual([1, 2, 3]);
    expect(native.renderFigureImage).toHaveBeenCalledWith(null, 0, 0);
    expect(native.renderFigureImageWithTextmark).not.toHaveBeenCalled();

    const marked = await renderFigureImage({ textmark: "draft" });
    expect(Array.from(marked)).toEqual([9, 9, 9]);
    expect(native.renderFigureImageWithTextmark).toHaveBeenCalledWith(null, 0, 0, "draft");
  });
});

describe("figure scene bindings", () => {
  afterEach(() => {
    __internals.setNativeModuleOverride(null);
    vi.restoreAllMocks();
  });

  it("exports figure scenes from async native bindings as Uint8Array", async () => {
    const spy = vi.fn(async () => new Uint8Array([7, 8, 9]));
    const native: NativeModule = {
      default: async () => {},
      exportFigureScene: spy
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const scene = await exportFigureScene(11);
    expect(Array.from(scene ?? [])).toEqual([7, 8, 9]);
    expect(spy).toHaveBeenCalledWith(11);
  });

  it("imports figure scenes and returns created handle", async () => {
    const scene = new Uint8Array([1, 2, 3]);
    const spy = vi.fn(() => 42);
    const native: NativeModule = {
      default: async () => {},
      importFigureScene: spy
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await expect(importFigureScene(scene)).resolves.toBe(42);
    expect(spy).toHaveBeenCalledWith(scene);
  });

  it("imports figure scenes by artifact path", async () => {
    const spy = vi.fn(() => 77);
    const native: NativeModule = {
      default: async () => {},
      importFigureSceneFromPath: spy,
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await expect(importFigureSceneFromPath("./.artifacts/objects/aa/scene.scene.json")).resolves.toBe(77);
    expect(spy).toHaveBeenCalledWith("./.artifacts/objects/aa/scene.scene.json");
  });

  it("rejects with a coerced figure error when figure scene import throws", async () => {
    const native: NativeModule = {
      default: async () => {},
      importFigureScene: vi.fn(() => {
        throw { code: "ReplayDecodeFailed" };
      })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await expect(importFigureScene(new Uint8Array([9, 9, 9]))).rejects.toMatchObject({
      code: "ReplayDecodeFailed"
    });
  });

  it("returns null when figure scene bindings are unavailable", async () => {
    const native: NativeModule = {
      default: async () => {}
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await expect(exportFigureScene(3)).resolves.toBeNull();
    await expect(importFigureScene(new Uint8Array([1]))).resolves.toBeNull();
  });
});

describe("workspace replay bindings", () => {
  afterEach(() => {
    __internals.setNativeModuleOverride(null);
    vi.restoreAllMocks();
  });

  it("forwards workspace export mode to native module", async () => {
    const spy = vi.fn(async () => new Uint8Array([4, 5, 6]));
    const native: NativeModule = {
      default: async () => {},
      exportWorkspaceState: spy
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const bytes = await exportWorkspaceState({ includeVariables: "force" });
    expect(Array.from(bytes ?? [])).toEqual([4, 5, 6]);
    expect(spy).toHaveBeenCalledWith("force");
  });

  it("imports workspace state and reports failures", async () => {
    const state = new Uint8Array([9]);
    const okNative: NativeModule = {
      default: async () => {},
      importWorkspaceState: vi.fn(() => true)
    } as NativeModule;
    __internals.setNativeModuleOverride(okNative);

    await expect(importWorkspaceState(state)).resolves.toBe(true);

    const failingNative: NativeModule = {
      default: async () => {},
      importWorkspaceState: vi.fn(() => {
        throw new Error("bad state");
      })
    } as NativeModule;
    __internals.setNativeModuleOverride(failingNative);

    await expect(importWorkspaceState(state)).resolves.toBe(false);
  });

  it("wires session workspace export/import helpers", async () => {
    const exportSpy = vi.fn(async () => new Uint8Array([2, 2]));
    const importSpy = vi.fn(() => true);
    const native: NativeModule = {
      default: async () => {},
      initRunMat: async () =>
        createMockNativeSession({
          exportWorkspaceState: exportSpy,
          importWorkspaceState: importSpy
        })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({ enableGpu: false });
    const exported = await session.exportWorkspaceState({ includeVariables: "off" });
    const imported = await session.importWorkspaceState(new Uint8Array([3, 3]));

    expect(Array.from(exported ?? [])).toEqual([2, 2]);
    expect(imported).toBe(true);
    expect(exportSpy).toHaveBeenCalledWith("off");
    expect(importSpy).toHaveBeenCalledWith(new Uint8Array([3, 3]));
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

  it("preserves clear-screen stdout control entries from the native session", async () => {
    const native: NativeModule = {
      default: async () => {},
      initRunMat: async () =>
        createMockNativeSession({
          executeRequest: () => ({
            ...baseExecuteResult,
            stdout: [{ stream: "clear", text: "", timestampMs: 123 }]
          })
        })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({ enableGpu: false });
    const result = await session.executeRequest({
      source: { kind: "text", name: "<test>", text: "clc;" }
    });
    expect(result.stdout).toEqual([{ stream: "clear", text: "", timestampMs: 123 }]);
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
      initRunMat: async () =>
        createMockNativeSession({
          materializeVariable: spy
        })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({ enableGpu: false });
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
      initRunMat: async () =>
        createMockNativeSession({
          materializeVariable: spy
        })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({ enableGpu: false });
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
      initRunMat: async () =>
        createMockNativeSession({
          setFusionPlanEnabled: spy
        })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({ enableGpu: false });
    session.setFusionPlanEnabled(true);
    expect(spy).toHaveBeenCalledWith(true);
  });
});

describe("browser project handoff and lock codecs", () => {
  afterEach(() => {
    __internals.setNativeModuleOverride(null);
  });

  it("keeps lock parsing and handoff construction in portable Rust exports", async () => {
    const lock = { schema_version: 1 };
    const handoff = { schema_version: 1, project: { graph: "fixed" } };
    const native: NativeModule = {
      default: async () => {},
      initRunMat: async () => createMockNativeSession(),
      decodePackageLock: vi.fn(() => lock),
      encodePackageLock: vi.fn(() => "schema_version = 1\n"),
      handoffFromFrozenProject: vi.fn(() => handoff)
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await expect(decodePackageLock("schema_version = 1\n")).resolves.toBe(lock);
    await expect(encodePackageLock(lock)).resolves.toBe("schema_version = 1\n");
    await expect(handoffFromFrozenProject({ graph: "fixed" })).resolves.toBe(handoff);
  });

  it("installs and preserves project revision through the session boundary", async () => {
    const install = vi.fn(() => ({ graph: "sha256:graph", sources: "sha256:sources" }));
    const clear = vi.fn();
    const revision = { graph: "sha256:graph", sources: "sha256:sources" };
    const native: NativeModule = {
      default: async () => {},
      initRunMat: async () =>
        createMockNativeSession({
          installProjectHandoff: install,
          clearProjectHandoff: clear,
          projectRevision: () => revision
        })
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    const session = await initRunMat({ enableGpu: false });
    const handoff = { schema_version: 1, project: {} };
    await expect(session.installProjectHandoff(handoff)).resolves.toEqual(revision);
    await expect(session.projectRevision()).resolves.toEqual(revision);
    await session.clearProjectHandoff();
    expect(install).toHaveBeenCalledWith(handoff);
    expect(clear).toHaveBeenCalledOnce();
  });
});

describe("canonical config authority", () => {
  afterEach(() => {
    __internals.setNativeModuleOverride(null);
  });

  it("delegates resolve, patch, and migration to the Rust/WASM exports", async () => {
    const resolved = {
      desktop: { artifacts: { root: ".artifacts" } },
      runtime: {
        error_namespace: "RunMat",
        language: { compat: "runmat" },
        accelerate: { enabled: true },
      },
    };
    const migration = {
      source: '[desktop.artifacts]\nroot = ".artifacts"\n',
      changed: true,
      removedKeys: ["artifact_root"],
    };
    const native: NativeModule = {
      default: async () => {},
      initRunMat: async () => createMockNativeSession(),
      resolveRunmatConfig: vi.fn(() => resolved),
      patchRunmatConfig: vi.fn(() => "patched"),
      migrateLegacyRunmatConfig: vi.fn(() => migration),
      migrateLegacyRunmatConfigInto: vi.fn(() => migration),
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await expect(resolveRunmatConfig("source", "toml")).resolves.toBe(resolved);
    await expect(patchRunmatConfig("source", "toml", {})).resolves.toBe(
      "patched"
    );
    await expect(
      migrateLegacyRunmatConfig("source", "toml")
    ).resolves.toBe(migration);
    await expect(
      migrateLegacyRunmatConfigInto("legacy", "canonical", "toml")
    ).resolves.toBe(migration);
  });
});

describe("signal trace helpers", () => {
  afterEach(() => {
    setSignalTraceHandler(null);
  });

  it("wraps callbacks with signal trace handler", () => {
    const handler = vi.fn((traceId: string, _name: string, fn: () => number) => {
      expect(traceId).toBe("trace-1");
      return fn();
    });
    setSignalTraceHandler(handler);
    const result = withSignalTrace("trace-1", "signal.process", () => 42);
    expect(result).toBe(42);
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it("bypasses trace handler when missing", () => {
    const handler = vi.fn();
    setSignalTraceHandler(handler);
    const result = withSignalTrace(undefined, "signal.process", () => 7);
    expect(result).toBe(7);
    expect(handler).not.toHaveBeenCalled();
  });
});

describe("resetPlotState", () => {
  afterEach(() => {
    __internals.setNativeModuleOverride(null);
  });

  it("delegates to the native module", async () => {
    const spy = vi.fn();
    const native: NativeModule = {
      default: async () => {},
      initRunMat: async () => createMockNativeSession(),
      resetPlotState: spy
    } as NativeModule;
    __internals.setNativeModuleOverride(native);

    await resetPlotState();

    expect(spy).toHaveBeenCalledTimes(1);
  });
});
