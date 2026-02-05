import type {
  RunMatFilesystemProvider
} from "./fs/provider-types.js";
import { createDefaultFsProvider } from "./fs/default.js";
import { __internals as workspaceHoverInternals } from "./workspace-hover.js";
import { installWebGpuCompatibilityShims } from "./webgpu-shims.js";
export {
  createInMemoryFsProvider,
  createIndexedDbFsHandle,
  createIndexedDbFsProvider,
  createDefaultFsProvider,
  createRemoteFsProvider,
  MemoryVolume
} from "./fs/index.js";
export type {
  InMemoryFsProviderOptions,
  IndexedDbFsHandle,
  IndexedDbProviderOptions,
  RemoteProviderOptions,
  VolumeSnapshotEntry
} from "./fs/index.js";
export type {
  RunMatFsFileType,
  RunMatFilesystemDirEntry,
  RunMatFilesystemMetadata,
  RunMatFilesystemProvider
} from "./fs/provider-types.js";
export { createWorkspaceHoverProvider } from "./workspace-hover.js";
export type {
  WorkspaceHoverOptions,
  WorkspaceHoverController,
  HoverFormatState
} from "./workspace-hover.js";
export { createFusionPlanAdapter } from "./fusion-plan.js";
export type {
  FusionPlanAdapter,
  FusionPlanAdapterOptions
} from "./fusion-plan.js";

export type LanguageCompatMode = "matlab" | "strict";
type RunMatPresetLogLevel = "trace" | "debug" | "info" | "warn" | "error";
export type RunMatLogLevel = RunMatPresetLogLevel | (string & Record<never, never>);

export type WasmInitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface RunMatSnapshotSource {
  bytes?: Uint8Array | ArrayBuffer;
  url?: string;
  fetcher?: SnapshotFetcher;
  stream?: ReadableStream<Uint8Array | ArrayBufferView | ArrayBuffer>;
}

export interface SnapshotFetcherContext {
  url?: string;
}

export type SnapshotFetcher = (
  ctx: SnapshotFetcherContext
) => Promise<SnapshotFetcherResult>;

export type SnapshotFetcherResult =
  | Uint8Array
  | ArrayBuffer
  | ArrayBufferView
  | Response
  | ReadableStream<Uint8Array | ArrayBufferView | ArrayBuffer>;

export interface RunMatInitOptions {
  snapshot?: RunMatSnapshotSource;
  enableGpu?: boolean;
  enableJit?: boolean;
  verbose?: boolean;
  logLevel?: RunMatLogLevel;
  gpuBufferPoolMaxPerKey?: number;
  telemetryConsent?: boolean;
  telemetryId?: string;
  telemetryRunKind?: "script" | "repl" | "benchmark" | "install";
  telemetryEmitter?: (envelope: unknown) => void;
  wgpuPowerPreference?: "auto" | "high-performance" | "low-power";
  wgpuForceFallbackAdapter?: boolean;
  wasmModule?: WasmInitInput;
  fsProvider?: RunMatFilesystemProvider;
  plotCanvas?: HTMLCanvasElement;
  scatterTargetPoints?: number;
  surfaceVertexBudget?: number;
  emitFusionPlan?: boolean;
  callstackLimit?: number;
  errorNamespace?: string;
  language?: {
    compat?: LanguageCompatMode;
  };
}

export type FigureEventKind = "created" | "updated" | "cleared" | "closed";

export interface FigureEvent {
  handle: number;
  kind: FigureEventKind;
  figure?: FigureSnapshot;
}

export interface FigureSnapshot {
  layout: FigureLayout;
  metadata: FigureMetadata;
  plots: FigurePlotDescriptor[];
}

export interface FigureLayout {
  axesRows: number;
  axesCols: number;
  axesIndices: number[];
}

export interface FigureMetadata {
  title?: string;
  xLabel?: string;
  yLabel?: string;
  gridEnabled: boolean;
  legendEnabled: boolean;
  colorbarEnabled: boolean;
  axisEqual: boolean;
  backgroundRgba: [number, number, number, number];
  colormap?: string;
  colorLimits?: [number, number];
  legendEntries: FigureLegendEntry[];
}

export interface FigureLegendEntry {
  label: string;
  plotType: FigurePlotKind;
  colorRgba: [number, number, number, number];
}

export interface FigurePlotDescriptor {
  kind: FigurePlotKind;
  label?: string;
  axesIndex: number;
  colorRgba: [number, number, number, number];
  visible: boolean;
}

export type FigurePlotKind =
  | "line"
  | "scatter"
  | "bar"
  | "error_bar"
  | "stairs"
  | "stem"
  | "area"
  | "quiver"
  | "pie"
  | "image"
  | "surface"
  | "scatter3"
  | "contour"
  | "contour_fill";

export type FigureEventListener = (event: FigureEvent) => void;
export type HoldMode = "on" | "off" | "toggle" | boolean;

export type PlotSurfaceEvent =
  | {
      kind: "mouseDown";
      x: number;
      y: number;
      button: number;
      shiftKey?: boolean;
      ctrlKey?: boolean;
      altKey?: boolean;
      metaKey?: boolean;
    }
  | {
      kind: "mouseUp";
      x: number;
      y: number;
      button: number;
      shiftKey?: boolean;
      ctrlKey?: boolean;
      altKey?: boolean;
      metaKey?: boolean;
    }
  | {
      kind: "mouseMove";
      x: number;
      y: number;
      dx: number;
      dy: number;
      shiftKey?: boolean;
      ctrlKey?: boolean;
      altKey?: boolean;
      metaKey?: boolean;
    }
  | {
      kind: "wheel";
      x: number;
      y: number;
    wheelDeltaX: number;
    wheelDeltaY: number;
      wheelDeltaMode?: number;
      shiftKey?: boolean;
      ctrlKey?: boolean;
      altKey?: boolean;
      metaKey?: boolean;
    };

export type StdoutStreamKind = "stdout" | "stderr";

export interface StdoutEntry {
  stream: StdoutStreamKind;
  text: string;
  timestampMs: number;
}

export type StdoutListener = (entry: StdoutEntry) => void;

export interface RuntimeLogEntry {
  ts: string;
  level: string;
  target: string;
  message: string;
  traceId?: string;
  spanId?: string;
  fields?: Record<string, unknown>;
}

export type RuntimeLogListener = (entry: RuntimeLogEntry) => void;

export interface TraceEvent {
  name: string;
  cat: string;
  ph: string;
  ts: number;
  dur?: number;
  pid?: number;
  tid?: number;
  traceId?: string;
  spanId?: string;
  parentSpanId?: string;
  args?: Record<string, unknown>;
}

export type TraceEventListener = (entries: TraceEvent[]) => void;

export type SignalTraceHandler = <T>(traceId: string, name: string, fn: () => T) => T;

let signalTraceHandler: SignalTraceHandler | null = null;

export function setSignalTraceHandler(handler: SignalTraceHandler | null): void {
  signalTraceHandler = handler;
}

export function withSignalTrace<T>(traceId: string | undefined, name: string, fn: () => T): T {
  if (traceId && signalTraceHandler) {
    return signalTraceHandler(traceId, name, fn);
  }
  return fn();
}

export type InputRequest =
  | { kind: "line"; prompt: string; echo: boolean }
  | { kind: "keyPress"; prompt: string; echo: boolean };

export interface ResumeInputPayload {
  value?: string | number | boolean;
  line?: string | number | boolean;
  kind?: "line" | "keyPress";
  error?: string;
}

export type ResumeInputScalar = string | number | boolean | null | undefined;
export type ResumeInputValue = ResumeInputScalar | ResumeInputPayload;

export type InputHandlerResult = ResumeInputValue | Promise<ResumeInputValue>;
export type InputHandler = (request: InputRequest) => InputHandlerResult;

export interface AxesInfo {
  handle: number;
  axesRows: number;
  axesCols: number;
  activeIndex: number;
}

export interface FigureImageOptions {
  handle?: number;
  width?: number;
  height?: number;
}

export interface FigureBindingError extends Error {
  code: "InvalidHandle" | "InvalidSubplotGrid" | "InvalidSubplotIndex" | "RenderFailure" | "Unknown";
  handle?: number;
  rows?: number;
  cols?: number;
  index?: number;
  details?: string;
}

export interface MatlabWarning {
  identifier: string;
  message: string;
}

export interface StdinEventLog {
  prompt: string;
  kind: "line" | "keyPress";
  echo: boolean;
  value?: string;
  error?: string;
}

export interface ExecuteResult {
  valueText?: string;
  valueJson?: unknown;
  typeInfo?: string;
  error?: RunMatErrorDetails;
  executionTimeMs: number;
  usedJit: boolean;
  stdout: StdoutEntry[];
  workspace: WorkspaceSnapshot;
  figuresTouched: number[];
  warnings: MatlabWarning[];
  stdinEvents: StdinEventLog[];
  profiling?: ProfilingSummary;
  fusionPlan?: FusionPlanSnapshot;
}

export type RunMatErrorKind = "syntax" | "semantic" | "compile" | "runtime";

export interface RunMatErrorSpan {
  start: number;
  end: number;
  line: number;
  column: number;
}

export interface RunMatErrorDetails {
  kind: RunMatErrorKind;
  message: string;
  identifier?: string;
  diagnostic: string;
  span?: RunMatErrorSpan;
  callstack: string[];
  callstackElided?: number;
}

export class RunMatExecutionError extends Error {
  readonly kind: RunMatErrorKind;
  readonly identifier?: string;
  readonly diagnostic: string;
  readonly span?: RunMatErrorSpan;
  readonly callstack: string[];
  readonly callstackElided?: number;

  constructor(details: RunMatErrorDetails) {
    super(details.message);
    this.name = "RunMatExecutionError";
    this.kind = details.kind;
    this.identifier = details.identifier;
    this.diagnostic = details.diagnostic;
    this.span = details.span;
    this.callstack = details.callstack;
    this.callstackElided = details.callstackElided;
  }
}

export type WorkspaceResidency = "cpu" | "gpu" | "unknown";

export interface WorkspaceSnapshot {
  full: boolean;
  version: number;
  values: WorkspaceEntry[];
}

export interface WorkspaceEntry {
  name: string;
  className: string;
  dtype?: string;
  shape: number[];
  isGpu: boolean;
  sizeBytes?: number;
  preview?: WorkspacePreview;
  residency: WorkspaceResidency;
  previewToken?: string;
}

export interface WorkspacePreview {
  values: number[];
  truncated: boolean;
}

export type WorkspaceMaterializeSelector =
  | string
  | {
      name?: string;
      previewToken?: string;
    };

export interface MaterializeSliceOptions {
  start: number[];
  shape: number[];
}

export interface MaterializeVariableOptions {
  limit?: number;
  slice?: MaterializeSliceOptions;
}

export interface MaterializedVariable {
  name: string;
  className: string;
  dtype?: string;
  shape: number[];
  isGpu: boolean;
  residency: WorkspaceResidency;
  sizeBytes?: number;
  preview?: WorkspacePreview;
  valueText: string;
  valueJson: unknown;
}

export interface ProfilingSummary {
  totalMs: number;
  cpuMs?: number;
  gpuMs?: number;
  kernelCount?: number;
}

export interface FusionPlanSnapshot {
  nodes: FusionPlanNode[];
  edges: FusionPlanEdge[];
  shaders: FusionPlanShader[];
  decisions: FusionPlanDecision[];
}

export interface FusionPlanNode {
  id: string;
  kind: string;
  label: string;
  shape: number[];
  residency?: string;
}

export interface FusionPlanEdge {
  from: string;
  to: string;
  reason?: string;
}

export interface FusionPlanShader {
  name: string;
  stage: string;
  workgroupSize?: [number, number, number];
  sourceHash?: string;
}

export interface FusionPlanDecision {
  nodeId: string;
  fused: boolean;
  reason?: string;
  thresholds?: string;
}

export interface SessionStats {
  totalExecutions: number;
  jitCompiled: number;
  interpreterFallback: number;
  totalExecutionTimeMs: number;
  averageExecutionTimeMs: number;
}

export interface GpuStatus {
  requested: boolean;
  active: boolean;
  error?: string;
  adapter?: GpuAdapterInfo;
}

export interface GpuAdapterInfo {
  deviceId: number;
  name: string;
  vendor: string;
  backend?: string;
  memoryBytes?: number;
  precision?: "single" | "double";
}

export interface MemoryUsage {
  bytes: number;
  pages: number;
}

export interface RunMatSessionHandle {
  execute(source: string): Promise<ExecuteResult>;
  resetSession(): Promise<void>;
  stats(): Promise<SessionStats>;
  clearWorkspace(): void;
  dispose(): void;
  telemetryConsent(): boolean;
  memoryUsage(): Promise<MemoryUsage>;
  telemetryClientId(): string | undefined;
  gpuStatus(): GpuStatus;
  cancelExecution(): void;
  cancelPendingRequests(): void;
  setInputHandler(handler: InputHandler | null): Promise<void>;
  materializeVariable(
    selector: WorkspaceMaterializeSelector,
    options?: MaterializeVariableOptions
  ): Promise<MaterializedVariable>;
  setFusionPlanEnabled(enabled: boolean): void;
  setLanguageCompat(mode: LanguageCompatMode): void;
  fusionPlanForSource?(source: string): Promise<FusionPlanSnapshot | null>;
}

interface NativeInitOptions {
  snapshotBytes?: Uint8Array;
  snapshotStream?: ReadableStream<Uint8Array | ArrayBufferView | ArrayBuffer>;
  snapshotUrl?: string;
  enableGpu?: boolean;
  enableJit?: boolean;
  verbose?: boolean;
  logLevel?: RunMatLogLevel;
  gpuBufferPoolMaxPerKey?: number;
  telemetryConsent?: boolean;
  telemetryId?: string;
  telemetryRunKind?: string;
  telemetryEmitter?: (envelope: unknown) => void;
  wgpuPowerPreference?: string;
  wgpuForceFallbackAdapter?: boolean;
  scatterTargetPoints?: number;
  surfaceVertexBudget?: number;
  emitFusionPlan?: boolean;
  callstackLimit?: number;
  errorNamespace?: string;
  languageCompat?: LanguageCompatMode;
}

interface RunMatNativeSession {
  execute(source: string): Promise<ExecuteResult>;
  resetSession(): void;
  stats(): SessionStats;
  clearWorkspace(): void;
  dispose?: () => void;
  telemetryConsent(): boolean;
  memoryUsage?: () => MemoryUsage;
  telemetryClientId?: () => string | undefined;
  gpuStatus(): GpuStatus;
  cancelExecution?: () => void;
  cancelPendingRequests?: () => void;
  setInputHandler?: (handler: InputHandler | null) => void;
  materializeVariable?: (
    selector: WorkspaceMaterializeSelectorWire,
    options?: MaterializeVariableOptionsWire
  ) => MaterializedVariable;
  setFusionPlanEnabled?: (enabled: boolean) => void;
  setLanguageCompat?: (mode: LanguageCompatMode) => void;
  fusionPlanForSource?: (source: string) => FusionPlanSnapshot | null;
}

type WorkspaceMaterializeSelectorWire =
  | string
  | {
      name?: string;
      token?: string;
      previewToken?: string;
    };

interface MaterializeVariableOptionsWire {
  limit?: number;
  slice?: {
    start: number[];
    shape: number[];
  };
}

interface RunMatNativeModule {
  default: (module?: WasmInitInput | Promise<WasmInitInput>) => Promise<unknown>;
  initRunMat(options: NativeInitOptions): Promise<RunMatNativeSession>;
  registerFsProvider?: (provider: RunMatFilesystemProvider) => void;
  // Legacy plot canvas bindings (handle-based).
  registerPlotCanvas?: (canvas: HTMLCanvasElement | OffscreenCanvas) => Promise<void>;
  deregisterPlotCanvas?: () => void;
  registerFigureCanvas?: (handle: number, canvas: HTMLCanvasElement | OffscreenCanvas) => Promise<void>;
  deregisterFigureCanvas?: (handle: number) => void;
  plotRendererReady?: () => boolean;
  renderCurrentFigureScene?: (handle: number) => void;
  createPlotSurface?: (canvas: HTMLCanvasElement | OffscreenCanvas) => Promise<number>;
  destroyPlotSurface?: (surfaceId: number) => void;
  resizePlotSurface?: (surfaceId: number, width: number, height: number) => void;
  bindSurfaceToFigure?: (surfaceId: number, handle: number) => void;
  presentSurface?: (surfaceId: number) => void;
  presentFigureOnSurface?: (surfaceId: number, handle: number) => void;
  handlePlotSurfaceEvent?: (surfaceId: number, event: PlotSurfaceEvent) => void;
  onFigureEvent?: (callback: ((event: FigureEvent) => void) | null) => void;
  newFigureHandle?: () => number;
  selectFigure?: (handle: number) => void;
  currentFigureHandle?: () => number;
  setHoldMode?: (mode: HoldMode) => boolean;
  configureSubplot?: (rows: number, cols: number, index: number) => void;
  clearFigure?: (handle: number | null) => number;
  closeFigure?: (handle: number | null) => number;
  currentAxesInfo?: () => AxesInfo;
  renderFigureImage?: (handle: number | null, width: number, height: number) => Promise<Uint8Array>;
  subscribeStdout?: (listener: (entry: StdoutEntry) => void) => number;
  unsubscribeStdout?: (id: number) => void;
  subscribeRuntimeLog?: (listener: (entry: RuntimeLogEntry) => void) => number;
  unsubscribeRuntimeLog?: (id: number) => void;
  setLogFilter?: (filter: string) => void;
  subscribeTraceEvents?: (listener: (entries: TraceEvent[]) => void) => number;
  unsubscribeTraceEvents?: (id: number) => void;
}

let loadPromise: Promise<RunMatNativeModule> | null = null;
let nativeModuleOverride: RunMatNativeModule | RunMatNativeSession | null = null;

async function loadNativeModule(wasmModule?: WasmInitInput): Promise<RunMatNativeModule> {
  installWebGpuCompatibilityShims();
  if (nativeModuleOverride) {
    if (isNativeSession(nativeModuleOverride)) {
      return {
        default: async () => {},
        initRunMat: async () => nativeModuleOverride
      } as RunMatNativeModule;
    }
    return nativeModuleOverride;
  }
  if (!loadPromise) {
    loadPromise = (async () => {
      const wasmModuleUrl = new URL("./pkg/runmat_wasm.js", import.meta.url);
      const native = (await import(wasmModuleUrl.href)) as unknown as RunMatNativeModule;
      if (typeof native.default === "function") {
        await native.default(wasmModule);
      }
      return native;
    })();
  }
  return loadPromise;
}

export async function initRunMat(options: RunMatInitOptions = {}): Promise<RunMatSessionHandle> {
  const native = await loadNativeModule(options.wasmModule);
  const snapshotResolution = await resolveSnapshotSource(options.snapshot);
  const fsProvider = await resolveFsProvider(options.fsProvider);
  if (fsProvider) {
    if (typeof native.registerFsProvider !== "function") {
      throw new Error("The loaded runmat-wasm module does not support filesystem providers yet.");
    }
    native.registerFsProvider(fsProvider);
  }
  if (options.plotCanvas) {
    if (typeof native.registerPlotCanvas === "function") {
      // Legacy binding: register the default plot canvas before initializing the session.
      await native.registerPlotCanvas(options.plotCanvas);
    } else if (typeof native.createPlotSurface === "function") {
      // New binding: create a surface for the provided canvas. The caller is responsible for binding/presenting.
      await native.createPlotSurface(options.plotCanvas);
    } else {
      const err = new Error(
        "The loaded runmat-wasm module does not support WebGPU plotting surfaces yet."
      ) as Error & { code?: string };
      err.code = "PlotCanvas";
      throw err;
    }
  }
  const supportsWebGpu = typeof navigator !== "undefined" && typeof (navigator as any).gpu !== "undefined";
  const hasExplicitEnableFlag = Object.prototype.hasOwnProperty.call(options, "enableGpu");
  const requestedGpu = options.enableGpu ?? true;
  let effectiveEnableGpu: boolean;
  if (hasExplicitEnableFlag) {
    if (requestedGpu && !supportsWebGpu) {
      console.warn(
        "[runmat] GPU acceleration was explicitly requested, but WebGPU APIs are unavailable in this context."
      );
      effectiveEnableGpu = false;
    } else {
      effectiveEnableGpu = requestedGpu;
    }
  } else {
    effectiveEnableGpu = requestedGpu && supportsWebGpu;
    if (requestedGpu && !supportsWebGpu) {
      console.warn("[runmat] WebGPU is not available in this environment; falling back to CPU execution.");
    }
  }
  const session = await native.initRunMat({
    snapshotBytes: snapshotResolution.bytes,
    snapshotStream: snapshotResolution.stream,
    snapshotUrl: options.snapshot?.url,
    enableGpu: effectiveEnableGpu,
    enableJit: options.enableJit ?? false,
    verbose: options.verbose ?? false,
    logLevel: options.logLevel,
    gpuBufferPoolMaxPerKey: options.gpuBufferPoolMaxPerKey,
    telemetryConsent: options.telemetryConsent ?? true,
    telemetryId: options.telemetryId,
    telemetryRunKind: options.telemetryRunKind,
    telemetryEmitter: options.telemetryEmitter,
    wgpuPowerPreference: options.wgpuPowerPreference ?? "auto",
    wgpuForceFallbackAdapter: options.wgpuForceFallbackAdapter ?? false,
    scatterTargetPoints: options.scatterTargetPoints,
    surfaceVertexBudget: options.surfaceVertexBudget,
    emitFusionPlan: options.emitFusionPlan ?? false,
    callstackLimit: options.callstackLimit,
    errorNamespace: options.errorNamespace,
    languageCompat: options.language?.compat
  });
  return new WebRunMatSession(session);
}

export async function plotRendererReady(): Promise<boolean> {
  const native = await loadNativeModule();
  if (typeof native.plotRendererReady !== "function") {
    return false;
  }
  return native.plotRendererReady();
}

export async function deregisterPlotCanvas(): Promise<void> {
  const native = await loadNativeModule();
  if (typeof native.deregisterPlotCanvas !== "function") {
    throw new Error("The loaded runmat-wasm module does not expose deregisterPlotCanvas.");
  }
  native.deregisterPlotCanvas();
}

export async function deregisterFigureCanvas(handle: number): Promise<void> {
  const native = await loadNativeModule();
  if (typeof native.deregisterFigureCanvas !== "function") {
    throw new Error("The loaded runmat-wasm module does not expose deregisterFigureCanvas.");
  }
  native.deregisterFigureCanvas(handle);
}

export async function createPlotSurface(canvas: HTMLCanvasElement | OffscreenCanvas): Promise<number> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "createPlotSurface");
  return native.createPlotSurface(canvas);
}

export async function destroyPlotSurface(surfaceId: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "destroyPlotSurface");
  native.destroyPlotSurface(surfaceId);
}

export async function resizePlotSurface(surfaceId: number, widthPx: number, heightPx: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "resizePlotSurface");
  native.resizePlotSurface(surfaceId, widthPx, heightPx);
}

export async function bindSurfaceToFigure(surfaceId: number, handle: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "bindSurfaceToFigure");
  native.bindSurfaceToFigure(surfaceId, handle);
}

export async function presentSurface(surfaceId: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "presentSurface");
  native.presentSurface(surfaceId);
}

export async function presentFigureOnSurface(surfaceId: number, handle: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "presentFigureOnSurface");
  native.presentFigureOnSurface(surfaceId, handle);
}

export async function renderCurrentFigureScene(handle: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "renderCurrentFigureScene");
  native.renderCurrentFigureScene(handle);
}

export async function handlePlotSurfaceEvent(surfaceId: number, event: PlotSurfaceEvent): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "handlePlotSurfaceEvent");
  native.handlePlotSurfaceEvent(surfaceId, event);
}

export async function onFigureEvent(listener: FigureEventListener | null): Promise<void> {
  const native = await loadNativeModule();
  if (typeof native.onFigureEvent !== "function") {
    throw new Error("The loaded runmat-wasm module does not expose figure events yet.");
  }
  native.onFigureEvent(listener ? (event: FigureEvent) => listener(event) : null);
}

export async function subscribeStdout(listener: StdoutListener): Promise<number> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "subscribeStdout");
  return native.subscribeStdout((entry: StdoutEntry) => listener(entry));
}

export async function unsubscribeStdout(id: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "unsubscribeStdout");
  native.unsubscribeStdout(id);
}

export async function subscribeRuntimeLog(listener: RuntimeLogListener): Promise<number> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "subscribeRuntimeLog");
  return native.subscribeRuntimeLog((entry: RuntimeLogEntry) => listener(entry));
}

export async function unsubscribeRuntimeLog(id: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "unsubscribeRuntimeLog");
  native.unsubscribeRuntimeLog(id);
}

export async function setLogFilter(filter: string): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "setLogFilter");
  native.setLogFilter(filter);
}

export async function subscribeTraceEvents(listener: TraceEventListener): Promise<number> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "subscribeTraceEvents");
  return native.subscribeTraceEvents((entries: TraceEvent[]) => listener(entries));
}

export async function unsubscribeTraceEvents(id: number): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "unsubscribeTraceEvents");
  native.unsubscribeTraceEvents(id);
}

export async function figure(handle?: number): Promise<number> {
  const native = await loadNativeModule();
  if (typeof handle === "number") {
    requireNativeFunction(native, "selectFigure");
    native.selectFigure(handle);
    return handle;
  }
  requireNativeFunction(native, "newFigureHandle");
  return native.newFigureHandle();
}

export async function newFigureHandle(): Promise<number> {
  return figure();
}

export async function currentFigureHandle(): Promise<number> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "currentFigureHandle");
  return native.currentFigureHandle();
}

export async function setHoldMode(mode: HoldMode = "toggle"): Promise<boolean> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "setHoldMode");
  return native.setHoldMode(mode);
}

export async function hold(mode: HoldMode = "toggle"): Promise<boolean> {
  return setHoldMode(mode);
}

export async function holdOn(): Promise<boolean> {
  return setHoldMode("on");
}

export async function holdOff(): Promise<boolean> {
  return setHoldMode("off");
}

export async function configureSubplot(rows: number, cols: number, index = 0): Promise<void> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "configureSubplot");
  try {
    native.configureSubplot(rows, cols, index);
  } catch (error) {
    throw coerceFigureError(error);
  }
}

export async function subplot(rows: number, cols: number, index = 0): Promise<void> {
  return configureSubplot(rows, cols, index);
}

export async function clearFigure(handle?: number): Promise<number> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "clearFigure");
  try {
    return native.clearFigure(handle ?? null);
  } catch (error) {
    throw coerceFigureError(error);
  }
}

export async function closeFigure(handle?: number): Promise<number> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "closeFigure");
  try {
    return native.closeFigure(handle ?? null);
  } catch (error) {
    throw coerceFigureError(error);
  }
}

export async function currentAxesInfo(): Promise<AxesInfo> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "currentAxesInfo");
  return native.currentAxesInfo();
}

export async function renderFigureImage(options: FigureImageOptions = {}): Promise<Uint8Array> {
  const native = await loadNativeModule();
  requireNativeFunction(native, "renderFigureImage");
  const handle = typeof options.handle === "number" ? options.handle : null;
  const width = options.width ?? 0;
  const height = options.height ?? 0;
  try {
    const bytes = await native.renderFigureImage(handle, width, height);
    if (bytes instanceof Uint8Array) {
      return bytes;
    }
    return new Uint8Array(bytes ?? []);
  } catch (error) {
    throw coerceFigureError(error);
  }
}

class WebRunMatSession implements RunMatSessionHandle {
  private disposed = false;

  constructor(private readonly native: RunMatNativeSession) {}

  private ensureActive(): void {
    if (this.disposed) {
      throw new Error("RunMat session has been disposed");
    }
  }

  async execute(source: string): Promise<ExecuteResult> {
    this.ensureActive();
    try {
      return await this.native.execute(source);
    } catch (error) {
      throw coerceRunMatError(error);
    }
  }

  async resetSession(): Promise<void> {
    this.ensureActive();
    this.native.resetSession();
  }

  async stats(): Promise<SessionStats> {
    this.ensureActive();
    return this.native.stats();
  }

  clearWorkspace(): void {
    this.ensureActive();
    this.native.clearWorkspace();
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    if (typeof this.native.dispose === "function") {
      this.native.dispose();
    }
    this.disposed = true;
  }

  telemetryConsent(): boolean {
    this.ensureActive();
    return this.native.telemetryConsent();
  }

  telemetryClientId(): string | undefined {
    this.ensureActive();
    if (typeof this.native.telemetryClientId !== "function") {
      return undefined;
    }
    return this.native.telemetryClientId() ?? undefined;
  }

  async memoryUsage(): Promise<MemoryUsage> {
    this.ensureActive();
    if (typeof this.native.memoryUsage !== "function") {
      return { bytes: 0, pages: 0 };
    }
    return this.native.memoryUsage();
  }

  gpuStatus(): GpuStatus {
    this.ensureActive();
    return this.native.gpuStatus();
  }

  cancelExecution(): void {
    if (this.disposed) {
      return;
    }
    if (typeof this.native.cancelExecution === "function") {
      this.native.cancelExecution();
    }
  }

  cancelPendingRequests(): void {
    if (this.disposed) {
      return;
    }
    if (typeof this.native.cancelPendingRequests === "function") {
      this.native.cancelPendingRequests();
    }
  }

  async setInputHandler(handler: InputHandler | null): Promise<void> {
    this.ensureActive();
    if (typeof this.native.setInputHandler !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose setInputHandler yet.");
    }
    this.native.setInputHandler(handler ? (request: InputRequest) => handler(request) : null);
  }

  async materializeVariable(
    selector: WorkspaceMaterializeSelector,
    options?: MaterializeVariableOptions
  ): Promise<MaterializedVariable> {
    this.ensureActive();
    requireNativeFunction(this.native, "materializeVariable");
    const wireSelector = normalizeMaterializeSelector(selector);
    const wireOptions = normalizeMaterializeOptions(options);
    return this.native.materializeVariable(
      wireSelector,
      wireOptions as MaterializeVariableOptionsWire | undefined
    );
  }

  setFusionPlanEnabled(enabled: boolean): void {
    this.ensureActive();
    requireNativeFunction(this.native, "setFusionPlanEnabled");
    this.native.setFusionPlanEnabled(enabled);
  }

  setLanguageCompat(mode: LanguageCompatMode): void {
    this.ensureActive();
    requireNativeFunction(this.native, "setLanguageCompat");
    this.native.setLanguageCompat(mode);
  }

  async fusionPlanForSource(source: string): Promise<FusionPlanSnapshot | null> {
    this.ensureActive();
    if (typeof this.native.fusionPlanForSource !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose fusionPlanForSource yet.");
    }
    try {
      return this.native.fusionPlanForSource(source) ?? null;
    } catch (error) {
      throw coerceRunMatError(error);
    }
  }
}

function ensureFsProvider(provider: RunMatFilesystemProvider): void {
  const requiredMethods: (keyof RunMatFilesystemProvider)[] = [
    "readFile",
    "writeFile",
    "removeFile",
    "metadata",
    "readDir"
  ];
  for (const method of requiredMethods) {
    if (typeof provider[method] !== "function") {
      throw new Error(`fsProvider.${String(method)} must be a function`);
    }
  }
}

function requireNativeFunction<T, K extends keyof T>(
  native: T,
  method: K
): asserts native is T & Required<Pick<T, K>> {
  if (typeof native[method] !== "function") {
    throw new Error(`The loaded runmat-wasm module does not expose ${String(method)} yet.`);
  }
}

type FigureErrorPayload = {
    code?: string;
    message?: string;
    handle?: number;
    rows?: number;
  cols?: number;
  index?: number;
  details?: string;
};

type RunMatErrorPayload = {
  kind: RunMatErrorKind;
  message: string;
  identifier?: string;
  diagnostic: string;
  span?: RunMatErrorSpan;
  callstack?: string[];
  callstackElided?: number;
};

function isRunMatErrorPayload(value: unknown): value is RunMatErrorPayload {
  if (!value || typeof value !== "object") {
    return false;
  }
  const payload = value as RunMatErrorPayload;
  return (
    typeof payload.kind === "string" &&
    typeof payload.message === "string" &&
    typeof payload.diagnostic === "string"
  );
}

function coerceRunMatError(value: unknown): RunMatExecutionError {
  if (isRunMatErrorPayload(value)) {
    return new RunMatExecutionError({
      kind: value.kind,
      message: value.message,
      identifier: value.identifier,
      diagnostic: value.diagnostic,
      span: value.span,
      callstack: value.callstack ?? [],
      callstackElided: value.callstackElided
    });
  }
  if (value instanceof RunMatExecutionError) {
    return value;
  }
  if (value instanceof Error) {
    return new RunMatExecutionError({
      kind: "runtime",
      message: value.message,
      diagnostic: value.message,
      callstack: []
    });
  }
  const message = String(value ?? "RunMat execution failed");
  return new RunMatExecutionError({
    kind: "runtime",
    message,
    diagnostic: message,
    callstack: []
  });
}

function isFigureErrorPayload(value: unknown): value is FigureErrorPayload {
  return (
    typeof value === "object" &&
    value !== null &&
    "code" in value &&
    typeof (value as FigureErrorPayload).code === "string"
  );
}

function coerceFigureError(value: unknown): FigureBindingError {
  if (isFigureErrorPayload(value)) {
    const err = new Error(value.message ?? value.code ?? "Figure error") as FigureBindingError;
    err.code = (value.code as FigureBindingError["code"]) ?? "Unknown";
    if (typeof value.handle === "number") {
      err.handle = value.handle;
    }
    if (typeof value.rows === "number") {
      err.rows = value.rows;
    }
    if (typeof value.cols === "number") {
      err.cols = value.cols;
    }
    if (typeof value.index === "number") {
      err.index = value.index;
    }
    if (typeof value.details === "string") {
      err.details = value.details;
    }
    return err;
  }
  if (value instanceof Error) {
    const err = value as FigureBindingError;
    if (!err.code) {
      err.code = "Unknown";
    }
    return err;
  }
  const err = new Error(String(value)) as FigureBindingError;
  err.code = "Unknown";
  return err;
}

async function resolveFsProvider(
  provided?: RunMatFilesystemProvider
): Promise<RunMatFilesystemProvider | undefined> {
  if (provided) {
    ensureFsProvider(provided);
    return provided;
  }
  try {
    const autoProvider = await createDefaultFsProvider();
    ensureFsProvider(autoProvider);
    return autoProvider;
  } catch (error) {
    console.warn("[runmat] Unable to initialize default filesystem provider.", error);
    return undefined;
  }
}

type SnapshotStream = ReadableStream<Uint8Array | ArrayBufferView | ArrayBuffer>;

interface SnapshotResolution {
  bytes?: Uint8Array;
  stream?: SnapshotStream;
}

async function resolveSnapshotSource(source?: RunMatSnapshotSource): Promise<SnapshotResolution> {
  if (!source) {
    return {};
  }
  if (source.bytes) {
    return { bytes: toUint8Array(source.bytes) };
  }
  if (source.stream) {
    return { stream: source.stream };
  }
  if (source.fetcher) {
    const fetched = await source.fetcher({ url: source.url });
    return coerceSnapshotFetcherResult(fetched);
  }
  if (source.url) {
    if (typeof fetch === "undefined") {
      throw new Error(
        "Global fetch API is unavailable; provide snapshot.bytes or snapshot.fetcher instead."
      );
    }
    const response = await fetch(source.url);
    return coerceResponseForSnapshot(response, source.url);
  }
  return {};
}

async function coerceSnapshotFetcherResult(value: SnapshotFetcherResult): Promise<SnapshotResolution> {
  if (value instanceof Uint8Array) {
    return { bytes: value };
  }
  if (value instanceof ArrayBuffer) {
    return { bytes: new Uint8Array(value) };
  }
  if (ArrayBuffer.isView(value)) {
    return { bytes: toUint8Array(value) };
  }
  if (isReadableStream(value)) {
    return { stream: value };
  }
  if (isResponse(value)) {
    return coerceResponseForSnapshot(value);
  }
  throw new Error("Unsupported snapshot fetcher result");
}

async function coerceResponseForSnapshot(response: Response, origin?: string): Promise<SnapshotResolution> {
  if (!response.ok) {
    const suffix = origin ? ` from ${origin}` : "";
    throw new Error(`Failed to fetch snapshot${suffix} (status ${response.status})`);
  }
  if (response.body) {
    return { stream: response.body as SnapshotStream };
  }
  const buffer = await response.arrayBuffer();
  return { bytes: new Uint8Array(buffer) };
}

function isReadableStream(value: unknown): value is SnapshotStream {
  return typeof ReadableStream !== "undefined" && value instanceof ReadableStream;
}

function isResponse(value: unknown): value is Response {
  return typeof Response !== "undefined" && value instanceof Response;
}

async function fetchSnapshotFromUrl(url: string): Promise<Uint8Array> {
  if (typeof fetch === "undefined") {
    throw new Error("Global fetch API is unavailable; provide snapshot.bytes or snapshot.fetcher instead.");
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch snapshot from ${url} (status ${response.status})`);
  }
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    return new Uint8Array(buffer);
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    if (!value) {
      continue;
    }
    const chunk = value instanceof Uint8Array ? value : new Uint8Array(value);
    total += chunk.length;
    chunks.push(chunk);
  }
  const result = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

function toUint8Array(data: Uint8Array | ArrayBuffer | ArrayBufferView): Uint8Array {
  if (data instanceof Uint8Array) {
    return data;
  }
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
  }
  throw new Error("Unsupported snapshot buffer type");
}

function normalizeMaterializeSelector(
  selector: WorkspaceMaterializeSelector
): WorkspaceMaterializeSelectorWire {
  if (typeof selector === "string") {
    const trimmed = selector.trim();
    if (!trimmed) {
      throw new Error("materializeVariable selector string must not be empty");
    }
    return trimmed;
  }
  if (!selector || typeof selector !== "object") {
    throw new Error("materializeVariable selector must be a string or object");
  }
  if (typeof selector.previewToken === "string" && selector.previewToken.trim()) {
    return { previewToken: selector.previewToken.trim() };
  }
  const payload: { name?: string } = {};
  if (typeof selector.name === "string" && selector.name.trim()) {
    payload.name = selector.name.trim();
  }
  if (!payload.name) {
    throw new Error("materializeVariable selector requires name or previewToken");
  }
  return payload;
}

function normalizeMaterializeOptions(
  options?: MaterializeVariableOptions
): MaterializeVariableOptionsWire | undefined {
  if (!options) {
    return undefined;
  }
  const payload: MaterializeVariableOptionsWire = {};
  if (typeof options.limit === "number" && Number.isFinite(options.limit)) {
    const limit = Math.floor(options.limit);
    if (limit > 0) {
      payload.limit = limit;
    }
  }
  if (options.slice) {
    const start = normalizeSliceVector(options.slice.start, false);
    const shape = normalizeSliceVector(options.slice.shape, true);
    if (start && shape) {
      payload.slice = { start, shape };
    }
  }
  return payload;
}

function normalizeSliceVector(values?: number[], requirePositive = false): number[] | undefined {
  if (!Array.isArray(values) || values.length === 0) {
    return undefined;
  }
  const normalized: number[] = [];
  for (const raw of values) {
    if (typeof raw !== "number" || !Number.isFinite(raw)) {
      return undefined;
    }
    const base = Math.floor(raw);
    if (requirePositive) {
      if (base <= 0) {
        return undefined;
      }
      normalized.push(base);
    } else {
      normalized.push(Math.max(0, base));
    }
  }
  return normalized;
}

function normalizeResumeInputValue(value: ResumeInputValue): ResumeInputPayload {
  if (value && typeof value === "object") {
    const payload = value as ResumeInputPayload;
    if (payload.error) {
      return { error: String(payload.error) };
    }
    if (payload.kind === "keyPress") {
      return { kind: "keyPress" };
    }
    if (payload.kind === "line") {
      const raw = payload.value ?? payload.line ?? "";
      return { kind: "line", value: String(raw ?? "") };
    }
  }
  // Scalars and nulls are treated as line inputs.
  if (value === null || value === undefined) {
    return { kind: "line", value: "" };
  }
  return { kind: "line", value: String(value) };
}

export const __internals = {
  resolveSnapshotSource,
  fetchSnapshotFromUrl,
  coerceFigureError,
  normalizeResumeInputValue,
  workspaceHover: workspaceHoverInternals as unknown as Record<string, unknown>,
  setNativeModuleOverride(module: RunMatNativeModule | RunMatNativeSession | null): void {
    nativeModuleOverride = module;
    if (!module) {
      loadPromise = null;
    }
  }
};

function isNativeSession(value: unknown): value is RunMatNativeSession {
  return Boolean(value && typeof (value as RunMatNativeSession).execute === "function");
}
