import type {
  RunMatFilesystemProvider
} from "./fs/provider-types.js";
import { createDefaultFsProvider } from "./fs/default.js";
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
  telemetryConsent?: boolean;
  wgpuPowerPreference?: "auto" | "high-performance" | "low-power";
  wgpuForceFallbackAdapter?: boolean;
  wasmModule?: WasmInitInput;
  fsProvider?: RunMatFilesystemProvider;
  plotCanvas?: HTMLCanvasElement;
  scatterTargetPoints?: number;
  surfaceVertexBudget?: number;
}

export type FigureEventKind = "created" | "updated" | "cleared" | "closed";

export interface FigureEvent {
  handle: number;
  kind: FigureEventKind;
  axesRows: number;
  axesCols: number;
  plotCount: number;
  axesIndices: number[];
  title?: string;
}

export type FigureEventListener = (event: FigureEvent) => void;
export type HoldMode = "on" | "off" | "toggle" | boolean;

export type StdoutStreamKind = "stdout" | "stderr";

export interface StdoutEntry {
  stream: StdoutStreamKind;
  text: string;
  timestampMs: number;
}

export type StdoutListener = (entry: StdoutEntry) => void;

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

export interface PendingStdinRequest {
  id: string;
  request: {
    prompt: string;
    kind: "line" | "keyPress";
    echo: boolean;
  };
}

export interface AxesInfo {
  handle: number;
  axesRows: number;
  axesCols: number;
  activeIndex: number;
}

export interface FigureBindingError extends Error {
  code: "InvalidHandle" | "InvalidSubplotGrid" | "InvalidSubplotIndex" | "Unknown";
  handle?: number;
  rows?: number;
  cols?: number;
  index?: number;
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
  error?: string;
  executionTimeMs: number;
  usedJit: boolean;
  stdout: StdoutEntry[];
  workspace: WorkspaceSnapshot;
  figuresTouched: number[];
  warnings: MatlabWarning[];
  stdinEvents: StdinEventLog[];
  profiling?: ProfilingSummary;
  fusionPlan?: FusionPlanSnapshot;
  stdinRequested?: PendingStdinRequest;
}

export interface WorkspaceSnapshot {
  full: boolean;
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
}

export interface WorkspacePreview {
  values: number[];
  truncated: boolean;
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

export interface RunMatSessionHandle {
  execute(source: string): Promise<ExecuteResult>;
  resetSession(): Promise<void>;
  stats(): Promise<SessionStats>;
  clearWorkspace(): void;
  telemetryConsent(): boolean;
  gpuStatus(): GpuStatus;
  cancelExecution(): void;
  setInputHandler(handler: InputHandler | null): Promise<void>;
  resumeInput(requestId: string, value: ResumeInputValue): Promise<ExecuteResult>;
  pendingStdinRequests(): Promise<PendingStdinRequest[]>;
}

interface NativeInitOptions {
  snapshotBytes?: Uint8Array;
  snapshotStream?: ReadableStream<Uint8Array | ArrayBufferView | ArrayBuffer>;
  snapshotUrl?: string;
  enableGpu?: boolean;
  enableJit?: boolean;
  verbose?: boolean;
  telemetryConsent?: boolean;
  wgpuPowerPreference?: string;
  wgpuForceFallbackAdapter?: boolean;
  scatterTargetPoints?: number;
  surfaceVertexBudget?: number;
}

interface RunMatNativeSession {
  execute(source: string): ExecuteResult;
  resetSession(): void;
  stats(): SessionStats;
  clearWorkspace(): void;
  telemetryConsent(): boolean;
  gpuStatus(): GpuStatus;
  cancelExecution?: () => void;
  setInputHandler?: (handler: InputHandler | null) => void;
  resumeInput?: (requestId: string, value: ResumeInputWireValue) => ExecuteResult;
  pendingStdinRequests?: () => PendingStdinRequest[];
}

interface ResumeInputWireValue {
  kind?: "line" | "keyPress";
  value?: string;
  error?: string;
}

interface RunMatNativeModule {
  default: (module?: WasmInitInput | Promise<WasmInitInput>) => Promise<unknown>;
  initRunMat(options: NativeInitOptions): Promise<RunMatNativeSession>;
  registerFsProvider?: (provider: RunMatFilesystemProvider) => void;
  registerPlotCanvas?: (canvas: HTMLCanvasElement) => Promise<void>;
  plotRendererReady?: () => boolean;
  registerFigureCanvas?: (handle: number, canvas: HTMLCanvasElement) => Promise<void>;
  onFigureEvent?: (callback: ((event: FigureEvent) => void) | null) => void;
  newFigureHandle?: () => number;
  selectFigure?: (handle: number) => void;
  currentFigureHandle?: () => number;
  setHoldMode?: (mode: HoldMode) => boolean;
  configureSubplot?: (rows: number, cols: number, index: number) => void;
  clearFigure?: (handle: number | null) => number;
  closeFigure?: (handle: number | null) => number;
  currentAxesInfo?: () => AxesInfo;
  subscribeStdout?: (listener: (entry: StdoutEntry) => void) => number;
  unsubscribeStdout?: (id: number) => void;
  resumeInput?: (requestId: string, value: ResumeInputWireValue) => ExecuteResult;
  pendingStdinRequests?: () => PendingStdinRequest[];
}

let loadPromise: Promise<RunMatNativeModule> | null = null;

async function loadNativeModule(wasmModule?: WasmInitInput): Promise<RunMatNativeModule> {
  if (!loadPromise) {
    loadPromise = (async () => {
      const native = (await import("../pkg/runmat_wasm.js")) as unknown as RunMatNativeModule;
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
    if (typeof native.registerPlotCanvas !== "function") {
      throw new Error("The loaded runmat-wasm module does not support WebGPU plotting yet.");
    }
    await native.registerPlotCanvas(options.plotCanvas);
  }
  const supportsWebGpu = typeof navigator !== "undefined" && typeof (navigator as any).gpu !== "undefined";
  const requestedGpu = options.enableGpu ?? true;
  const effectiveEnableGpu = requestedGpu && supportsWebGpu;
  if (requestedGpu && !supportsWebGpu) {
    console.warn("[runmat] WebGPU is not available in this environment; falling back to CPU execution.");
  }
  const session = await native.initRunMat({
    snapshotBytes: snapshotResolution.bytes,
    snapshotStream: snapshotResolution.stream,
    snapshotUrl: options.snapshot?.url,
    enableGpu: effectiveEnableGpu,
    enableJit: options.enableJit ?? false,
    verbose: options.verbose ?? false,
    telemetryConsent: options.telemetryConsent ?? true,
    wgpuPowerPreference: options.wgpuPowerPreference ?? "auto",
    wgpuForceFallbackAdapter: options.wgpuForceFallbackAdapter ?? false,
    scatterTargetPoints: options.scatterTargetPoints,
    surfaceVertexBudget: options.surfaceVertexBudget
  });
  return new WebRunMatSession(session);
}

export async function attachPlotCanvas(canvas: HTMLCanvasElement): Promise<void> {
  const native = await loadNativeModule();
  if (typeof native.registerPlotCanvas !== "function") {
    throw new Error("The loaded runmat-wasm module does not support WebGPU plotting yet.");
  }
  await native.registerPlotCanvas(canvas);
}

export async function plotRendererReady(): Promise<boolean> {
  const native = await loadNativeModule();
  if (typeof native.plotRendererReady !== "function") {
    return false;
  }
  return native.plotRendererReady();
}

export async function registerFigureCanvas(handle: number, canvas: HTMLCanvasElement): Promise<void> {
  const native = await loadNativeModule();
  if (typeof native.registerFigureCanvas !== "function") {
    throw new Error("The loaded runmat-wasm module does not support figure-specific canvases yet.");
  }
  await native.registerFigureCanvas(handle, canvas);
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

class WebRunMatSession implements RunMatSessionHandle {
  constructor(private readonly native: RunMatNativeSession) {}

  async execute(source: string): Promise<ExecuteResult> {
    return this.native.execute(source);
  }

  async resetSession(): Promise<void> {
    this.native.resetSession();
  }

  async stats(): Promise<SessionStats> {
    return this.native.stats();
  }

  clearWorkspace(): void {
    this.native.clearWorkspace();
  }

  telemetryConsent(): boolean {
    return this.native.telemetryConsent();
  }

  gpuStatus(): GpuStatus {
    return this.native.gpuStatus();
  }

  cancelExecution(): void {
    if (typeof this.native.cancelExecution === "function") {
      this.native.cancelExecution();
    }
  }

  async setInputHandler(handler: InputHandler | null): Promise<void> {
    if (typeof this.native.setInputHandler !== "function") {
      throw new Error("The loaded runmat-wasm module does not expose setInputHandler yet.");
    }
    this.native.setInputHandler(handler ? (request: InputRequest) => handler(request) : null);
  }

  async resumeInput(requestId: string, value: ResumeInputValue): Promise<ExecuteResult> {
    requireNativeFunction(this.native, "resumeInput");
    const payload = normalizeResumeInputValue(value);
    return this.native.resumeInput(requestId, payload);
  }

  async pendingStdinRequests(): Promise<PendingStdinRequest[]> {
    if (typeof this.native.pendingStdinRequests !== "function") {
      return [];
    }
    return this.native.pendingStdinRequests();
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

function requireNativeFunction<K extends keyof RunMatNativeModule>(
  native: RunMatNativeModule,
  method: K
): asserts native is RunMatNativeModule & Required<Pick<RunMatNativeModule, K>> {
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
};

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

function normalizeResumeInputValue(input: ResumeInputValue): ResumeInputWireValue {
  if (isResumeInputPayload(input)) {
    if (typeof input.error === "string") {
      return { error: input.error };
    }
    if (input.kind === "keyPress") {
      return { kind: "keyPress" };
    }
    const raw = input.value ?? input.line;
    return { kind: "line", value: coerceResumeValue(raw) };
  }
  if (input === null || input === undefined) {
    return { kind: "line", value: "" };
  }
  return { kind: "line", value: String(input) };
}

function coerceResumeValue(value: string | number | boolean | undefined): string {
  if (value === undefined) {
    return "";
  }
  return String(value);
}

function isResumeInputPayload(value: ResumeInputValue): value is ResumeInputPayload {
  if (value === null || value === undefined) {
    return false;
  }
  if (typeof value !== "object") {
    return false;
  }
  const payload = value as Record<string, unknown>;
  return (
    "value" in payload ||
    "line" in payload ||
    "kind" in payload ||
    "error" in payload
  );
}

export const __internals = {
  resolveSnapshotSource,
  fetchSnapshotFromUrl,
  coerceFigureError,
  normalizeResumeInputValue
};
