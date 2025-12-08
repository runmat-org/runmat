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

export interface FigureEvent {
  handle: number;
  axesRows: number;
  axesCols: number;
  plotCount: number;
  axesIndices: number[];
  title?: string;
}

export type FigureEventListener = (event: FigureEvent) => void;
export type HoldMode = "on" | "off" | "toggle" | boolean;

export interface ExecuteResult {
  valueText?: string;
  valueJson?: unknown;
  typeInfo?: string;
  error?: string;
  executionTimeMs: number;
  usedJit: boolean;
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
  native.configureSubplot(rows, cols, index);
}

export async function subplot(rows: number, cols: number, index = 0): Promise<void> {
  return configureSubplot(rows, cols, index);
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

export const __internals = {
  resolveSnapshotSource,
  fetchSnapshotFromUrl
};
