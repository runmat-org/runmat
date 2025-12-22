import type { RunMatFilesystemProvider } from "./fs/provider-types.js";
export { createInMemoryFsProvider, createIndexedDbFsHandle, createIndexedDbFsProvider, createDefaultFsProvider, createRemoteFsProvider, MemoryVolume } from "./fs/index.js";
export type { InMemoryFsProviderOptions, IndexedDbFsHandle, IndexedDbProviderOptions, RemoteProviderOptions, VolumeSnapshotEntry } from "./fs/index.js";
export type { RunMatFsFileType, RunMatFilesystemDirEntry, RunMatFilesystemMetadata, RunMatFilesystemProvider } from "./fs/provider-types.js";
export { createWorkspaceHoverProvider } from "./workspace-hover.js";
export type { WorkspaceHoverOptions, WorkspaceHoverController, HoverFormatState } from "./workspace-hover.js";
export { createFusionPlanAdapter } from "./fusion-plan.js";
export type { FusionPlanAdapter, FusionPlanAdapterOptions } from "./fusion-plan.js";
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
export type SnapshotFetcher = (ctx: SnapshotFetcherContext) => Promise<SnapshotFetcherResult>;
export type SnapshotFetcherResult = Uint8Array | ArrayBuffer | ArrayBufferView | Response | ReadableStream<Uint8Array | ArrayBufferView | ArrayBuffer>;
export interface RunMatInitOptions {
    snapshot?: RunMatSnapshotSource;
    enableGpu?: boolean;
    enableJit?: boolean;
    verbose?: boolean;
    telemetryConsent?: boolean;
    telemetryId?: string;
    wgpuPowerPreference?: "auto" | "high-performance" | "low-power";
    wgpuForceFallbackAdapter?: boolean;
    wasmModule?: WasmInitInput;
    fsProvider?: RunMatFilesystemProvider;
    plotCanvas?: HTMLCanvasElement;
    scatterTargetPoints?: number;
    surfaceVertexBudget?: number;
    emitFusionPlan?: boolean;
}
export type FigureEventKind = "created" | "updated" | "cleared" | "closed";
export interface FigureLegendEntry {
    label: string;
    plotType: string;
    color: [number, number, number, number];
}
export interface FigureEvent {
    handle: number;
    kind: FigureEventKind;
    axesRows: number;
    axesCols: number;
    plotCount: number;
    axesIndices: number[];
    title?: string;
    xLabel?: string;
    yLabel?: string;
    gridEnabled?: boolean;
    legendEnabled?: boolean;
    legendEntries?: FigureLegendEntry[];
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
export type InputRequest = {
    kind: "line";
    prompt: string;
    echo: boolean;
} | {
    kind: "keyPress";
    prompt: string;
    echo: boolean;
};
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
    waitingMs: number;
}
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
export type WorkspaceMaterializeSelector = string | {
    name?: string;
    previewToken?: string;
};
export interface MaterializeVariableOptions {
    limit?: number;
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
    setInputHandler(handler: InputHandler | null): Promise<void>;
    resumeInput(requestId: string, value: ResumeInputValue): Promise<ExecuteResult>;
    pendingStdinRequests(): Promise<PendingStdinRequest[]>;
    materializeVariable(selector: WorkspaceMaterializeSelector, options?: MaterializeVariableOptions): Promise<MaterializedVariable>;
    setFusionPlanEnabled(enabled: boolean): void;
}
interface NativeInitOptions {
    snapshotBytes?: Uint8Array;
    snapshotStream?: ReadableStream<Uint8Array | ArrayBufferView | ArrayBuffer>;
    snapshotUrl?: string;
    enableGpu?: boolean;
    enableJit?: boolean;
    verbose?: boolean;
    telemetryConsent?: boolean;
    telemetryId?: string;
    wgpuPowerPreference?: string;
    wgpuForceFallbackAdapter?: boolean;
    scatterTargetPoints?: number;
    surfaceVertexBudget?: number;
    emitFusionPlan?: boolean;
}
interface RunMatNativeSession {
    execute(source: string): ExecuteResult;
    resetSession(): void;
    stats(): SessionStats;
    clearWorkspace(): void;
    dispose?: () => void;
    telemetryConsent(): boolean;
    memoryUsage?: () => MemoryUsage;
    telemetryClientId?: () => string | undefined;
    gpuStatus(): GpuStatus;
    cancelExecution?: () => void;
    setInputHandler?: (handler: InputHandler | null) => void;
    resumeInput?: (requestId: string, value: ResumeInputWireValue) => ExecuteResult;
    pendingStdinRequests?: () => PendingStdinRequest[];
    materializeVariable?: (selector: WorkspaceMaterializeSelectorWire, options?: MaterializeVariableOptionsWire) => MaterializedVariable;
    setFusionPlanEnabled?: (enabled: boolean) => void;
}
interface ResumeInputWireValue {
    kind?: "line" | "keyPress";
    value?: string;
    error?: string;
}
type WorkspaceMaterializeSelectorWire = string | {
    name?: string;
    token?: string;
    previewToken?: string;
};
interface MaterializeVariableOptionsWire {
    limit?: number;
}
interface RunMatNativeModule {
    default: (module?: WasmInitInput | Promise<WasmInitInput>) => Promise<unknown>;
    initRunMat(options: NativeInitOptions): Promise<RunMatNativeSession>;
    registerFsProvider?: (provider: RunMatFilesystemProvider) => void;
    registerPlotCanvas?: (canvas: HTMLCanvasElement) => Promise<void>;
    deregisterPlotCanvas?: () => void;
    plotRendererReady?: () => boolean;
    registerFigureCanvas?: (handle: number, canvas: HTMLCanvasElement) => Promise<void>;
    deregisterFigureCanvas?: (handle: number) => void;
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
    resumeInput?: (requestId: string, value: ResumeInputWireValue) => ExecuteResult;
    pendingStdinRequests?: () => PendingStdinRequest[];
}
export declare function initRunMat(options?: RunMatInitOptions): Promise<RunMatSessionHandle>;
export declare function attachPlotCanvas(canvas: HTMLCanvasElement): Promise<void>;
export declare function plotRendererReady(): Promise<boolean>;
export declare function registerFigureCanvas(handle: number, canvas: HTMLCanvasElement): Promise<void>;
export declare function deregisterPlotCanvas(): Promise<void>;
export declare function deregisterFigureCanvas(handle: number): Promise<void>;
export declare function onFigureEvent(listener: FigureEventListener | null): Promise<void>;
export declare function subscribeStdout(listener: StdoutListener): Promise<number>;
export declare function unsubscribeStdout(id: number): Promise<void>;
export declare function figure(handle?: number): Promise<number>;
export declare function newFigureHandle(): Promise<number>;
export declare function currentFigureHandle(): Promise<number>;
export declare function setHoldMode(mode?: HoldMode): Promise<boolean>;
export declare function hold(mode?: HoldMode): Promise<boolean>;
export declare function holdOn(): Promise<boolean>;
export declare function holdOff(): Promise<boolean>;
export declare function configureSubplot(rows: number, cols: number, index?: number): Promise<void>;
export declare function subplot(rows: number, cols: number, index?: number): Promise<void>;
export declare function clearFigure(handle?: number): Promise<number>;
export declare function closeFigure(handle?: number): Promise<number>;
export declare function currentAxesInfo(): Promise<AxesInfo>;
export declare function renderFigureImage(options?: FigureImageOptions): Promise<Uint8Array>;
declare function coerceFigureError(value: unknown): FigureBindingError;
type SnapshotStream = ReadableStream<Uint8Array | ArrayBufferView | ArrayBuffer>;
interface SnapshotResolution {
    bytes?: Uint8Array;
    stream?: SnapshotStream;
}
declare function resolveSnapshotSource(source?: RunMatSnapshotSource): Promise<SnapshotResolution>;
declare function fetchSnapshotFromUrl(url: string): Promise<Uint8Array>;
declare function normalizeResumeInputValue(input: ResumeInputValue): ResumeInputWireValue;
export declare const __internals: {
    resolveSnapshotSource: typeof resolveSnapshotSource;
    fetchSnapshotFromUrl: typeof fetchSnapshotFromUrl;
    coerceFigureError: typeof coerceFigureError;
    normalizeResumeInputValue: typeof normalizeResumeInputValue;
    workspaceHover: Record<string, unknown>;
    setNativeModuleOverride(module: RunMatNativeModule | RunMatNativeSession | null): void;
};
//# sourceMappingURL=index.d.ts.map