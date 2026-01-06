import { createDefaultFsProvider } from "./fs/default.js";
import { __internals as workspaceHoverInternals } from "./workspace-hover.js";
import { installWebGpuCompatibilityShims } from "./webgpu-shims.js";
export { createInMemoryFsProvider, createIndexedDbFsHandle, createIndexedDbFsProvider, createDefaultFsProvider, createRemoteFsProvider, MemoryVolume } from "./fs/index.js";
export { createWorkspaceHoverProvider } from "./workspace-hover.js";
export { createFusionPlanAdapter } from "./fusion-plan.js";
let loadPromise = null;
let nativeModuleOverride = null;
async function loadNativeModule(wasmModule) {
    installWebGpuCompatibilityShims();
    if (nativeModuleOverride) {
        if (isNativeSession(nativeModuleOverride)) {
            return {
                default: async () => { },
                initRunMat: async () => nativeModuleOverride
            };
        }
        return nativeModuleOverride;
    }
    if (!loadPromise) {
        loadPromise = (async () => {
            const native = (await import("../pkg/runmat_wasm.js"));
            if (typeof native.default === "function") {
                await native.default(wasmModule);
            }
            return native;
        })();
    }
    return loadPromise;
}
export async function initRunMat(options = {}) {
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
    const supportsWebGpu = typeof navigator !== "undefined" && typeof navigator.gpu !== "undefined";
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
        telemetryId: options.telemetryId,
        wgpuPowerPreference: options.wgpuPowerPreference ?? "auto",
        wgpuForceFallbackAdapter: options.wgpuForceFallbackAdapter ?? false,
        scatterTargetPoints: options.scatterTargetPoints,
        surfaceVertexBudget: options.surfaceVertexBudget,
        emitFusionPlan: options.emitFusionPlan ?? false,
        languageCompat: options.language?.compat
    });
    return new WebRunMatSession(session);
}
export async function attachPlotCanvas(canvas) {
    const native = await loadNativeModule();
    if (typeof native.registerPlotCanvas !== "function") {
        throw new Error("The loaded runmat-wasm module does not support WebGPU plotting yet.");
    }
    await native.registerPlotCanvas(canvas);
}
export async function plotRendererReady() {
    const native = await loadNativeModule();
    if (typeof native.plotRendererReady !== "function") {
        return false;
    }
    return native.plotRendererReady();
}
export async function registerFigureCanvas(handle, canvas) {
    const native = await loadNativeModule();
    if (typeof native.registerFigureCanvas !== "function") {
        throw new Error("The loaded runmat-wasm module does not support figure-specific canvases yet.");
    }
    await native.registerFigureCanvas(handle, canvas);
}
export async function resizeFigureCanvas(handle, widthPx, heightPx) {
    const native = await loadNativeModule();
    requireNativeFunction(native, "resizeFigureCanvas");
    native.resizeFigureCanvas(handle, widthPx, heightPx);
}
export async function renderCurrentFigureScene(handle) {
    const native = await loadNativeModule();
    requireNativeFunction(native, "renderCurrentFigureScene");
    native.renderCurrentFigureScene(handle);
}
export async function deregisterPlotCanvas() {
    const native = await loadNativeModule();
    requireNativeFunction(native, "deregisterPlotCanvas");
    native.deregisterPlotCanvas();
}
export async function deregisterFigureCanvas(handle) {
    const native = await loadNativeModule();
    requireNativeFunction(native, "deregisterFigureCanvas");
    native.deregisterFigureCanvas(handle);
}
export async function onFigureEvent(listener) {
    const native = await loadNativeModule();
    if (typeof native.onFigureEvent !== "function") {
        throw new Error("The loaded runmat-wasm module does not expose figure events yet.");
    }
    native.onFigureEvent(listener ? (event) => listener(event) : null);
}
export async function subscribeStdout(listener) {
    const native = await loadNativeModule();
    requireNativeFunction(native, "subscribeStdout");
    return native.subscribeStdout((entry) => listener(entry));
}
export async function unsubscribeStdout(id) {
    const native = await loadNativeModule();
    requireNativeFunction(native, "unsubscribeStdout");
    native.unsubscribeStdout(id);
}
export async function subscribeRuntimeLog(listener) {
    const native = await loadNativeModule();
    requireNativeFunction(native, "subscribeRuntimeLog");
    return native.subscribeRuntimeLog((entry) => listener(entry));
}
export async function unsubscribeRuntimeLog(id) {
    const native = await loadNativeModule();
    requireNativeFunction(native, "unsubscribeRuntimeLog");
    native.unsubscribeRuntimeLog(id);
}
export async function subscribeTraceEvents(listener) {
    const native = await loadNativeModule();
    requireNativeFunction(native, "subscribeTraceEvents");
    return native.subscribeTraceEvents((entries) => listener(entries));
}
export async function unsubscribeTraceEvents(id) {
    const native = await loadNativeModule();
    requireNativeFunction(native, "unsubscribeTraceEvents");
    native.unsubscribeTraceEvents(id);
}
export async function figure(handle) {
    const native = await loadNativeModule();
    if (typeof handle === "number") {
        requireNativeFunction(native, "selectFigure");
        native.selectFigure(handle);
        return handle;
    }
    requireNativeFunction(native, "newFigureHandle");
    return native.newFigureHandle();
}
export async function newFigureHandle() {
    return figure();
}
export async function currentFigureHandle() {
    const native = await loadNativeModule();
    requireNativeFunction(native, "currentFigureHandle");
    return native.currentFigureHandle();
}
export async function setHoldMode(mode = "toggle") {
    const native = await loadNativeModule();
    requireNativeFunction(native, "setHoldMode");
    return native.setHoldMode(mode);
}
export async function hold(mode = "toggle") {
    return setHoldMode(mode);
}
export async function holdOn() {
    return setHoldMode("on");
}
export async function holdOff() {
    return setHoldMode("off");
}
export async function configureSubplot(rows, cols, index = 0) {
    const native = await loadNativeModule();
    requireNativeFunction(native, "configureSubplot");
    try {
        native.configureSubplot(rows, cols, index);
    }
    catch (error) {
        throw coerceFigureError(error);
    }
}
export async function subplot(rows, cols, index = 0) {
    return configureSubplot(rows, cols, index);
}
export async function clearFigure(handle) {
    const native = await loadNativeModule();
    requireNativeFunction(native, "clearFigure");
    try {
        return native.clearFigure(handle ?? null);
    }
    catch (error) {
        throw coerceFigureError(error);
    }
}
export async function closeFigure(handle) {
    const native = await loadNativeModule();
    requireNativeFunction(native, "closeFigure");
    try {
        return native.closeFigure(handle ?? null);
    }
    catch (error) {
        throw coerceFigureError(error);
    }
}
export async function currentAxesInfo() {
    const native = await loadNativeModule();
    requireNativeFunction(native, "currentAxesInfo");
    return native.currentAxesInfo();
}
export async function renderFigureImage(options = {}) {
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
    }
    catch (error) {
        throw coerceFigureError(error);
    }
}
class WebRunMatSession {
    native;
    disposed = false;
    constructor(native) {
        this.native = native;
    }
    ensureActive() {
        if (this.disposed) {
            throw new Error("RunMat session has been disposed");
        }
    }
    async execute(source) {
        this.ensureActive();
        return this.native.execute(source);
    }
    async resetSession() {
        this.ensureActive();
        this.native.resetSession();
    }
    async stats() {
        this.ensureActive();
        return this.native.stats();
    }
    clearWorkspace() {
        this.ensureActive();
        this.native.clearWorkspace();
    }
    dispose() {
        if (this.disposed) {
            return;
        }
        if (typeof this.native.dispose === "function") {
            this.native.dispose();
        }
        this.disposed = true;
    }
    telemetryConsent() {
        this.ensureActive();
        return this.native.telemetryConsent();
    }
    telemetryClientId() {
        this.ensureActive();
        if (typeof this.native.telemetryClientId !== "function") {
            return undefined;
        }
        return this.native.telemetryClientId() ?? undefined;
    }
    async memoryUsage() {
        this.ensureActive();
        if (typeof this.native.memoryUsage !== "function") {
            return { bytes: 0, pages: 0 };
        }
        return this.native.memoryUsage();
    }
    gpuStatus() {
        this.ensureActive();
        return this.native.gpuStatus();
    }
    cancelExecution() {
        if (this.disposed) {
            return;
        }
        if (typeof this.native.cancelExecution === "function") {
            this.native.cancelExecution();
        }
    }
    async setInputHandler(handler) {
        this.ensureActive();
        if (typeof this.native.setInputHandler !== "function") {
            throw new Error("The loaded runmat-wasm module does not expose setInputHandler yet.");
        }
        this.native.setInputHandler(handler ? (request) => handler(request) : null);
    }
    async resumeInput(requestId, value) {
        this.ensureActive();
        requireNativeFunction(this.native, "resumeInput");
        const payload = normalizeResumeInputValue(value);
        return this.native.resumeInput(requestId, payload);
    }
    async pendingStdinRequests() {
        this.ensureActive();
        if (typeof this.native.pendingStdinRequests !== "function") {
            return [];
        }
        return this.native.pendingStdinRequests();
    }
    async materializeVariable(selector, options) {
        this.ensureActive();
        requireNativeFunction(this.native, "materializeVariable");
        const wireSelector = normalizeMaterializeSelector(selector);
        const wireOptions = normalizeMaterializeOptions(options);
        return this.native.materializeVariable(wireSelector, wireOptions);
    }
    setFusionPlanEnabled(enabled) {
        this.ensureActive();
        requireNativeFunction(this.native, "setFusionPlanEnabled");
        this.native.setFusionPlanEnabled(enabled);
    }
    setLanguageCompat(mode) {
        this.ensureActive();
        requireNativeFunction(this.native, "setLanguageCompat");
        this.native.setLanguageCompat(mode);
    }
    async fusionPlanForSource(source) {
        this.ensureActive();
        if (typeof this.native.fusionPlanForSource !== "function") {
            throw new Error("The loaded runmat-wasm module does not expose fusionPlanForSource yet.");
        }
        return this.native.fusionPlanForSource(source) ?? null;
    }
}
function ensureFsProvider(provider) {
    const requiredMethods = [
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
function requireNativeFunction(native, method) {
    if (typeof native[method] !== "function") {
        throw new Error(`The loaded runmat-wasm module does not expose ${String(method)} yet.`);
    }
}
function isFigureErrorPayload(value) {
    return (typeof value === "object" &&
        value !== null &&
        "code" in value &&
        typeof value.code === "string");
}
function coerceFigureError(value) {
    if (isFigureErrorPayload(value)) {
        const err = new Error(value.message ?? value.code ?? "Figure error");
        err.code = value.code ?? "Unknown";
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
        const err = value;
        if (!err.code) {
            err.code = "Unknown";
        }
        return err;
    }
    const err = new Error(String(value));
    err.code = "Unknown";
    return err;
}
async function resolveFsProvider(provided) {
    if (provided) {
        ensureFsProvider(provided);
        return provided;
    }
    try {
        const autoProvider = await createDefaultFsProvider();
        ensureFsProvider(autoProvider);
        return autoProvider;
    }
    catch (error) {
        console.warn("[runmat] Unable to initialize default filesystem provider.", error);
        return undefined;
    }
}
async function resolveSnapshotSource(source) {
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
            throw new Error("Global fetch API is unavailable; provide snapshot.bytes or snapshot.fetcher instead.");
        }
        const response = await fetch(source.url);
        return coerceResponseForSnapshot(response, source.url);
    }
    return {};
}
async function coerceSnapshotFetcherResult(value) {
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
async function coerceResponseForSnapshot(response, origin) {
    if (!response.ok) {
        const suffix = origin ? ` from ${origin}` : "";
        throw new Error(`Failed to fetch snapshot${suffix} (status ${response.status})`);
    }
    if (response.body) {
        return { stream: response.body };
    }
    const buffer = await response.arrayBuffer();
    return { bytes: new Uint8Array(buffer) };
}
function isReadableStream(value) {
    return typeof ReadableStream !== "undefined" && value instanceof ReadableStream;
}
function isResponse(value) {
    return typeof Response !== "undefined" && value instanceof Response;
}
async function fetchSnapshotFromUrl(url) {
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
    const chunks = [];
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
function toUint8Array(data) {
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
function normalizeResumeInputValue(input) {
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
function normalizeMaterializeSelector(selector) {
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
    const payload = {};
    if (typeof selector.name === "string" && selector.name.trim()) {
        payload.name = selector.name.trim();
    }
    if (!payload.name) {
        throw new Error("materializeVariable selector requires name or previewToken");
    }
    return payload;
}
function normalizeMaterializeOptions(options) {
    if (!options) {
        return undefined;
    }
    const payload = {};
    if (typeof options.limit === "number" && Number.isFinite(options.limit)) {
        const limit = Math.floor(options.limit);
        if (limit > 0) {
            payload.limit = limit;
        }
    }
    return payload;
}
function coerceResumeValue(value) {
    if (value === undefined) {
        return "";
    }
    return String(value);
}
function isResumeInputPayload(value) {
    if (value === null || value === undefined) {
        return false;
    }
    if (typeof value !== "object") {
        return false;
    }
    const payload = value;
    return ("value" in payload ||
        "line" in payload ||
        "kind" in payload ||
        "error" in payload);
}
export const __internals = {
    resolveSnapshotSource,
    fetchSnapshotFromUrl,
    coerceFigureError,
    normalizeResumeInputValue,
    workspaceHover: workspaceHoverInternals,
    setNativeModuleOverride(module) {
        nativeModuleOverride = module;
        if (!module) {
            loadPromise = null;
        }
    }
};
function isNativeSession(value) {
    return Boolean(value && typeof value.execute === "function");
}
//# sourceMappingURL=index.js.map