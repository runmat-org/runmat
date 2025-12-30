/**
 * Temporary WebGPU compatibility shim.
 *
 * Why: wgpu 0.19 now requests the spec-compliant limit
 * `maxInterStageShaderComponents`, but Chrome/Safari stable (as of Dec 2025,
 * Chrome M131 / Safari 18) still only accept the legacy Dawn limit name
 * `maxInterStageShaderVariables`. When the new field is present, the browser
 * rejects every `GPUAdapter.requestDevice` call and hardware acceleration
 * fails entirely.
 *
 * What: we intercept `navigator.gpu.requestAdapter`, wrap the resulting
 * adapters' `requestDevice`, and rewrite `requiredLimits` so the legacy field
 * is populated and the new one is omitted. Once the host browsers accept the
 * renamed limit, this shim becomes a no-op.
 *
 * Removal plan: delete this file (and its import in `index.ts`) after Chrome
 * stable and Safari release builds ship support for
 * `maxInterStageShaderComponents` (Chrome M132+ / a Safari build that includes
 * the same Dawn roll). Track via browser release notes or by observing that
 * production Chrome no longer throws during `requestDevice`.
 */
let installed = false;
export function installWebGpuCompatibilityShims() {
    if (installed) {
        return;
    }
    installed = true;
    if (typeof navigator === "undefined") {
        return;
    }
    const gpu = navigator.gpu;
    if (!gpu || typeof gpu.requestAdapter !== "function") {
        return;
    }
    const originalRequestAdapter = gpu.requestAdapter.bind(gpu);
    gpu.requestAdapter = async function (...args) {
        const adapter = (await originalRequestAdapter(...args));
        if (adapter && typeof adapter.requestDevice === "function" && !adapter.__runmatShimmed) {
            shimAdapter(adapter);
        }
        return adapter;
    };
}
function shimAdapter(adapter) {
    const originalRequestDevice = adapter.requestDevice.bind(adapter);
    adapter.requestDevice = function (descriptor) {
        const patched = patchDeviceDescriptor(descriptor);
        return originalRequestDevice(patched);
    };
    Object.defineProperty(adapter, "__runmatShimmed", {
        value: true,
        configurable: true,
        enumerable: false,
        writable: false
    });
}
function patchDeviceDescriptor(descriptor) {
    if (!descriptor || typeof descriptor !== "object") {
        return descriptor;
    }
    const requiredLimits = cloneLimits(descriptor.requiredLimits);
    if (!requiredLimits) {
        return descriptor;
    }
    if (typeof requiredLimits.maxInterStageShaderComponents === "number" &&
        typeof requiredLimits.maxInterStageShaderVariables === "undefined") {
        requiredLimits.maxInterStageShaderVariables = requiredLimits.maxInterStageShaderComponents;
        delete requiredLimits.maxInterStageShaderComponents;
        return {
            ...descriptor,
            requiredLimits
        };
    }
    return descriptor;
}
function cloneLimits(limits) {
    if (!limits || typeof limits !== "object") {
        return null;
    }
    return { ...limits };
}
export const __internals = {
    patchDeviceDescriptor
};
//# sourceMappingURL=webgpu-shims.js.map