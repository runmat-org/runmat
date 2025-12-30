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

type AnyObject = Record<string, unknown>;

declare global {
  // eslint-disable-next-line @typescript-eslint/consistent-type-definitions
  interface AnyGPUAdapter extends AnyObject {
    requestDevice(descriptor?: AnyObject): Promise<unknown>;
    __runmatShimmed?: boolean;
  }
}

export function installWebGpuCompatibilityShims(): void {
  if (installed) {
    return;
  }
  installed = true;
  if (typeof navigator === "undefined") {
    return;
  }
  const gpu = (navigator as unknown as AnyObject).gpu as AnyObject | undefined;
  if (!gpu || typeof gpu.requestAdapter !== "function") {
    return;
  }
  const originalRequestAdapter = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async function (...args: unknown[]) {
    const adapter = (await originalRequestAdapter(...args)) as AnyGPUAdapter | null;
    if (adapter && typeof adapter.requestDevice === "function" && !adapter.__runmatShimmed) {
      shimAdapter(adapter);
    }
    return adapter;
  };
}

function shimAdapter(adapter: AnyGPUAdapter): void {
  const originalRequestDevice = adapter.requestDevice.bind(adapter);
  adapter.requestDevice = function (descriptor?: AnyObject) {
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

function patchDeviceDescriptor(descriptor?: AnyObject): AnyObject | undefined {
  if (!descriptor || typeof descriptor !== "object") {
    return descriptor;
  }
  const requiredLimits = cloneLimits(descriptor.requiredLimits);
  if (!requiredLimits) {
    return descriptor;
  }
  if (
    typeof requiredLimits.maxInterStageShaderComponents === "number" &&
    typeof requiredLimits.maxInterStageShaderVariables === "undefined"
  ) {
    requiredLimits.maxInterStageShaderVariables = requiredLimits.maxInterStageShaderComponents;
    delete requiredLimits.maxInterStageShaderComponents;
    return {
      ...descriptor,
      requiredLimits
    };
  }
  return descriptor;
}

function cloneLimits(limits: unknown): AnyObject | null {
  if (!limits || typeof limits !== "object") {
    return null;
  }
  return { ...(limits as AnyObject) };
}

export const __internals = {
  patchDeviceDescriptor
};

