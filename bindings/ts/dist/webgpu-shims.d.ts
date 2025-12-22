type AnyObject = Record<string, unknown>;
declare global {
    interface AnyGPUAdapter extends AnyObject {
        requestDevice(descriptor?: AnyObject): Promise<unknown>;
        __runmatShimmed?: boolean;
    }
}
export declare function installWebGpuCompatibilityShims(): void;
declare function patchDeviceDescriptor(descriptor?: AnyObject): AnyObject | undefined;
export declare const __internals: {
    patchDeviceDescriptor: typeof patchDeviceDescriptor;
};
export {};
//# sourceMappingURL=webgpu-shims.d.ts.map