import type { RunMatFilesystemProvider } from "./provider-types.js";
export interface RemoteProviderOptions {
    baseUrl: string;
    authToken?: string;
    headers?: Record<string, string>;
    chunkBytes?: number;
    timeoutMs?: number;
}
export declare function createRemoteFsProvider(options: RemoteProviderOptions): RunMatFilesystemProvider;
//# sourceMappingURL=remote.d.ts.map