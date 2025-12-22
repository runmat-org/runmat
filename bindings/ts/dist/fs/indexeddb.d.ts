import { type InMemoryFsProviderOptions } from "./memory.js";
import type { RunMatFilesystemProvider } from "./provider-types.js";
export interface IndexedDbProviderOptions extends InMemoryFsProviderOptions {
    dbName?: string;
    storeName?: string;
    version?: number;
    flushDebounceMs?: number;
}
export interface IndexedDbFsHandle {
    provider: RunMatFilesystemProvider;
    flush(): Promise<void>;
    close(): void;
}
export declare function createIndexedDbFsHandle(options?: IndexedDbProviderOptions): Promise<IndexedDbFsHandle>;
export declare function createIndexedDbFsProvider(options?: IndexedDbProviderOptions): Promise<RunMatFilesystemProvider>;
//# sourceMappingURL=indexeddb.d.ts.map