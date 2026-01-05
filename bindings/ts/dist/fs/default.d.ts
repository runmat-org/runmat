import type { RunMatFilesystemProvider } from "./provider-types.js";
/**
 * Picks the best filesystem backend available in the current runtime.
 * Preference order: IndexedDB (browser) → in-memory fallback.
 */
export declare function createDefaultFsProvider(): Promise<RunMatFilesystemProvider>;
//# sourceMappingURL=default.d.ts.map