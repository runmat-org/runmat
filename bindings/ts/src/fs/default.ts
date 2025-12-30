import type { RunMatFilesystemProvider } from "./provider-types.js";
import { createInMemoryFsProvider } from "./memory.js";
import { createIndexedDbFsProvider } from "./indexeddb.js";

/**
 * Picks the best filesystem backend available in the current runtime.
 * Preference order: IndexedDB (browser) → in-memory fallback.
 */
export async function createDefaultFsProvider(): Promise<RunMatFilesystemProvider> {
  if (supportsIndexedDb()) {
    try {
      return await createIndexedDbFsProvider();
    } catch (error) {
      console.warn("[runmat] Failed to init IndexedDB provider, falling back to in-memory.", error);
    }
  }
  return createInMemoryFsProvider();
}

function supportsIndexedDb(): boolean {
  return typeof indexedDB !== "undefined" && typeof IDBDatabase !== "undefined";
}
