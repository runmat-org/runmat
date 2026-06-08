import type { RunMatFilesystemProvider } from "./provider-types.js";
import { createInMemoryFsProvider } from "./memory.js";
import { createIndexedDbFsProvider } from "./indexeddb.js";

const DEFAULT_PROVIDER_SYMBOL = Symbol.for("runmat.fs.defaultProvider");

/**
 * Picks the best filesystem backend available in the current runtime.
 * Preference order: IndexedDB (browser) → in-memory fallback.
 */
export async function createDefaultFsProvider(): Promise<RunMatFilesystemProvider> {
  const globalState = globalThis as typeof globalThis & {
    [DEFAULT_PROVIDER_SYMBOL]?: Promise<RunMatFilesystemProvider>;
  };
  if (!globalState[DEFAULT_PROVIDER_SYMBOL]) {
    globalState[DEFAULT_PROVIDER_SYMBOL] = createDefaultFsProviderUncached().catch((error) => {
      delete globalState[DEFAULT_PROVIDER_SYMBOL];
      throw error;
    });
  }
  return globalState[DEFAULT_PROVIDER_SYMBOL];
}

async function createDefaultFsProviderUncached(): Promise<RunMatFilesystemProvider> {
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
