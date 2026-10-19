import { dirname, normalizePath } from "./path.js";
import { MemoryVolume, type VolumeSnapshotEntry } from "./memory.js";
import type { BrowserWorkspaceReceipt } from "./browser-workspace.js";

/** One storage transaction owns each read or mutation, including across windows.
 * Callbacks are synchronous: hashing and checking happen before admission. */
export class IndexedDbVolumeStore {
  private pending = new Set<Promise<unknown>>();
  private closed = false;
  constructor(private readonly db: IDBDatabase, private readonly entries: string,
    private readonly receipts: string, private readonly now?: () => number) {}

  /** Ordinary reads fetch only requested records (and direct children for a
   * listing), rather than deserializing unrelated CAD files or receipt history. */
  read<T>(paths: readonly string[], children: boolean, operation: (volume: MemoryVolume) => T): Promise<T> {
    if (this.closed) return Promise.reject(new Error("IndexedDB filesystem is closed"));
    const result = new Promise<T>((resolve, reject) => {
      const tx = this.db.transaction(this.entries, "readonly");
      const store = tx.objectStore(this.entries);
      const loaded = new Map<string, VolumeSnapshotEntry>();
      const requested = new Set<string>();
      let pending = 0;
      let value: T;
      let failure: unknown;
      const fetch = (path: string, listChildren: boolean) => {
        if (requested.has(path)) return;
        requested.add(path); ++pending;
        const request = store.get(path);
        request.onsuccess = () => {
          const entry: VolumeSnapshotEntry | undefined = request.result;
          if (entry) {
            loaded.set(path, entry);
            if (listChildren && entry.kind === "dir") {
              for (const child of entry.children ?? []) fetch(normalizePath(`${path}/${child}`), false);
            }
          }
          if (--pending === 0) {
            try {
              const volume = new MemoryVolume({ now: this.now });
              volume.load([...loaded.values()]);
              value = operation(volume);
            } catch (error) { failure = error; tx.abort(); }
          }
        };
      };
      if (!paths.length) {
        const volume = new MemoryVolume({ now: this.now });
        try { value = operation(volume); } catch (error) { failure = error; tx.abort(); }
      } else for (const path of paths) {
        const normalized = normalizePath(path);
        fetch(normalized, children);
        if (normalized !== "/") {
          let parent = dirname(normalized);
          for (;;) {
            fetch(parent, false);
            if (parent === "/") break;
            parent = dirname(parent);
          }
        }
      }
      tx.oncomplete = () => resolve(value);
      tx.onabort = tx.onerror = () => reject(failure ?? tx.error ?? new Error("IndexedDB read failed"));
    });
    this.pending.add(result);
    void result.then(() => this.pending.delete(result), () => this.pending.delete(result));
    return result;
  }

  transaction<T>(write: boolean, operation: (volume: MemoryVolume, receipts: Map<string, BrowserWorkspaceReceipt>) => T): Promise<T> {
    if (this.closed) return Promise.reject(new Error("IndexedDB filesystem is closed"));
    const result = new Promise<T>((resolve, reject) => {
      const tx = this.db.transaction([this.entries, this.receipts], write ? "readwrite" : "readonly");
      const store = tx.objectStore(this.entries);
      const receiptStore = tx.objectStore(this.receipts);
      const files = store.getAll();
      const receiptRequest = receiptStore.getAll();
      let value: T;
      let failure: unknown;
      let ready = 0;
      const apply = () => {
        if (++ready !== 2) return;
        try {
          const volume = new MemoryVolume({ now: this.now });
          const original: VolumeSnapshotEntry[] = files.result;
          volume.load(original);
          const receipts = new Map<string, BrowserWorkspaceReceipt>(receiptRequest.result.map((receipt: BrowserWorkspaceReceipt) => [receipt.key, receipt]));
          value = operation(volume, receipts);
          if (write) {
            const next = volume.serialize();
            const paths = new Set(next.map(entry => entry.path));
            for (const entry of original) if (!paths.has(entry.path)) store.delete(entry.path);
            const previous = new Map(original.map(entry => [entry.path, entry]));
            for (const entry of next) if (!equalEntry(previous.get(entry.path), entry)) store.put(entry);
            const oldReceipts = new Set<string>(receiptRequest.result.map((receipt: BrowserWorkspaceReceipt) => receipt.key));
            for (const [key, receipt] of receipts) if (!oldReceipts.has(key)) receiptStore.put(receipt);
          }
        } catch (error) { failure = error; tx.abort(); }
      };
      files.onsuccess = apply;
      receiptRequest.onsuccess = apply;
      tx.oncomplete = () => resolve(value);
      tx.onabort = tx.onerror = () => reject(failure ?? tx.error ?? new Error("IndexedDB transaction failed"));
    });
    this.pending.add(result);
    void result.then(() => this.pending.delete(result), () => this.pending.delete(result));
    return result;
  }
  async flush(): Promise<void> { await Promise.all([...this.pending]); }
  close(): void { this.closed = true; this.db.close(); }
}

function equalEntry(a: VolumeSnapshotEntry | undefined, b: VolumeSnapshotEntry): boolean {
  if (!a || a.kind !== b.kind || a.readonly !== b.readonly || a.modified !== b.modified) return false;
  if (b.kind === "file") return !!a.data && !!b.data && a.data.length === b.data.length && a.data.every((byte, index) => byte === b.data![index]);
  const children = [...(a.children ?? [])].sort();
  const other = [...(b.children ?? [])].sort();
  return children.length === other.length && children.every((name, index) => name === other[index]);
}

export function openIndexedDbVolume(idb: IDBFactory, name: string, store: string, version: number): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = idb.open(name, Math.max(2, version));
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(store)) request.result.createObjectStore(store, { keyPath: "path" });
      if (!request.result.objectStoreNames.contains(`${store}_workspace_receipts`)) request.result.createObjectStore(`${store}_workspace_receipts`, { keyPath: "key" });
    };
    request.onsuccess = () => {
      const db = request.result;
      db.onversionchange = () => db.close();
      resolve(db);
    };
    request.onerror = () => reject(request.error ?? new Error("IndexedDB open failed"));
    request.onblocked = () => reject(new Error("IndexedDB upgrade blocked by another connection"));
  });
}
