import { afterEach, describe, expect, it } from "vitest";
import { createIndexedDbPackageCache } from "./indexeddb.js";
import { ImmutableBrowserPackageMount } from "./mount.js";
import type { PackageCacheSnapshot, PackageCacheTransaction } from "./provider-types.js";

const databases: string[] = [];

afterEach(async () => {
  for (const name of databases.splice(0)) {
    await deleteDatabase(name);
  }
});

function databaseName(label: string): string {
  const name = `runmat-package-cache-${label}-${Math.random()}`;
  databases.push(name);
  return name;
}

function initial(): PackageCacheSnapshot {
  return {
    revision: 0,
    state: { schema_version: 1, objects: {} }
  };
}

function transaction(
  revision: number,
  state: unknown,
  digest = "sha256:039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
  bytes = [1, 2, 3]
): PackageCacheTransaction {
  return {
    expected_revision: revision,
    next_state: state,
    writes: {
      [digest]: {
        object: { kind: "blob" },
        bytes
      }
    },
    deletes: []
  };
}

describe("IndexedDB package cache", () => {
  it("atomically initializes, commits payloads, and rejects a stale tab", async () => {
    const dbName = databaseName("conflict");
    const first = await createIndexedDbPackageCache({ dbName });
    const second = await createIndexedDbPackageCache({ dbName });
    expect(await first.provider.snapshot()).toBeNull();
    await first.provider.initialize(initial());
    const left = await first.provider.snapshot();
    const right = await second.provider.snapshot();
    expect(left?.revision).toBe(0);
    expect(right?.revision).toBe(0);

    expect(
      await first.provider.commit(transaction(0, { schema_version: 1, objects: { left: true } }))
    ).toEqual({ outcome: "committed", revision: 1 });
    expect(
      await second.provider.commit(transaction(0, { schema_version: 1, objects: { right: true } }))
    ).toEqual({ outcome: "conflict", actual_revision: 1 });
    expect(
      await second.provider.readObjectBytes(
        "sha256:039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"
      )
    ).toEqual(
      Uint8Array.from([1, 2, 3])
    );
    expect((await second.provider.snapshot())?.state).toEqual({
      schema_version: 1,
      objects: { left: true }
    });
    first.close();
    second.close();
  });

  it("aborts state and payload publication when a worker is interrupted", async () => {
    const dbName = databaseName("abort");
    const handle = await createIndexedDbPackageCache({
      dbName,
      faultInjector: {
        beforeApply: (active) => active.abort()
      }
    });
    await handle.provider.initialize(initial());
    await expect(
      handle.provider.commit(transaction(0, { schema_version: 1, objects: { partial: true } }))
    ).rejects.toBeDefined();
    expect(await handle.provider.snapshot()).toEqual(initial());
    expect(
      await handle.provider.readObjectBytes(
        "sha256:039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"
      )
    ).toBeNull();
    handle.close();
  });

  it("surfaces quota failures without publishing partial state", async () => {
    const dbName = databaseName("quota");
    const handle = await createIndexedDbPackageCache({
      dbName,
      faultInjector: {
        beforeApply: () => {
          throw new DOMException("quota exhausted", "QuotaExceededError");
        }
      }
    });
    await handle.provider.initialize(initial());
    await expect(
      handle.provider.commit(transaction(0, { schema_version: 1, objects: { partial: true } }))
    ).rejects.toMatchObject({ name: "QuotaExceededError" });
    expect(await handle.provider.snapshot()).toEqual(initial());
    handle.close();
  });

  it("treats an evicted payload as a precise immutable-mount miss", async () => {
    const dbName = databaseName("eviction");
    const handle = await createIndexedDbPackageCache({ dbName });
    await handle.provider.initialize(initial());
    await handle.provider.commit(transaction(0, { schema_version: 1, objects: {} }));
    const mount = new ImmutableBrowserPackageMount(
      {
        digest: "sha256:tree",
        entries: [
          {
            path: "src/main.m",
            kind: "file",
            digest:
              "sha256:039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81",
            byte_len: 3,
            executable: false
          },
          {
            path: "main.m",
            kind: "symlink",
            byte_len: 0,
            executable: false,
            link_target: "src/main.m"
          }
        ]
      },
      handle.provider
    );
    expect(await mount.readFile("main.m")).toEqual(Uint8Array.from([1, 2, 3]));
    await deletePayload(
      dbName,
      "sha256:039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"
    );
    await expect(mount.readFile("src/main.m")).rejects.toMatchObject({
      code: "PackageCacheMiss",
      digest: "sha256:039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"
    });
    handle.close();
  });

  it("restores and navigates a large immutable MATLAB source tree", async () => {
    const dbName = databaseName("large-tree");
    const handle = await createIndexedDbPackageCache({ dbName });
    await handle.provider.initialize(initial());
    const sourceCount = 512;
    const sources = await Promise.all(
      Array.from({ length: sourceCount }, async (_, index) => {
        const name = `ecosystem_fn_${index.toString().padStart(4, "0")}.m`;
        const bytes = new TextEncoder().encode(
          `function value = ecosystem_fn_${index.toString().padStart(4, "0")}(input)\nvalue = input + 2;\nend\n`
        );
        return { name, bytes, digest: await digest(bytes) };
      })
    );
    await handle.provider.commit({
      expected_revision: 0,
      next_state: {
        schema_version: 1,
        objects: Object.fromEntries(sources.map((source) => [source.digest, { kind: "blob" }]))
      },
      writes: Object.fromEntries(
        sources.map((source) => [
          source.digest,
          { object: { kind: "blob" }, bytes: source.bytes }
        ])
      ),
      deletes: []
    });
    const mount = new ImmutableBrowserPackageMount(
      {
        digest: "sha256:large-representative-tree",
        entries: [
          {
            path: "src",
            kind: "directory",
            byte_len: 0,
            executable: false
          },
          ...sources.map((source) => ({
            path: `src/${source.name}`,
            kind: "file" as const,
            digest: source.digest,
            byte_len: source.bytes.byteLength,
            executable: false
          }))
        ]
      },
      handle.provider
    );

    expect(mount.readDir("src")).toHaveLength(sourceCount);
    for (const index of [0, 255, 511]) {
      const source = sources[index];
      expect(await mount.readFile(`src/${source.name}`)).toEqual(source.bytes);
    }
    handle.close();

    const reopened = await createIndexedDbPackageCache({ dbName });
    expect((await reopened.provider.snapshot())?.revision).toBe(1);
    expect(await reopened.provider.readObjectBytes(sources[511].digest)).toEqual(
      sources[511].bytes
    );
    reopened.close();
  });

  it("reports a blocked schema upgrade cleanly", async () => {
    const dbName = databaseName("blocked");
    const blocker = await openRawDatabase(dbName, 1);
    await expect(createIndexedDbPackageCache({ dbName, version: 2 })).rejects.toThrow(
      "blocked"
    );
    blocker.close();
  });
});

function openRawDatabase(name: string, version: number): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name, version);
    request.onupgradeneeded = () => {
      const db = request.result;
      db.createObjectStore("meta", { keyPath: "key" });
      db.createObjectStore("payloads", { keyPath: "digest" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function deletePayload(name: string, digest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name);
    request.onsuccess = () => {
      const db = request.result;
      const active = db.transaction("payloads", "readwrite");
      active.objectStore("payloads").delete(digest);
      active.oncomplete = () => {
        db.close();
        resolve();
      };
      active.onerror = () => reject(active.error);
    };
    request.onerror = () => reject(request.error);
  });
}

function deleteDatabase(name: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.deleteDatabase(name);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
    request.onblocked = () => resolve();
  });
}

async function digest(bytes: Uint8Array): Promise<string> {
  const value = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return `sha256:${Array.from(value, (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}
