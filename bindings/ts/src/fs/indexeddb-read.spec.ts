import { it, expect, vi } from "vitest";
import "fake-indexeddb/auto";
import { createIndexedDbFsHandle } from "./indexeddb.js";

it("ordinary reads use keyed requests without loading the whole volume or receipts", async () => {
  const handle = await createIndexedDbFsHandle({ dbName: crypto.randomUUID() });
  try {
    await handle.provider.createDir!("/src");
    await handle.provider.writeFile("/src/main.fea", new TextEncoder().encode("model assembly {}"));
    await handle.provider.writeFile("/unrelated.cad", new Uint8Array(1024 * 1024));
    const getAll = vi.spyOn(IDBObjectStore.prototype, "getAll");
    const get = vi.spyOn(IDBObjectStore.prototype, "get");
    try {
      expect(new TextDecoder().decode(await handle.provider.readFile("/src/main.fea"))).toBe("model assembly {}");
      expect(await handle.provider.readMany!(["/src/main.fea", "/missing"])).toHaveLength(2);
      expect(await handle.provider.metadata("/src/main.fea")).toMatchObject({ len: 17 });
      expect(await handle.provider.canonicalize("/src/./main.fea")).toBe("/src/main.fea");
      expect(await handle.provider.readDir("/src")).toMatchObject([{ fileName: "main.fea" }]);
      expect(getAll).not.toHaveBeenCalled();
      expect(get.mock.calls.some(([key]) => key === "/unrelated.cad")).toBe(false);
    } finally { getAll.mockRestore(); get.mockRestore(); }
  } finally { handle.close(); }
});

it("atomic journal creation owns its bytes before yielding to IndexedDB", async () => {
  const handle = await createIndexedDbFsHandle({ dbName: crypto.randomUUID() });
  try {
    const bytes = new TextEncoder().encode("accepted decision");
    const creating = handle.provider.atomicCreateNew!("/decision", bytes);
    bytes.fill(0);
    expect(await creating).toEqual({ kind: "published", result: { durability: "durable" } });
    expect(new TextDecoder().decode(await handle.provider.readFile("/decision"))).toBe("accepted decision");
  } finally { handle.close(); }
});
