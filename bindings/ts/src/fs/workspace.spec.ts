import { describe, it, expect } from "vitest";
import "fake-indexeddb/auto";
import { createInMemoryFsProvider } from "./memory.js";
import { createIndexedDbFsHandle } from "./indexeddb.js";
import type { RunMatFilesystemProvider } from "./provider-types.js";
import type { WorkspaceCaptureRequest, WorkspacePublicationRequest } from "../generated/fea-workspace-contracts.js";
const capture: WorkspaceCaptureRequest = { root: "/", excluded_subtrees: [".journal"], max_files: 100, max_bytes: 10000 };
const encoder = new TextEncoder();
const text = async (provider: RunMatFilesystemProvider, path: string) => new TextDecoder().decode(await provider.readFile(path));
async function request(provider: RunMatFilesystemProvider, byte = 1): Promise<WorkspacePublicationRequest> {
  return { id: [byte,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], root: "/", excluded_subtrees: [".journal"], expected_revision: (await provider.captureWorkspace!(capture)).revision,
    edits: [{ type: "rename", from: "a.fea", to: "shared.fea" }, { type: "write", path: "main.fea", bytes: Array.from(encoder.encode("new")) }] };
}
for (const kind of ["memory", "indexeddb"] as const) describe(`${kind} workspace authority`, () => {
  it("publishes a coherent rename, retains recovery files and recognizes exact retries", async () => {
    const handle = kind === "indexeddb" ? await createIndexedDbFsHandle({ dbName: crypto.randomUUID() }) : null;
    const provider = handle?.provider ?? createInMemoryFsProvider();
    try {
      await provider.writeFile("/a.fea", encoder.encode("original"));
      await provider.writeFile("/main.fea", encoder.encode("old"));
      const publication = await request(provider);
      await provider.createDir!("/.journal");
      await provider.writeFile("/.journal/recovery", encoder.encode("retained"));
      const result = await provider.publishWorkspace!(publication);
      expect(result.type).toBe("published");
      expect(await text(provider, "/shared.fea")).toBe("original");
      expect(await text(provider, "/main.fea")).toBe("new");
      expect(await text(provider, "/.journal/recovery")).toBe("retained");
      await provider.writeFile("/main.fea", encoder.encode("teammate"));
      expect(await provider.publishWorkspace!(publication)).toEqual(result);
      expect(await text(provider, "/main.fea")).toBe("teammate");
      expect(await provider.lookupWorkspacePublication!({ ...publication, edits: [] })).toEqual({ type: "id_reused" });
    } finally { handle?.close(); }
  });
  it("rejects stale revisions, readonly files, traversal and incomplete candidates without mutations", async () => {
    const handle = kind === "indexeddb" ? await createIndexedDbFsHandle({ dbName: crypto.randomUUID() }) : null;
    const provider = handle?.provider ?? createInMemoryFsProvider();
    try {
      await provider.writeFile("/a.fea", encoder.encode("original"));
      const publication = await request(provider);
      await provider.setReadonly!("/a.fea", true);
      expect((await provider.publishWorkspace!(publication)).type).toBe("conflict");
      const readonly = await request(provider, 2);
      expect((await provider.captureWorkspace!(capture)).files[0]?.readonly).toBe(true);
      await expect(provider.publishWorkspace!(readonly)).rejects.toThrow(/readonly/);
      await provider.setReadonly!("/a.fea", false);
      const valid = await request(provider, 3);
      await expect(provider.publishWorkspace!({ ...valid, edits: [...valid.edits, { type: "remove", path: "missing.fea" }] })).rejects.toThrow();
      await expect(provider.publishWorkspace!({ ...valid, edits: [{ type: "write", path: "../escape.fea", bytes: [] }] })).rejects.toThrow();
      expect(await text(provider, "/a.fea")).toBe("original");
    } finally { handle?.close(); }
  });
  it("arbitrates simultaneous publications without partially publishing a losing candidate", async () => {
    const handle = kind === "indexeddb" ? await createIndexedDbFsHandle({ dbName: crypto.randomUUID() }) : null;
    const provider = handle?.provider ?? createInMemoryFsProvider();
    try {
      await provider.writeFile("/a.fea", encoder.encode("original"));
      const a = await request(provider); const b = { ...a, id: (await request(provider, 2)).id };
      const results = await Promise.all([provider.publishWorkspace!(a), provider.publishWorkspace!(b)]);
      expect(results.map(value => value.type).sort()).toEqual(["conflict", "published"]);
      expect(await text(provider, "/shared.fea")).toBe("original");
    } finally { handle?.close(); }
  });
});
it("persists publication receipts across independent IndexedDB connections and restart", async () => {
  const dbName = crypto.randomUUID();
  const a = await createIndexedDbFsHandle({ dbName }); const b = await createIndexedDbFsHandle({ dbName });
  await a.provider.writeFile("/a.fea", encoder.encode("original"));
  const publication = await request(b.provider);
  const published = await a.provider.publishWorkspace!(publication);
  expect(await b.provider.lookupWorkspacePublication!(publication)).toEqual(published);
  a.close(); b.close();
  const restarted = await createIndexedDbFsHandle({ dbName });
  try {
    await restarted.provider.writeFile("/main.fea", encoder.encode("teammate"));
    expect(await restarted.provider.publishWorkspace!(publication)).toEqual(published);
    expect(await text(restarted.provider, "/main.fea")).toBe("teammate");
  } finally { restarted.close(); }
});
