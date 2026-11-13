import { expect, it } from "vitest";
import { createInMemoryFsProvider } from "../fs/memory.js";
import type { WorkspaceCaptureRequest, WorkspacePublicationRequest } from "../generated/fea-workspace-contracts.js";
import { BrowserPackageMountFilesystem } from "./mount.js";

const capture: WorkspaceCaptureRequest = { root: "/", excluded_subtrees: [], max_files: 100, max_bytes: 10000 };
const encoder = new TextEncoder();

it("delegates coherent workspace publication and exact receipts through package composition", async () => {
  const workspace = createInMemoryFsProvider();
  await workspace.writeFile("/study.fea", encoder.encode("original"));
  const mounted = new BrowserPackageMountFilesystem(workspace);
  const before = await mounted.captureWorkspace(capture);
  const request: WorkspacePublicationRequest = {
    id: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    root: "/", excluded_subtrees: [], expected_revision: before.revision,
    edits: [{ type: "write", path: "study.fea", bytes: encoder.encode("revised") }, { type: "write", path: "verify.m", bytes: encoder.encode("pure function") }],
  };
  const result = await mounted.publishWorkspace(request);
  expect(result.type).toBe("published");
  expect(await mounted.lookupWorkspacePublication(request)).toEqual(result);
  expect(await mounted.publishWorkspace(request)).toEqual(result);
  expect(new TextDecoder().decode(await workspace.readFile("/study.fea"))).toBe("revised");
  expect(new TextDecoder().decode(await workspace.readFile("/verify.m"))).toBe("pure function");
});

it("rejects the whole batch before mutation when any edit targets an immutable package", async () => {
  const workspace = createInMemoryFsProvider();
  await workspace.writeFile("/study.fea", encoder.encode("original"));
  const mounted = new BrowserPackageMountFilesystem(workspace);
  const digest = `sha256:${"a".repeat(64)}`;
  const root = mounted.registerVolatile({ source: { tree_digest: digest }, tree: { digest, entries: [] } });
  const request: WorkspacePublicationRequest = {
    id: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    root: "/", excluded_subtrees: [], expected_revision: (await mounted.captureWorkspace(capture)).revision,
    edits: [{ type: "write", path: "study.fea", bytes: encoder.encode("must not save") }, { type: "rename", from: "study.fea", to: `${root.slice(1)}/study.fea` }],
  };
  await expect(mounted.publishWorkspace(request)).rejects.toThrow(/read-only/);
  await expect(mounted.lookupWorkspacePublication(request)).rejects.toThrow(/read-only/);
  await expect(mounted.captureWorkspace({ ...capture, root })).rejects.toThrow(/read-only/);
  expect(new TextDecoder().decode(await workspace.readFile("/study.fea"))).toBe("original");
  mounted.clear();
});

it("does not emulate atomic workspace operations when the host lacks them", async () => {
  const workspace = createInMemoryFsProvider();
  workspace.captureWorkspace = undefined;
  const mounted = new BrowserPackageMountFilesystem(workspace);
  await expect(mounted.captureWorkspace(capture)).rejects.toThrow(/capture is unavailable/);
});
