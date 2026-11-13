//! Pure candidate preparation shared by browser storage providers. Providers
//! own the atomic compare-and-publish transaction and durable receipt.
import type { WorkspaceSnapshot, WorkspaceCaptureRequest, WorkspacePublicationOutcome, WorkspacePublicationRequest } from "../generated/fea-workspace-contracts.js";
import { MAX_WORKSPACE_PUBLICATION_BYTES, MAX_WORKSPACE_PUBLICATION_EDITS } from "../generated/fea-workspace-contracts.js";
import { workspaceRevisionSchema } from "./workspace-contracts.js";
import { encodeWorkspace } from "./workspace-codec.js";

import { canonicalIndexedDbPath, isDescendant, parentDirectories } from "./workspace-path.js";
import { invalidInput, notFound } from "./errors.js";
type WorkspaceRevision = WorkspaceSnapshot["revision"];

export type BrowserWorkspaceEntry = { kind: "dir"; readonly?: boolean } | { kind: "file"; bytes: Uint8Array; readonly?: boolean };
export type BrowserWorkspaceInventory = ReadonlyMap<string, BrowserWorkspaceEntry>;
export interface BrowserWorkspaceReceipt {
  key: string;
  digest: WorkspaceRevision;
  outcome: Extract<WorkspacePublicationOutcome, { type: "published" }>;
}
const MAX_FILES = MAX_WORKSPACE_PUBLICATION_EDITS;
const MAX_BYTES = MAX_WORKSPACE_PUBLICATION_BYTES;

export function publicationKey(request: WorkspacePublicationRequest): string {
  if (request.id.length !== 16 || request.id.some(value => !Number.isInteger(value) || value < 0 || value > 255)) throw invalidInput("Invalid publication identity");
  return request.id.map(value => value.toString(16).padStart(2, "0")).join("");
}
function validatePublicationBudget(request: WorkspacePublicationRequest): void {
  publicationKey(request);
  workspaceRevisionSchema.parse(request.expected_revision);
  if (request.edits.length > MAX_FILES || request.excluded_subtrees.length > 1024) throw invalidInput("Publication exceeds its item budget");
  let bytes = 0;
  for (const edit of request.edits) if (edit.type === "write") {
    bytes += edit.bytes.length;
    if (bytes > MAX_BYTES) throw invalidInput("Publication exceeds its byte budget");
    if (!(edit.bytes instanceof Uint8Array) && edit.bytes.some(value => !Number.isInteger(value) || value < 0 || value > 255)) throw invalidInput("Invalid publication byte array");
  }
}
/** Own mutable caller buffers before the first asynchronous operation. */
export function captureBrowserPublication(request: WorkspacePublicationRequest): WorkspacePublicationRequest {
  validatePublicationBudget(request);
  return { id: [...request.id], root: request.root, excluded_subtrees: [...request.excluded_subtrees], expected_revision: [...request.expected_revision],
    edits: request.edits.map(edit => {
      switch (edit.type) {
        case "write": return { type: edit.type, path: edit.path, bytes: Uint8Array.from(edit.bytes) };
        case "remove": return { type: edit.type, path: edit.path };
        case "rename": return { type: edit.type, from: edit.from, to: edit.to };
      }
    }) };
}
export async function publicationDigest(request: WorkspacePublicationRequest): Promise<WorkspaceRevision> {
  validatePublicationBudget(request);
  // Stable framing is owned by this provider protocol, not JSON object order.
  return hash([request.id, request.root, request.excluded_subtrees, request.expected_revision,
    request.edits.map(edit => {
      switch (edit.type) {
        case "write": return [edit.type, edit.path, Uint8Array.from(edit.bytes)];
        case "remove": return [edit.type, edit.path];
        case "rename": return [edit.type, edit.from, edit.to];
      }
    })]);
}
export function reconcileReceipt(receipt: BrowserWorkspaceReceipt | undefined, digest: WorkspaceRevision): WorkspacePublicationOutcome | null {
  return receipt ? equalRevision(receipt.digest, digest) ? structuredClone(receipt.outcome) : { type: "id_reused" } : null;
}
export function equalRevision(left: WorkspaceRevision, right: WorkspaceRevision): boolean {
  return left.every((value, index) => value === right[index]);
}
function relative(input: string): string {
  if (!input || input.startsWith("/") || input.length > 1024) throw invalidInput("Workspace paths must be relative project files");
  const canonical = canonicalIndexedDbPath(input);
  if (canonical === "/" || canonical.slice(1) !== input) throw invalidInput("Workspace path is not canonical");
  return input;
}
function scope(inventory: BrowserWorkspaceInventory, request: Pick<WorkspaceCaptureRequest, "root" | "excluded_subtrees">) {
  const root = canonicalIndexedDbPath(request.root);
  if (root !== request.root || inventory.get(root)?.kind !== "dir") throw notFound(root);
  if (request.excluded_subtrees.length > 1024) throw invalidInput("Too many excluded workspace subtrees");
  const excluded = [...new Set(request.excluded_subtrees.map(value => `${root === "/" ? "" : root}/${relative(value)}`))].sort();
  return { root, excluded };
}
function included(path: string, root: string, excluded: readonly string[]): boolean {
  return (path === root || isDescendant(path, root)) && !excluded.some(value => path === value || isDescendant(path, value));
}
function selected(inventory: BrowserWorkspaceInventory, request: Pick<WorkspaceCaptureRequest, "root" | "excluded_subtrees">) {
  const { root, excluded } = scope(inventory, request);
  const entries = [...inventory].filter(([path]) => included(path, root, excluded)).sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0);
  return { root, excluded, entries };
}
export async function captureBrowserWorkspace(inventory: BrowserWorkspaceInventory, request: WorkspaceCaptureRequest): Promise<WorkspaceSnapshot> {
  request = { ...request, excluded_subtrees: [...request.excluded_subtrees] };
  if (!Number.isSafeInteger(request.max_files) || request.max_files < 1 || request.max_files > MAX_FILES
    || !Number.isSafeInteger(request.max_bytes) || request.max_bytes < 1 || request.max_bytes > MAX_BYTES) throw invalidInput("Invalid workspace capture budget");
  const { root, excluded, entries } = selected(inventory, request);
  let bytes = 0; let count = 0;
  for (const [, entry] of entries) if (entry.kind === "file") {
    bytes += entry.bytes.length; ++count;
    if (bytes > request.max_bytes || count > request.max_files) throw invalidInput("Workspace capture exceeds its budget");
  }
  const revision = await hash(["runmat-browser-workspace-v1", root, excluded, entries.map(([path, entry]) =>
    entry.kind === "file" ? [path, entry.kind, !!entry.readonly, entry.bytes] : [path, entry.kind, !!entry.readonly])]);
  const version = revision.map(value => value.toString(16).padStart(2, "0")).join("");
  return { root, excluded_subtrees: [...request.excluded_subtrees], revision,
    files: entries.flatMap(([path, entry]) => entry.kind === "file"
      ? [{ path: path.slice(root === "/" ? 1 : root.length + 1), version, bytes: entry.bytes.slice(), readonly: !!entry.readonly }] : []) };
}
export function cloneBrowserInventory(inventory: BrowserWorkspaceInventory): Map<string, BrowserWorkspaceEntry> {
  return new Map([...inventory].map(([path, entry]) => [path, entry.kind === "file" ? { kind: "file", bytes: entry.bytes.slice(), readonly: entry.readonly } : { kind: "dir", readonly: entry.readonly }]));
}
/** Retain concurrent writes/removals outside the guarded source scope. */
export function mergeBrowserPublication(candidate: Map<string, BrowserWorkspaceEntry>, live: BrowserWorkspaceInventory,
  request: WorkspacePublicationRequest): Map<string, BrowserWorkspaceEntry> {
  const { root, excluded } = scope(live, request);
  for (const path of candidate.keys()) if (!included(path, root, excluded)) candidate.delete(path);
  for (const [path, entry] of live) if (!included(path, root, excluded)) candidate.set(path, entry);
  return candidate;
}
/** Compare the original source under the provider's transaction lock, after all
 * asynchronous hashing. Excluded recovery/journal files do not cause conflicts. */
export function equalBrowserInventory(left: BrowserWorkspaceInventory, right: BrowserWorkspaceInventory, request: WorkspacePublicationRequest): boolean {
  const a = selected(left, request).entries; const b = selected(right, request).entries;
  return a.length === b.length && a.every(([path, entry], index) => {
    const other = b[index];
    if (other?.[0] !== path || entry.kind !== other[1].kind || !!entry.readonly !== !!other[1].readonly) return false;
    if (entry.kind === "dir") return true;
    const compared = other[1];
    return compared.kind === "file" && entry.bytes.length === compared.bytes.length
      && entry.bytes.every((value, offset) => value === compared.bytes[offset]);
  });
}
export async function prepareBrowserPublication(inventory: BrowserWorkspaceInventory, request: WorkspacePublicationRequest): Promise<{
  type: "conflict"; outcome: Extract<WorkspacePublicationOutcome, { type: "conflict" }>
} | { type: "prepared"; entries: Map<string, BrowserWorkspaceEntry>; receipt: BrowserWorkspaceReceipt }> {
  const budget = { ...request, max_files: MAX_FILES, max_bytes: MAX_BYTES };
  const current = await captureBrowserWorkspace(inventory, budget);
  if (!equalRevision(current.revision, request.expected_revision)) return { type: "conflict", outcome: { type: "conflict", current: current.revision } };
  if (!request.edits.length || request.edits.length > MAX_FILES) throw invalidInput("Invalid publication edit count");
  let writtenBytes = 0;
  const next = cloneBrowserInventory(inventory);
  const { root, excluded } = scope(inventory, request);
  const pathOf = (value: string): string => {
    const path = `${root === "/" ? "" : root}/${relative(value)}`;
    if (!included(path, root, excluded)) throw invalidInput("Workspace edit targets an excluded subtree");
    return path;
  };
  const writable = (path: string, required: boolean) => {
    const value = next.get(path);
    if (!value && required) throw notFound(path);
    if (value?.readonly) throw invalidInput("Workspace source is readonly");
    if (value?.kind === "dir") throw invalidInput("Workspace source edit targets a directory");
  };
  const parents = (path: string) => {
    for (const parent of parentDirectories(path)) {
      if (next.get(parent)?.kind === "file") throw invalidInput("Workspace parent is a file");
      if (next.get(parent)?.readonly) throw invalidInput("Workspace parent is readonly");
      if (!next.has(parent)) next.set(parent, { kind: "dir" });
    }
  };
  if (next.get(root)?.readonly) throw invalidInput("Workspace root is readonly");
  for (const edit of request.edits) switch (edit.type) {
    case "write": {
      const path = pathOf(edit.path); writable(path, false); parents(path);
      writtenBytes += edit.bytes.length;
      if (writtenBytes > MAX_BYTES) throw invalidInput("Publication exceeds its byte budget");
      next.set(path, { kind: "file", bytes: Uint8Array.from(edit.bytes) }); break;
    }
    case "remove": { const path = pathOf(edit.path); writable(path, true); next.delete(path); break; }
    case "rename": {
      const from = pathOf(edit.from); const to = pathOf(edit.to); writable(from, true);
      if (next.has(to)) throw invalidInput("Workspace rename destination exists");
      parents(to); const entry = next.get(from);
      if (!entry) throw notFound(from);
      next.delete(from); next.set(to, entry); break;
    }
  }
  const after = await captureBrowserWorkspace(next, budget);
  return { type: "prepared", entries: next, receipt: { key: publicationKey(request), digest: await publicationDigest(request),
    outcome: { type: "published", receipt: { id: request.id, revision: after.revision } } } };
}
async function hash(value: unknown): Promise<WorkspaceRevision> {
  const wire = encodeWorkspace(value);
  const bytes = new Uint8Array(await crypto.subtle.digest("SHA-256", Uint8Array.from(wire).buffer));
  return workspaceRevisionSchema.parse(Array.from(bytes));
}
