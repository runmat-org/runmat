import {
  captureBrowserPublication, captureBrowserWorkspace, cloneBrowserInventory,
  equalBrowserInventory, mergeBrowserPublication, prepareBrowserPublication,
  publicationDigest, publicationKey, reconcileReceipt,
  type BrowserWorkspaceEntry, type BrowserWorkspaceInventory, type BrowserWorkspaceReceipt,
} from "./browser-workspace.js";
import {
  MAX_WORKSPACE_PUBLICATION_BYTES, MAX_WORKSPACE_PUBLICATION_EDITS,
  type WorkspaceCaptureRequest, type WorkspacePublicationRequest, type WorkspacePublicationOutcome,
} from "../generated/fea-workspace-contracts.js";
import { invalidInput } from "./errors.js";

/** One authority per volume: every mounted handle shares its receipts and CAS. */
export class MemoryWorkspaceAuthority {
  private readonly receipts = new Map<string, BrowserWorkspaceReceipt>();
  constructor(
    private readonly inventory: () => BrowserWorkspaceInventory,
    private readonly commit: (entries: Map<string, BrowserWorkspaceEntry>) => void,
  ) {}
  captureWorkspace(request: WorkspaceCaptureRequest) {
    return captureBrowserWorkspace(cloneBrowserInventory(this.inventory()), request);
  }
  async lookupWorkspacePublication(request: WorkspacePublicationRequest): Promise<WorkspacePublicationOutcome | null> {
    request = captureBrowserPublication(request);
    return reconcileReceipt(this.receipts.get(publicationKey(request)), await publicationDigest(request));
  }
  async publishWorkspace(request: WorkspacePublicationRequest): Promise<WorkspacePublicationOutcome> {
    request = captureBrowserPublication(request);
    const digest = await publicationDigest(request);
    const key = publicationKey(request);
    const prior = reconcileReceipt(this.receipts.get(key), digest);
    if (prior) return prior;
    const original = cloneBrowserInventory(this.inventory());
    const prepared = await prepareBrowserPublication(original, request);
    if (prepared.type === "conflict") return prepared.outcome;
    const concurrent = reconcileReceipt(this.receipts.get(key), digest);
    if (concurrent) return concurrent;
    const live = this.inventory();
    if (!equalBrowserInventory(original, live, request)) {
      const current = await captureBrowserWorkspace(cloneBrowserInventory(live), {
        ...request, max_files: MAX_WORKSPACE_PUBLICATION_EDITS, max_bytes: MAX_WORKSPACE_PUBLICATION_BYTES,
      });
      return { type: "conflict", current: current.revision };
    }
    if (this.receipts.size >= MAX_WORKSPACE_PUBLICATION_EDITS) throw invalidInput("Publication receipt budget exhausted");
    // No awaits between arbitration, replacement and receipt commit.
    this.commit(mergeBrowserPublication(prepared.entries, live, request));
    this.receipts.set(key, prepared.receipt);
    return structuredClone(prepared.receipt.outcome);
  }
}
