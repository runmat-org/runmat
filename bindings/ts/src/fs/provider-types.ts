import type { AtomicCreateResult } from "../generated/collaboration-contracts";
import type {
  WorkspaceCaptureRequest, WorkspaceSnapshot, WorkspacePublicationRequest,
  WorkspacePublicationOutcome,
} from "../generated/fea-workspace-contracts.js";
export type RunMatFsFileType = "file" | "dir" | "directory" | "symlink" | "other" | "unknown";

export interface RunMatFilesystemMetadata {
  fileType: RunMatFsFileType;
  len: number;
  modified?: number;
  readonly?: boolean;
}

export interface RunMatFilesystemDirEntry {
  path: string;
  fileName: string;
  fileType?: RunMatFsFileType;
}

export type RunMatMaybePromise<T> = T | Promise<T>;

export interface RunMatOpenFileDialogFilter {
  patterns: string[];
  description?: string;
}

export interface RunMatOpenFileDialogRequest {
  title?: string;
  defaultPath?: string;
  filters: RunMatOpenFileDialogFilter[];
  multiselect: boolean;
}

export interface RunMatOpenFileDialogSelection {
  paths: string[];
  filterIndex?: number;
}

export interface RunMatFilesystemProvider {
  /** Captures one authoritative inventory, including accepted working documents. */
  captureWorkspace?(request: WorkspaceCaptureRequest): RunMatMaybePromise<WorkspaceSnapshot>;
  /** Receipt lookup is scoped to the exact original conditional publication. */
  lookupWorkspacePublication?(request: WorkspacePublicationRequest): RunMatMaybePromise<WorkspacePublicationOutcome | null>;
  publishWorkspace?(request: WorkspacePublicationRequest): RunMatMaybePromise<WorkspacePublicationOutcome>;
  publishOutput?(path: string, data: Uint8Array, expectedVersionId: string | null): RunMatMaybePromise<string>;
  removeOutput?(path: string, expectedVersionId: string): RunMatMaybePromise<void>;
  atomicCreateNew?(path: string, data: Uint8Array): RunMatMaybePromise<AtomicCreateResult>;
  /** Immutable version read; unavailable versions must never fall back to live bytes. */
  readVersion?(path: string, versionId: string, size: number, snapshotId: string): RunMatMaybePromise<Uint8Array | ArrayBuffer>;
  readFile(path: string): RunMatMaybePromise<Uint8Array | ArrayBuffer>;
  readMany?(paths: string[]): RunMatMaybePromise<Array<Uint8Array | ArrayBuffer | null>>;
  writeFile(path: string, data: Uint8Array | ArrayBuffer | ArrayBufferView): RunMatMaybePromise<void>;
  removeFile(path: string): RunMatMaybePromise<void>;
  metadata(path: string): RunMatMaybePromise<RunMatFilesystemMetadata>;
  symlinkMetadata?(path: string): RunMatMaybePromise<RunMatFilesystemMetadata>;
  readDir(path: string): RunMatMaybePromise<RunMatFilesystemDirEntry[]>;
  canonicalize?(path: string): RunMatMaybePromise<string>;
  createDir?(path: string): RunMatMaybePromise<void>;
  createDirAll?(path: string): RunMatMaybePromise<void>;
  removeDir?(path: string): RunMatMaybePromise<void>;
  removeDirAll?(path: string): RunMatMaybePromise<void>;
  rename?(from: string, to: string): RunMatMaybePromise<void>;
  /** Replace a compatible destination in one provider-owned rename.
   * Reject unsupported/type/nonempty-directory cases before mutation.
   * Never emulate this by deleting the destination and then calling rename.
   */
  renameReplace?(from: string, to: string): RunMatMaybePromise<void>;
  setReadonly?(path: string, readonly: boolean): RunMatMaybePromise<void>;
  selectFileOpen?(
    request: RunMatOpenFileDialogRequest,
  ): RunMatMaybePromise<RunMatOpenFileDialogSelection | string | string[] | null | undefined | false>;
}
