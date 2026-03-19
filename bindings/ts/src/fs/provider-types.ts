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

export interface RunMatFilesystemProvider {
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
  setReadonly?(path: string, readonly: boolean): RunMatMaybePromise<void>;
}
