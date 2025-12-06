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

export interface RunMatFilesystemProvider {
  readFile(path: string): Uint8Array | ArrayBuffer;
  writeFile(path: string, data: Uint8Array | ArrayBuffer | ArrayBufferView): void;
  removeFile(path: string): void;
  metadata(path: string): RunMatFilesystemMetadata;
  symlinkMetadata?(path: string): RunMatFilesystemMetadata;
  readDir(path: string): RunMatFilesystemDirEntry[];
  canonicalize?(path: string): string;
  createDir?(path: string): void;
  createDirAll?(path: string): void;
  removeDir?(path: string): void;
  removeDirAll?(path: string): void;
  rename?(from: string, to: string): void;
  setReadonly?(path: string, readonly: boolean): void;
}
