export {
  createInMemoryFsProvider,
  MemoryVolume,
  type InMemoryFsProviderOptions,
  type VolumeSnapshotEntry
} from "./memory.js";

export {
  createIndexedDbFsHandle,
  createIndexedDbFsProvider,
  type IndexedDbFsHandle,
  type IndexedDbProviderOptions
} from "./indexeddb.js";

export { createDefaultFsProvider } from "./default.js";
export { createRemoteFsProvider, type RemoteProviderOptions } from "./remote.js";

export type {
  RunMatFsFileType,
  RunMatFilesystemDirEntry,
  RunMatFilesystemMetadata,
  RunMatFilesystemProvider,
  RunMatOpenFileDialogFilter,
  RunMatOpenFileDialogRequest,
  RunMatOpenFileDialogSelection
} from "./provider-types.js";
