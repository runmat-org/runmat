import type { RunMatFilesystemDirEntry, RunMatFilesystemMetadata, RunMatFilesystemProvider } from "./provider-types.js";
export interface InMemoryFsProviderOptions {
    initialFiles?: Record<string, Uint8Array | ArrayBuffer | ArrayBufferView | string>;
    now?: () => number;
}
export interface VolumeSnapshotEntry {
    path: string;
    kind: "file" | "dir";
    data?: Uint8Array;
    readonly: boolean;
    modified: number;
    children?: string[];
}
export declare class MemoryVolume {
    private readonly opts;
    private readonly entries;
    private readonly now;
    private readonly encoder;
    constructor(opts?: InMemoryFsProviderOptions);
    createProvider(onMutate?: () => void): RunMatFilesystemProvider;
    serialize(): VolumeSnapshotEntry[];
    load(entries: VolumeSnapshotEntry[]): void;
    readFile(path: string): Uint8Array;
    writeFile(path: string, data: Uint8Array, readonly?: boolean, modified?: number): void;
    removeFile(path: string): void;
    metadata(path: string): RunMatFilesystemMetadata;
    readDir(path: string): RunMatFilesystemDirEntry[];
    canonicalize(path: string): string;
    createDir(path: string, failOnExists?: boolean, readonly?: boolean, modified?: number): boolean;
    createDirAll(path: string): boolean;
    removeDir(path: string): void;
    removeDirAll(path: string): void;
    rename(from: string, to: string): void;
    setReadonly(path: string, readonly: boolean): void;
    private coerceBuffer;
    private getFile;
    private getDir;
    private makeDirEntry;
}
export declare function createInMemoryFsProvider(options?: InMemoryFsProviderOptions): RunMatFilesystemProvider;
//# sourceMappingURL=memory.d.ts.map