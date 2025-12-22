import type { MaterializedVariable, RunMatSessionHandle, WorkspaceEntry, WorkspaceSnapshot } from "./index.js";
type Disposable = {
    dispose(): void;
};
type HoverContent = {
    value: string;
};
type HoverResult = {
    contents: HoverContent[];
} | null;
type HoverProvider = {
    provideHover(model: TextModel, position: Position): HoverResult | Promise<HoverResult>;
};
type HoverRegistration = (languageId: string, provider: HoverProvider) => Disposable;
type WordInfo = {
    word: string;
} | null;
type TextModel = {
    getWordAtPosition(position: Position): WordInfo;
};
type Position = {
    lineNumber: number;
    column: number;
};
export interface WorkspaceHoverOptions {
    monaco: {
        languages: {
            registerHoverProvider: HoverRegistration;
        };
    };
    language?: string;
    session?: RunMatSessionHandle;
    previewLimit?: number;
    formatter?: (state: HoverFormatState) => string;
}
export interface HoverFormatState {
    name: string;
    entry: WorkspaceEntry;
    materialized?: MaterializedVariable;
    previewFromMaterialize: boolean;
}
export interface WorkspaceHoverController {
    updateWorkspace(snapshot: WorkspaceSnapshot): void;
    dispose(): void;
}
declare class WorkspaceMetadataStore {
    private entries;
    private version;
    private materializedByToken;
    private pending;
    applySnapshot(snapshot: WorkspaceSnapshot): void;
    getEntry(name: string): WorkspaceEntry | undefined;
    clear(): void;
    ensureMaterialized(entry: WorkspaceEntry, session: RunMatSessionHandle, previewLimit: number | undefined): Promise<MaterializedVariable | undefined>;
}
declare function formatShape(shape: number[]): string;
declare function formatBytes(size: number): string;
declare function formatPreview(values: number[], truncated: boolean): string;
declare function formatNumber(value: number): string;
export declare function createWorkspaceHoverProvider(options: WorkspaceHoverOptions): WorkspaceHoverController;
export declare const __internals: {
    WorkspaceMetadataStore: typeof WorkspaceMetadataStore;
    formatShape: typeof formatShape;
    formatBytes: typeof formatBytes;
    formatPreview: typeof formatPreview;
    formatNumber: typeof formatNumber;
};
export {};
//# sourceMappingURL=workspace-hover.d.ts.map