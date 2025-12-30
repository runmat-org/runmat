import type { MaterializedVariable, RunMatSessionHandle, WorkspaceEntry, WorkspaceSnapshot } from "./index.js";

type Disposable = { dispose(): void };

type HoverContent = { value: string };

type HoverResult = { contents: HoverContent[] } | null;

type HoverProvider = {
  provideHover(model: TextModel, position: Position): HoverResult | Promise<HoverResult>;
};

type HoverRegistration = (languageId: string, provider: HoverProvider) => Disposable;

type WordInfo = { word: string } | null;

type TextModel = {
  getWordAtPosition(position: Position): WordInfo;
};

type Position = { lineNumber: number; column: number };

export interface WorkspaceHoverOptions {
  monaco: { languages: { registerHoverProvider: HoverRegistration } };
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

interface StoredEntry {
  entry: WorkspaceEntry;
}

interface CachedMaterialized {
  token?: string;
  name: string;
  value: MaterializedVariable;
}

class WorkspaceMetadataStore {
  private entries = new Map<string, StoredEntry>();
  private version = 0;
  private materializedByToken = new Map<string, CachedMaterialized>();
  private pending = new Map<string, Promise<MaterializedVariable | undefined>>();

  applySnapshot(snapshot: WorkspaceSnapshot): void {
    if (snapshot.full || snapshot.version > this.version) {
      if (snapshot.full) {
        this.entries.clear();
        this.materializedByToken.clear();
        this.pending.clear();
      }
      this.version = snapshot.version;
    }

    for (const value of snapshot.values) {
      const existing = this.entries.get(value.name);
      if (existing?.entry.previewToken && existing.entry.previewToken !== value.previewToken) {
        this.materializedByToken.delete(existing.entry.previewToken);
        this.pending.delete(existing.entry.previewToken);
      }
      if (!value.previewToken) {
        this.materializedByToken.delete(value.name);
        this.pending.delete(value.name);
      }
      this.entries.set(value.name, { entry: value });
    }
  }

  getEntry(name: string): WorkspaceEntry | undefined {
    return this.entries.get(name)?.entry;
  }

  clear(): void {
    this.entries.clear();
    this.materializedByToken.clear();
    this.pending.clear();
  }

  async ensureMaterialized(
    entry: WorkspaceEntry,
    session: RunMatSessionHandle,
    previewLimit: number | undefined
  ): Promise<MaterializedVariable | undefined> {
    const token = entry.previewToken ?? entry.name;
    const cached = this.materializedByToken.get(token);
    if (cached) {
      return cached.value;
    }
    const pending = this.pending.get(token);
    if (pending) {
      return pending;
    }
    const selector = entry.previewToken ? { previewToken: entry.previewToken } : { name: entry.name };
    const resolver = session
      .materializeVariable(selector, previewLimit ? { limit: previewLimit } : undefined)
      .then((value) => {
        if (value) {
          this.materializedByToken.set(token, { token: entry.previewToken, name: entry.name, value });
        }
        this.pending.delete(token);
        return value;
      })
      .catch(() => {
        this.pending.delete(token);
        return undefined;
      });
    this.pending.set(token, resolver);
    return resolver;
  }
}

function defaultFormatter(state: HoverFormatState): string {
  const { name, entry, materialized, previewFromMaterialize } = state;
  const lines: string[] = [];
  lines.push(`**${name}**`);
  const dtype = entry.dtype ? ` (${entry.dtype})` : "";
  lines.push(`Class: \`${entry.className}\`${dtype}`);
  if (entry.shape && entry.shape.length > 0) {
    lines.push(`Shape: ${formatShape(entry.shape)}`);
  } else {
    lines.push("Shape: 1×1");
  }
  lines.push(`Residency: ${entry.residency ?? (entry.isGpu ? "gpu" : "cpu")}`);
  if (entry.sizeBytes != null) {
    lines.push(`Size: ${formatBytes(entry.sizeBytes)}`);
  }

  const preview = materialized?.preview ?? entry.preview;
  if (preview && preview.values.length > 0) {
    const previewLabel = previewFromMaterialize ? "Preview (materialized)" : "Preview";
    lines.push(`${previewLabel}: ${formatPreview(preview.values, preview.truncated)}`);
  }

  return lines.join("\n\n");
}

function formatShape(shape: number[]): string {
  if (!shape.length) {
    return "1×1";
  }
  return shape.join("×");
}

function formatBytes(size: number): string {
  if (size < 1024) {
    return `${size} B`;
  }
  const units = ["KB", "MB", "GB", "TB"];
  let value = size / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex++;
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`;
}

function formatPreview(values: number[], truncated: boolean): string {
  const formatted = values.slice(0, 8).map((value) => formatNumber(value));
  const suffix = truncated || values.length > 8 ? " …" : "";
  return `[${formatted.join(", ")}${suffix}]`;
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return value.toString();
  }
  if (Math.abs(value) >= 1 || value === 0) {
    return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  return value.toExponential(2);
}

const IDENTIFIER_REGEX = /^[A-Za-z_][A-Za-z0-9_]*$/;

export function createWorkspaceHoverProvider(options: WorkspaceHoverOptions): WorkspaceHoverController {
  const language = options.language ?? "matlab";
  const formatter = options.formatter ?? defaultFormatter;
  const previewLimit = options.previewLimit;
  const store = new WorkspaceMetadataStore();

  const disposable = options.monaco.languages.registerHoverProvider(language, {
    provideHover: async (model, position) => {
      const wordInfo = model.getWordAtPosition(position);
      if (!wordInfo || !wordInfo.word || !IDENTIFIER_REGEX.test(wordInfo.word)) {
        return null;
      }
      const entry = store.getEntry(wordInfo.word);
      if (!entry) {
        return null;
      }

      let materialized: MaterializedVariable | undefined;
      let previewFromMaterialize = false;
      if (options.session && (!entry.preview || entry.preview.truncated)) {
        materialized = await store.ensureMaterialized(entry, options.session, previewLimit);
        previewFromMaterialize = Boolean(materialized);
      }

      const value = formatter({
        name: wordInfo.word,
        entry,
        materialized,
        previewFromMaterialize,
      });

      return { contents: [{ value }] };
    },
  });

  return {
    updateWorkspace(snapshot) {
      store.applySnapshot(snapshot);
    },
    dispose() {
      store.clear();
      disposable.dispose();
    },
  };
}

export const __internals = {
  WorkspaceMetadataStore,
  formatShape,
  formatBytes,
  formatPreview,
  formatNumber,
};
