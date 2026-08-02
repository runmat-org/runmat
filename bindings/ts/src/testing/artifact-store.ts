import type { BrowserTestArtifactStore } from "./runner.js";
import type { RenderedTestReport } from "./types.js";

export interface ProjectTestArtifactWriter {
  write(path: string, bytes: Uint8Array, mediaType: string): Promise<void>;
}

export class ProjectTestArtifactStore implements BrowserTestArtifactStore {
  constructor(
    private readonly writer: ProjectTestArtifactWriter,
    private readonly root = ".runmat/test-results"
  ) {}

  put(runId: string, report: RenderedTestReport): Promise<void> {
    return this.writer.write(
      `${this.root}/${safeSegment(runId)}/${safeName(report.name)}`,
      report.bytes,
      report.mediaType
    );
  }
}

export class IndexedDbTestArtifactStore implements BrowserTestArtifactStore {
  constructor(private readonly databaseName = "runmat-test-artifacts") {}

  async put(runId: string, report: RenderedTestReport): Promise<void> {
    const database = await openDatabase(this.databaseName);
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction("reports", "readwrite");
      transaction.objectStore("reports").put({
        key: `${safeSegment(runId)}/${safeName(report.name)}`,
        runId,
        name: report.name,
        mediaType: report.mediaType,
        bytes: report.bytes,
        createdAt: Date.now()
      });
      transaction.oncomplete = () => resolve();
      transaction.onerror = () =>
        reject(transaction.error ?? new Error("failed to store test report"));
      transaction.onabort = transaction.onerror;
    });
    database.close();
  }
}

export class DownloadTestArtifactStore implements BrowserTestArtifactStore {
  async put(runId: string, report: RenderedTestReport): Promise<void> {
    if (typeof document === "undefined") {
      throw new Error("download artifacts require a browser document");
    }
    const url = URL.createObjectURL(
      new Blob([new Uint8Array(report.bytes).buffer as ArrayBuffer], {
        type: report.mediaType
      })
    );
    try {
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${safeSegment(runId)}-${safeName(report.name).replaceAll("/", "-")}`;
      anchor.click();
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

function openDatabase(name: string): Promise<IDBDatabase> {
  if (typeof indexedDB === "undefined") {
    return Promise.reject(new Error("IndexedDB is unavailable"));
  }
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains("reports")) {
        request.result.createObjectStore("reports", { keyPath: "key" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error ?? new Error("failed to open test artifact database"));
  });
}

function safeSegment(value: string): string {
  return value.replaceAll(/[^A-Za-z0-9._-]/g, "_");
}

function safeName(value: string): string {
  const normalized = value.replaceAll("\\", "/");
  if (
    !normalized ||
    normalized.startsWith("/") ||
    normalized.split("/").includes("..") ||
    normalized.includes("\0")
  ) {
    throw new Error(`unsafe test artifact name '${value}'`);
  }
  return normalized;
}
