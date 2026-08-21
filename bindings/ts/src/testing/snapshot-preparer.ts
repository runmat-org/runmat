import type { RunMatFilesystemProvider } from "../fs/provider-types.js";
import { dirname, normalizePath } from "../fs/path.js";

import type {
  BrowserPreparedTestRun,
  BrowserTestFilesystemEntry,
  BrowserTestPrepareRequest,
  ProjectTestLayout,
  RunMatTestPreparationNative,
  RunMatTestPreparationSession,
  SavedRunSource
} from "./types.js";

const MAX_SNAPSHOT_FILES = 10_000;
const MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024;

/**
 * Browser host adapter for immutable test preparation.
 *
 * Rust parses project test configuration, freezes source identity, performs
 * semantic discovery, applies selectors, and constructs the plan. This adapter
 * only snapshots bytes from the host filesystem in deterministic path order.
 */
export class BrowserTestSnapshotPreparer {
  constructor(
    private readonly native: RunMatTestPreparationNative,
    private readonly session: RunMatTestPreparationSession,
    private readonly filesystem: RunMatFilesystemProvider
  ) {}

  async prepare(request: BrowserTestPrepareRequest): Promise<BrowserPreparedTestRun> {
    const manifestPath = normalizePath(request.manifestPath);
    const projectRoot = dirname(manifestPath);
    const manifestBytes = normalizeBytes(await this.filesystem.readFile(manifestPath));
    const manifestContent = new TextDecoder("utf-8", { fatal: true }).decode(manifestBytes);
    const layout = await this.native.projectTestLayout({
      manifestPath,
      manifestContent
    });
    const targets = await this.targets(projectRoot, layout, request.targets ?? []);
    const catalogRoots = uniquePaths([
      ...targets,
      ...layout.sourceRoots.map((path) => resolvePath(projectRoot, path)),
      ...layout.testPaths.map((path) => resolvePath(projectRoot, path))
    ]);
    const captured = await this.captureFiles(projectRoot, catalogRoots);
    const savedSources = captured.savedSources;
    if (savedSources.length === 0) {
      throw new Error("No MATLAB source files were found in the selected test inputs");
    }
    const revision = revisionFields(request.projectRevision);
    const snapshot = await this.native.freezeTestSnapshot({
      graphDigest: revision.graphDigest,
      baseSourceDigest: revision.sourceRevision,
      testConfigDigest: layout.testConfigDigest,
      savedSources,
      unsavedBuffers: request.unsavedBuffers ?? []
    });
    const prepared = await this.session.prepareTests(snapshot, {
      names: request.names ?? [],
      tags: request.tags ?? [],
      source_prefixes: targets.map((path) => relativePrefix(projectRoot, path)),
      excluded_tags: request.excludedTags ?? []
    });
    return {
      ...prepared,
      filesystemSnapshot: mergeFilesystemEntries([
        {
          path: manifestPath,
          bytes: manifestBytes
        },
        ...captured.filesystemSnapshot
      ])
    };
  }

  private async targets(
    projectRoot: string,
    layout: ProjectTestLayout,
    requested: string[]
  ): Promise<string[]> {
    const targets =
      requested.length > 0
        ? requested.map((path) => resolvePath(projectRoot, path))
        : layout.testRoots.length > 0
          ? layout.testRoots.map((path) => resolvePath(projectRoot, path))
          : (await pathIsDirectory(this.filesystem, resolvePath(projectRoot, "tests")))
            ? [resolvePath(projectRoot, "tests")]
            : [projectRoot];
    const unique = uniquePaths(targets);
    for (const target of unique) {
      await this.filesystem.metadata(target);
    }
    return unique;
  }

  private async captureFiles(
    projectRoot: string,
    roots: string[]
  ): Promise<{
    savedSources: SavedRunSource[];
    filesystemSnapshot: BrowserTestFilesystemEntry[];
  }> {
    const files = new Set<string>();
    for (const root of roots) {
      await collectRegularFiles(this.filesystem, root, files);
    }
    if (files.size > MAX_SNAPSHOT_FILES) {
      throw new Error(
        `Browser test snapshot exceeds the ${MAX_SNAPSHOT_FILES}-file limit`
      );
    }
    const sources: SavedRunSource[] = [];
    const filesystemSnapshot: BrowserTestFilesystemEntry[] = [];
    let totalBytes = 0;
    for (const path of [...files].sort()) {
      const bytes = normalizeBytes(await this.filesystem.readFile(path));
      totalBytes += bytes.byteLength;
      if (totalBytes > MAX_SNAPSHOT_BYTES) {
        throw new Error(
          `Browser test snapshot exceeds the ${MAX_SNAPSHOT_BYTES}-byte limit`
        );
      }
      filesystemSnapshot.push({ path, bytes });
      if (path.toLowerCase().endsWith(".m")) {
        sources.push({
          owner_identity: "path:workspace",
          relative_path: relativePath(projectRoot, path),
          content: new TextDecoder("utf-8", { fatal: true }).decode(bytes)
        });
      }
    }
    return { savedSources: sources, filesystemSnapshot };
  }
}

async function collectRegularFiles(
  filesystem: RunMatFilesystemProvider,
  path: string,
  files: Set<string>
): Promise<void> {
  const metadata = filesystem.symlinkMetadata
    ? await filesystem.symlinkMetadata(path)
    : await filesystem.metadata(path);
  if (metadata.fileType === "symlink") return;
  if (metadata.fileType === "file") {
    files.add(normalizePath(path));
    return;
  }
  if (metadata.fileType !== "dir" && metadata.fileType !== "directory") return;
  const entries = [...(await filesystem.readDir(path))].sort((left, right) =>
    left.fileName.localeCompare(right.fileName)
  );
  for (const entry of entries) {
    await collectRegularFiles(
      filesystem,
      normalizePath(entry.path || `${path}/${entry.fileName}`),
      files
    );
  }
}

function mergeFilesystemEntries(
  entries: BrowserTestFilesystemEntry[]
): BrowserTestFilesystemEntry[] {
  const byPath = new Map<string, Uint8Array>();
  for (const entry of entries) byPath.set(normalizePath(entry.path), entry.bytes);
  return [...byPath]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([path, bytes]) => ({ path, bytes }));
}

async function pathIsDirectory(
  filesystem: RunMatFilesystemProvider,
  path: string
): Promise<boolean> {
  try {
    const metadata = await filesystem.metadata(path);
    return metadata.fileType === "dir" || metadata.fileType === "directory";
  } catch (error) {
    if (isNotFound(error)) return false;
    throw error;
  }
}

function revisionFields(value: unknown): {
  graphDigest: string;
  sourceRevision: string;
} {
  if (typeof value !== "object" || value === null) {
    throw new Error("Resolved project revision is required for browser test preparation");
  }
  const revision = value as Record<string, unknown>;
  const graphDigest = revision["graph_digest"] ?? revision["graphDigest"];
  const sourceRevision = revision["source_revision"] ?? revision["sourceRevision"];
  if (typeof graphDigest !== "string" || typeof sourceRevision !== "string") {
    throw new Error("Resolved project revision is missing graph or source identity");
  }
  return { graphDigest, sourceRevision };
}

function resolvePath(root: string, path: string): string {
  return normalizePath(path.startsWith("/") ? path : `${root}/${path}`);
}

function relativePath(root: string, path: string): string {
  const normalizedRoot = normalizePath(root);
  const normalizedPath = normalizePath(path);
  const prefix = normalizedRoot === "/" ? "/" : `${normalizedRoot}/`;
  if (!normalizedPath.startsWith(prefix)) {
    throw new Error(`Test source escapes the project root: ${normalizedPath}`);
  }
  return normalizedPath.slice(prefix.length);
}

function relativePrefix(root: string, path: string): string {
  const relative = relativePath(root, path);
  return relative && !relative.toLowerCase().endsWith(".m") ? `${relative}/` : relative;
}

function uniquePaths(paths: string[]): string[] {
  return [...new Set(paths.map(normalizePath))].sort();
}

function normalizeBytes(value: Uint8Array | ArrayBuffer): Uint8Array {
  return value instanceof Uint8Array ? value : new Uint8Array(value);
}

function isNotFound(error: unknown): boolean {
  if (typeof error === "object" && error !== null && "code" in error) {
    if ((error as { code?: unknown }).code === "ENOENT") return true;
  }
  const message = error instanceof Error ? error.message : String(error);
  return message.toLowerCase().includes("not found") || message.includes("404");
}
