type SceneChunkRef = {
  src?: string;
  artifactId?: string;
};

type SceneDataRef = {
  refKind: "runmat-data-array-v1";
  dtype?: string;
  shape?: number[];
  chunks?: SceneChunkRef[];
};

export interface SceneHydrationDiagnostics {
  phase: "resolve" | "hydrate";
  level: "debug" | "warn";
  message: string;
  fields?: Record<string, unknown>;
}

export interface SceneHydrationOptions {
  onDiagnostic?: (event: SceneHydrationDiagnostics) => void;
  verboseDebug?: boolean;
}

export async function hydrateFigureSceneDataRefs(
  sceneBytes: Uint8Array,
  readFile: (path: string) => Promise<Uint8Array>,
  options: SceneHydrationOptions = {}
): Promise<Uint8Array> {
  let payload: unknown;
  try {
    payload = JSON.parse(new TextDecoder().decode(sceneBytes));
  } catch {
    return sceneBytes;
  }
  const changedRef = { value: false };
  const metrics = { refsHydrated: 0, chunksRead: 0 };
  const hydrated = await hydrateNode(payload, readFile, changedRef, metrics, options);
  if (changedRef.value) {
    options.onDiagnostic?.({
      phase: "hydrate",
      level: "debug",
      message: "scene hydration completed",
      fields: {
        refsHydrated: metrics.refsHydrated,
        chunksRead: metrics.chunksRead,
      },
    });
  }
  if (!changedRef.value) {
    return sceneBytes;
  }
  return new TextEncoder().encode(JSON.stringify(hydrated));
}

async function hydrateNode(
  value: unknown,
  readFile: (path: string) => Promise<Uint8Array>,
  changedRef: { value: boolean },
  metrics: { refsHydrated: number; chunksRead: number },
  options: SceneHydrationOptions
): Promise<unknown> {
  if (Array.isArray(value)) {
    const out: unknown[] = [];
    for (const item of value) {
      out.push(await hydrateNode(item, readFile, changedRef, metrics, options));
    }
    return out;
  }
  if (!isRecord(value)) {
    return value;
  }
  if (isSceneDataRef(value)) {
    const hydrated = await hydrateDataRef(value, readFile, metrics, options);
    changedRef.value = true;
    metrics.refsHydrated += 1;
    return hydrated;
  }
  const out: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    out[key] = await hydrateNode(child, readFile, changedRef, metrics, options);
  }
  return out;
}

async function hydrateDataRef(
  ref: SceneDataRef,
  readFile: (path: string) => Promise<Uint8Array>,
  metrics: { refsHydrated: number; chunksRead: number },
  options: SceneHydrationOptions
): Promise<unknown> {
  const chunks = Array.isArray(ref.chunks) ? ref.chunks : [];
  const values: number[] = [];
  for (const chunk of chunks) {
    const bytes = await readChunkBytes(chunk, ref.dtype, readFile, options);
    metrics.chunksRead += 1;
    const payload = JSON.parse(new TextDecoder().decode(bytes)) as { values?: unknown[] };
    for (const entry of payload.values ?? []) {
      if (typeof entry === "number" && Number.isFinite(entry)) {
        values.push(entry);
      }
    }
  }
  const shape = Array.isArray(ref.shape) ? ref.shape : [];
  return shapeValues(values, shape);
}

async function readChunkBytes(
  chunk: SceneChunkRef,
  dtype: string | undefined,
  readFile: (path: string) => Promise<Uint8Array>,
  options: SceneHydrationOptions
): Promise<Uint8Array> {
  const candidates = chunkReadCandidates(chunk, dtype);
  for (const candidate of candidates) {
    try {
      const bytes = await readFile(candidate);
      if (options.verboseDebug) {
        options.onDiagnostic?.({
          phase: "resolve",
          level: "debug",
          message: "scene chunk resolved",
          fields: {
            candidate,
            bytes: bytes.byteLength,
            artifactId: chunk.artifactId ?? null,
            src: chunk.src ?? null,
          },
        });
      }
      return bytes;
    } catch {
      continue;
    }
  }
  options.onDiagnostic?.({
    phase: "resolve",
    level: "warn",
    message: "scene chunk resolution failed",
    fields: {
      artifactId: chunk.artifactId ?? null,
      src: chunk.src ?? null,
      candidates,
    },
  });
  throw new Error(`Scene chunk not found: ${chunk.src ?? chunk.artifactId ?? "<unknown>"}`);
}

function chunkReadCandidates(chunk: SceneChunkRef, dtype: string | undefined): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  const push = (raw: string) => {
    const normalized = raw.trim();
    if (!normalized || seen.has(normalized)) {
      return;
    }
    seen.add(normalized);
    out.push(normalized);
  };

  if (typeof chunk.src === "string") {
    push(chunk.src);
    const stripped = chunk.src.replace(/^\.\//, "");
    push(stripped);
    push(`/${stripped}`);
  }

  if (typeof chunk.artifactId === "string") {
    const hash = parseSha256ArtifactId(chunk.artifactId);
    if (hash) {
      const prefix = hash.slice(0, 2);
      const suffixes = dtype === "f32"
        ? ["f32.chunk.json", "f64.chunk.json", "chunk.json", "json", "bin"]
        : ["f64.chunk.json", "f32.chunk.json", "chunk.json", "json", "bin"];
      for (const suffix of suffixes) {
        const rel = `.artifacts/objects/${prefix}/${hash}.${suffix}`;
        push(rel);
        push(`/${rel}`);
      }
    }
  }

  return out;
}

function shapeValues(values: number[], shape: number[]): unknown {
  if (shape.length === 0) {
    return values;
  }
  if (shape.length === 1) {
    return values;
  }
  if (shape.length === 2) {
    const rows = Math.max(0, Math.floor(shape[0] ?? 0));
    const cols = Math.max(0, Math.floor(shape[1] ?? 0));
    if (rows <= 0 || cols <= 0) {
      return [];
    }
    const matrix: number[][] = [];
    for (let row = 0; row < rows; row += 1) {
      const start = row * cols;
      matrix.push(values.slice(start, start + cols));
    }
    return matrix;
  }
  return values;
}

function parseSha256ArtifactId(value: string): string | null {
  const match = /^sha256:([a-f0-9]{64})$/.exec(value);
  return match?.[1] ?? null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isSceneDataRef(value: Record<string, unknown>): value is SceneDataRef {
  return value.refKind === "runmat-data-array-v1";
}
