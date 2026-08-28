export type PackageCacheRevision = number | bigint;

export interface PackageCacheSnapshot {
  revision: PackageCacheRevision;
  state: unknown;
}

export interface PackageCacheObjectWrite {
  object: unknown;
  bytes?: number[] | Uint8Array | null;
}

export interface PackageCacheTransaction {
  expected_revision: PackageCacheRevision;
  next_state: unknown;
  writes: Record<string, PackageCacheObjectWrite> | Map<string, PackageCacheObjectWrite>;
  deletes: string[] | Set<string>;
}

export type PackageCacheCommitOutcome =
  | { outcome: "committed"; revision: PackageCacheRevision }
  | { outcome: "conflict"; actual_revision: PackageCacheRevision };

/**
 * Storage-only port consumed by RunMat's portable Rust cache policy.
 *
 * Implementations atomically compare the revision and publish state plus payload changes.
 * They do not interpret package objects, leases, pins, materialization, or GC policy.
 */
export interface RunMatPackageCacheProvider {
  snapshot(): Promise<PackageCacheSnapshot | null>;
  initialize(initial: PackageCacheSnapshot): Promise<PackageCacheSnapshot>;
  commit(transaction: PackageCacheTransaction): Promise<PackageCacheCommitOutcome>;
  readObjectBytes(digest: string): Promise<Uint8Array | null>;
}

export interface PackageCacheStatus {
  schema_version: number;
  object_count: number | bigint;
  objects_by_kind: Record<"blob" | "tree" | "source-index", number | bigint>;
  logical_bytes: number | bigint;
  stored_payload_bytes: number | bigint;
  pin_count: number | bigint;
  lease_count: number | bigint;
  corruption_count: number | bigint;
  materialization_count: number | bigint;
}

export interface PackageCacheGcPlan {
  delete: string[];
  reclaim_bytes: number | bigint;
}

export interface PackageCacheLease {
  id: string;
  owner: string;
  objects: string[];
  acquired_at_ms: number | bigint;
  expires_at_ms: number | bigint;
  generation: number | bigint;
}
