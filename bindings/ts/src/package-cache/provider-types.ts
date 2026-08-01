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
