# RunMat Filesystem

This guide explains how the RunMat filesystem works, how to choose a backend, and how to use the remote filesystem backed by RunMat Server.

## Overview

RunMat’s filesystem keeps scripts consistent across laptop, browser, desktop, and cloud without requiring runtime scripts to rewrite `load` or `save`. You can scale from a local sandbox to petabytes in the cloud without changing your code.

Replay note: notebook replay artifacts (figure scene/workspace replay payloads) are host-managed
files. The runtime only exports/imports opaque payload bytes; artifact naming, retention, and
storage location are controlled by the host application.

```
           ┌────────────────────┐
           │ RunMat Runtime     │
           │ (wasm/native)      │
           ├────────────────────┤
           │ Virtual FS (VFS)   │  <-- filesystem abstraction
           │  • open/read/write │
           │  • metadata        │
           │  • directory ops   │
           └────────┬───────────┘
                    │
        ┌───────────┴─────────────────────────────────────────────────────┐
        │                Backends                                         │
        │                                                                 │
   ┌────┴────┐   ┌────────────┐   ┌─────────────┐   ┌─────────────────┐   │
   │ Native  │   │ Browser    │   │ Desktop     │   │ Remote          │   │
   │ Std FS  │   │ Storage    │   │ Host Proxy  │   │ Gateway         │   │
   └─────────┘   └────────────┘   └─────────────┘   └─────────────────┘   │
   - std::fs      - IndexedDB       - Native shell    - Signed URL fetch  │
   - mmap         - OPFS            - Native FS       - Chunked streaming │
   - async disk   - In-memory       - Cache layer     - Cred mgmt         │
        │                │                 │                 │            │
        └────────────────┴─────────────────┴─────────────────┴────────────┘
```

Backends include:
- **Native** — local development and low‑latency work with OS filesystem performance.
- **Browser** — zero‑install demos and lightweight workflows where portability matters most.
- **Desktop** — browser UX with native disk performance and enterprise policies.
- **Remote (RunMat Server)** — multi‑TB/PB data, collaboration, and high‑throughput I/O at scale.

## Backend details

### Native
- Best for local development, single‑node workloads, and low‑latency iteration
- Uses the OS filesystem and page cache for peak local performance
- Ideal for small to mid‑sized datasets and quick iteration loops

### Browser
- Best for zero‑install usage, onboarding, and lightweight workflows
- Storage is sandboxed and portable across sessions
- Great for demos and education; not optimized for very large datasets

### Desktop
- Best for teams that want a desktop UX with native filesystem access
- Uses a privileged host process for high‑performance local reads/writes
- Ideal when you need native disk access with a sandboxed UI

### Remote (RunMat Server)
Use RunMat Server when data outgrows local disks or when collaboration and repeatability matter. It stays fast at terabyte‑to‑petabyte scale while preserving I/O semantics within code executing in the runtime.

You get:
- High throughput for big reads and writes
- Elastic scale without managing storage infrastructure
- Copy‑on‑write updates so large datasets evolve without full rewrites
- Versioned datasets when you need traceability, with smart defaults to control storage costs
- Content-addressed blobs with ETags derived from hash + size for integrity

## Using the filesystem from RunMat scripts

```matlab
% Write a dataset
data = rand(1, 1000)
save("/data/example.mat", "data")

% Read it back
load("/data/example.mat")
```

If the runtime is configured with a remote filesystem provider, these calls read and write to the remote storage automatically.

For portable path assembly, use `fullfile` to join segments with the platform-specific separator:

```matlab
rawPath = fullfile("data", "raw", "sample.dat");
fid = fopen(rawPath, "w"); fclose(fid);
```

## Using the CLI with the remote filesystem

### Authenticate and select a project

```sh
runmat login
runmat org list
runmat project list --org <org-id>
runmat project select <project-id>
```

You can pass a private server URL with `--server`. RunMat defaults to `https://api.runmat.com` if omitted.

### Run scripts with the remote filesystem

```sh
runmat remote run /script.m
```

### Basic filesystem operations

```sh
runmat fs ls /data
runmat fs read /data/example.mat --output example.mat
runmat fs write /data/example.mat ./example.mat
runmat fs mkdir /data/new --recursive
runmat fs rm /data/example.mat
```

### Selecting a project

You can select a project by:

- running `runmat project select <project-id>`
- passing `--project <project-id>` to the command
- setting the environment variable `RUNMAT_PROJECT_ID` to the project ID
- providing a project ID when logging in with `runmat login --project <project-id>`

## Versioning policy

Versioning lets you restore a previous file or dataset state without copying data yourself.

What users should expect:
- Source/code and small files are versioned by default.
- Large datasets are versioned when they are sharded or explicitly configured.
- Restoring a version is instant because it switches the active version pointer.

Storage behavior:
- Versioned files keep their previous blobs.
- Non‑versioned updates keep only the latest blob.
- Sharded datasets always version the manifest, so old datasets remain recoverable.

When history is pruned:
- If a file is not configured for versioning, every new write replaces the previous data and the older history is removed automatically.
- When versioning is enabled, RunMat keeps history based on a max‑versions policy. Defaults come from the server plan, and you can override per project.
  - Retention policy is enforced by the background cleanup job and applies after version creation.
- Versions referenced by snapshots are never pruned.

### Using version history from the CLI

```sh
runmat fs history /data/example.mat
runmat fs restore <version-id>
runmat fs history-delete <version-id>
```

### Snapshots (project history)

Snapshots give you a fast, durable “project checkpoint” without duplicating data. They are ideal for:
- Marking a dataset or model before a risky migration
- Creating a reproducible baseline before experiments
- Capturing a stable project state for handoff or review

Snapshots are a single-parent chain (like a simple git history). Restoring a snapshot rewires file pointers back to the recorded versions with zero-copy behavior. Snapshots are only removed when explicitly deleted, and versions referenced by snapshots are never pruned.

Tags let you attach stable names (like `baseline` or `release-2026-01`) to any snapshot for quick retrieval.

```sh
runmat fs snapshot-create --message "baseline" --tag baseline
runmat fs snapshot-list
runmat fs snapshot-restore <snapshot-id>
runmat fs snapshot-tag-list
runmat fs snapshot-tag-set <snapshot-id> release-2026-01
```

### Git sync

RunMat exposes a minimal git-compatible workflow backed by snapshots. It lets you clone a project into a git working tree, pull new snapshot history, and push linear commits back to the server.

Git sync is fast-forward only and currently supports a single branch (`refs/heads/main`).

```sh
runmat fs git-clone ./project-repo
cd project-repo
runmat fs git-pull
runmat fs git-push
```

### Git export

Snapshots can be exported to a git fast-import stream, so you can materialize a git history for backup, sharing, or downstream tooling. Each snapshot becomes a commit, and tags map to git tags.

```sh
curl -L "$RUNMAT_SERVER_URL/v1/projects/$RUNMAT_PROJECT_ID/fs/snapshots/<snapshot-id>/git-export" \
  -H "authorization: Bearer $RUNMAT_API_KEY" \
  -o snapshot.fast-import

git init export
cd export
git fast-import < ../snapshot.fast-import
git log --oneline
```

### Retention settings

```sh
runmat project retention get
runmat project retention set 50
```

## Scaling to petabytes

RunMat Server lets you work with massive datasets without re‑architecting your code. It streams only what you need, parallelizes reads and writes, and supports shard‑based datasets so updates are incremental — not all‑or‑nothing. That means a 1 GB update inside a multi‑PB dataset is still fast and cost‑efficient.

## Sharded datasets and manifests

Sharding splits very large files into smaller pieces (shards) so RunMat can stream them efficiently and update only what changes.

When sharding applies:
- Large datasets above the shard threshold (default: 4 GB).
- Workloads that need fast random access or partial updates.

Why it matters:
- Reads are parallelized across shards for high throughput.
- Updates rewrite only the touched shards, not the entire dataset.

Implementation details you may see in diagnostics:
- The manifest is stored at the dataset path and tagged with `hash=manifest:v1`.
- Shards are stored under `/.runmat/shards/<uuid>` and streamed in order.
- The server computes content hashes; the only client-provided hash is the `manifest:v1` marker.

Manifest schema:

```json
{
  "version": 1,
  "total_size": 123456,
  "shard_size": 536870912,
  "shards": [
    { "path": "/.runmat/shards/<uuid>", "size": 536870912 }
  ]
}
```

### Manifest workflows

```sh
runmat fs manifest-history /data/dataset
runmat fs manifest-restore <version-id>
runmat fs manifest-update /data/dataset --base-version <version-id> --manifest ./manifest.json
```

For high-throughput ingestion, `/fs/manifest/urls` returns presigned URLs for each shard so clients can download in parallel without routing through the server.

### When should I use Remote?
- You need to share datasets across teams or regions
- Your data is too large for local disk
- You want fast, parallel I/O without managing storage
- You need versioned datasets and reproducible workflows

## RunMat Server Configuration

| Value | Description | Required | Default |
| --- | --- | --- | --- |
| `RUNMAT_SERVER_URL` | Base API URL | No | `https://api.runmat.com` |
| `RUNMAT_API_KEY` | API key or access token | Yes (unless using `runmat login`) | None |
| `RUNMAT_ORG_ID` | Default org | No | None |
| `RUNMAT_PROJECT_ID` | Default project | No | None |
| `RUNMAT_FS_SHARD_THRESHOLD_BYTES` | Size at which sharding begins | No | `4294967296` (4 GB) |
| `RUNMAT_FS_SHARD_SIZE_BYTES` | Shard size for large datasets | No | `536870912` (512 MB) |
| `RUNMAT_FS_VERSION_RETENTION_MAX_VERSIONS` | Default history limit per file (0 = unlimited) | No | `0` |
