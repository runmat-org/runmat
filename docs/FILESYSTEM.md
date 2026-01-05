# RunMat Filesystem Strategy

## Goals
- Preserve MATLAB-compatible file I/O APIs (`dlmwrite`, `save`, `load`, etc.) across four deployment modes:
  1. **Native CLI / server:** Native `std::fs` behaviour for local disks and network shares.
  2. **Browser-only:** No native filesystem, must use sandboxed storage.
  3. **Multi-process:** Rust runtime still runs in a sandboxed process but can forward requests to a privileged host process (e.g. Tauri and Electron style multi-process architectures).
  4. **Virtualized/remote storage (e.g., S3-backed “drive”):** File APIs should work against logical paths served by a virtual filesystem provider.
- Provide a uniform abstraction inside the runtime so individual builtins do not need to know whether they are reading from IndexedDB, the local disk, or a cloud bucket.
- Keep the design extensible so future storage backends (OPFS, WebDAV, IPFS, etc.) can plug in without rewriting builtins.

## Architecture Overview

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
   │ Native  │   │ Browser    │   │ Desktop     │   │ Remote/S3       │   │
   │ Std FS  │   │ Storage    │   │ Host Proxy  │   │ Gateway         │   │
   └─────────┘   └────────────┘   └─────────────┘   └─────────────────┘   │
   - std::fs      - IndexedDB       - Native shell    - Signed URL fetch  │
   - mmap         - OPFS            - Native FS       - Chunked streaming │
   - async disk   - In-memory       - Cache layer     - Cred mgmt         │
        │                │                 │                 │            │
        └────────────────┴─────────────────┴─────────────────┴────────────┘
```

### Virtual FS Layer
- Define a `FsProvider` trait (now housed in the dedicated `runmat-filesystem` crate) with async-friendly methods (`open`, `read`, `write`, `seek`, `stat`, `list`, `remove`, `rename`, `mkdir`).
- Provide a core set of implementations:
  - `NativeFsProvider` (default for `cfg(not(target_arch = "wasm32"))`), wrapping `std::fs`, `mmap`, and async disk helpers so the CLI/server build keeps today’s behaviour.
  - `HostBridgeFsProvider` (wasm): compiled into `runmat-wasm` and installed via the exported `registerFsProvider` API (or the `fsProvider` init option). Hosts hand us a synchronous JS object that satisfies the `RunMatFilesystemProvider` contract from `bindings/ts`, and the wasm runtime only sees opaque file handles plus byte buffers.
  - `BrowserFsProvider` variants (memory, IndexedDB, OPFS) compiled into `runmat-wasm`.
  - `RemoteFsProvider` that speaks to REST/S3 and can be reused by both browser and native builds.
  - `SandboxFsProvider` for host-side testing: mounts a temp directory as the virtual root so builtins can run against a hermetic filesystem without touching the developer machine.
- Add a `FsRegistry`/`set_fs_provider` API so embedders (runmat-wasm, host shell) can inject the appropriate provider during initialization.

### Native / CLI Backend
- Keep the current semantics for `runmat-cli` and server environments by wiring `NativeFsProvider` as the default provider.
- Expose knobs for advanced scenarios (e.g., custom root directories, jailed sandboxes) by letting the CLI swap in different providers via command-line flags or environment variables.
- Ensure the provider supports memory-mapped reads/writes and zero-copy `std::io::Read`/`Write` adapters so existing high-throughput paths (MAT-file load/save, snapshot writer) remain fast.

### Browser-Only Backend
- Implement the provider in `runmat-wasm` using a layered approach:
  1. **In-memory FS** for ephemeral sessions and tests (default fallback). Implemented today via `createInMemoryFsProvider` inside `@runmat/wasm`.
  2. **IndexedDB/IDBFS** for persistent storage across reloads. Implemented via `createIndexedDbFsHandle`, which mirrors an in-memory volume and flushes it to IndexedDB in the background.
  3. **OPFS (Origin Private FileSystem)** detection for browsers that support it (Chrome-based). (Planned follow-up; today we default to IndexedDB unless hosts provide a bridge.)
- Provide path normalization and namespacing (e.g., `/user/<session>/...`) to prevent collisions between tabs.
- Surfaced API: JS host passes a `fsProvider` object (see the `RunMatFilesystemProvider` interface exported from `bindings/ts`) to `initRunMat` or calls `registerFsProvider`. Methods are synchronous today (return `Uint8Array`/`ArrayBuffer`, throw on errors, etc.), so browser hosts wrap their async storage of choice (IndexedDB, OPFS) and only call into wasm once the data is ready. Future async plumbing can stream through `wasm-bindgen-futures`, but the contract stays stable.

### Desktop Host Backend (Native Shell)
- The packaged desktop shell embeds the same wasm runtime/UI code but installs a privileged helper process that exposes the filesystem to the sandbox.
- Requirements for the helper:
  - Implement the provider in native code, linking directly to `std::fs` (or platform APIs) for zero-copy reads/writes and access to the OS page cache.
  - Offer a streaming IPC protocol so large files can be transferred in chunks without base64 encoding or extra copies.
  - Provide an optional “direct attach” path for workloads where the privileged process hosts `runmat-core` natively (e.g., snapshot generation).
- On the JS side, the same `fsProvider` interface is implemented, but the desktop shell forwards each call across the IPC bridge into the privileged helper. For bulk transfers we will add handle-based APIs (`openFileHandle`, `readChunk`, etc.) so wasm can stream data efficiently.

### Remote / S3-backed Backend
- Introduce a `RemoteFsProvider` that implements `FsProvider` but internally speaks to a small REST surface (documented below). The service can proxy to S3, handle auth, or stream from any object store.
- Reads/writes support chunking/streaming so large files don’t exhaust wasm memory. The Rust provider uses multi-chunk workers (configurable `chunk_bytes` + `parallel_requests`) and reassembles data locally, allowing us to saturate high-bandwidth NICs on AWS hosts.
- Expose credentials/session tokens through the same `fsProvider` object; wasm runtime stays agnostic. Native builds can reuse this provider to mount remote “drives” without bundling browser glue.
- Browser/Tauri builds install the same contract via the `createRemoteFsProvider` helper in `bindings/ts`. It issues synchronous `XMLHttpRequest`s today (for wasm compatibility) but still streams in fixed chunks to avoid megabyte-scale intermediate buffers.

#### Remote REST Contract
All routes live under a configurable `base_url`. Query parameters are URL encoded, and the service responds with either `application/json` or `application/octet-stream`.

| Route | Method | Purpose |
| --- | --- | --- |
| `/fs/metadata?path=/foo/bar` | `GET` | Returns `{ fileType, len, modified?, readonly? }`. Used before every read and to honor MATLAB metadata calls. |
| `/fs/read?path=/foo/bar&offset=0&length=1048576` | `GET` | Streams binary payload for the requested slice. Clients issue multiple chunked requests concurrently. |
| `/fs/write?path=/foo/bar&offset=0&truncate=true` | `PUT` | Writes a binary chunk at the provided offset. The first chunk sets `truncate=true`; subsequent chunks omit it so the service can assemble data in place. |
| `/fs/file?path=/foo/bar` | `DELETE` | Removes a file. |
| `/fs/dir?path=/foo` | `GET` | Lists directory entries `{ path, fileName, fileType }`. |
| `/fs/mkdir` | `POST` | JSON body `{ path, recursive }` creates directories. |
| `/fs/dir?path=/foo&recursive=true` | `DELETE` | Removes (optionally recursive) directories. |
| `/fs/rename` | `POST` | JSON body `{ from, to }` renames/moves entries. |
| `/fs/set-readonly` | `POST` | JSON body `{ path, readonly }` toggles platform read-only flags. |
| `/fs/canonicalize?path=/foo/bar` | `GET` | Returns `{ path }` with the canonicalized remote path so MATLAB builtins can reflect absolute paths. |

Implementations may add auth headers (Bearer tokens, signed cookies, etc.)—the provider simply forwards the configured header on every call.

## Feature Detection & Selection
- `initRunMat` (and the `@runmat/wasm` wrapper) now accept a `fsProvider` object. Hosts choose which implementation to supply: in-memory for tests, IndexedDB/OPFS for browsers, IPC bridges for desktop shells, or remote adapters.
- Default selection order (implemented today in `createDefaultFsProvider`) is IndexedDB → in-memory. OPFS and remote providers remain future work; the wasm layer simply consumes whichever provider gets registered.
- Desktop builds automatically install the bridge provider that talks to the native helper.
- Remote mode is opt-in; the host’s provider owns REST endpoints/auth headers instead of shoving those through Rust-side init options.

## API Surface Changes
- Define `runmat_runtime::fs::set_provider(Box<dyn FsProvider>)`.
- Update file-related builtins (`dlmwrite`, `load`, `save`, `csvread`, etc.) to call the provider instead of `std::fs`.
- For wasm builds, ensure any `libc`-dependent helpers (e.g., `snprintf`) are behind `cfg(not(target_arch = "wasm32"))` and the wasm path funnels through the provider or errors with “operation unsupported”.

## Testing Strategy
- **Filesystem crate tests:** unit tests in `runmat-filesystem` exercise `copy_file`, `set_readonly`, and provider swapping via the new `replace_provider`/`with_provider` helpers, so we can confidently plug in in-memory or bridge implementations. We will extend these tests with mock providers (memory, remote) as the other backends land.
- **Sandbox provider coverage:** the `SandboxFsProvider` wraps the native filesystem inside an isolated temp root so runtime tests can install an alternate provider and exercise MATLAB builtins (e.g., `mkdir`, `dlmwrite`) without touching the developer’s workspace. These tests ensure the provider abstraction actually mediates real I/O.
- **JS bindings tests:** `bindings/ts` ships Vitest coverage for `createInMemoryFsProvider` and the IndexedDB mirror (`createIndexedDbFsHandle`), ensuring the wasm-facing provider contract behaves the same way the Rust runtime expects.
- **Remote filesystem contract:** The Vitest suite also spins up a worker-isolated HTTP server and drives the synchronous Remote FS provider through a Node-specific XHR bridge so we validate chunked uploads/downloads without stubbing the transport.
- **Builtin coverage:** each filesystem-touching builtin already has an extensive test module. By routing their production code through `runmat_filesystem`, the existing tests (csv/dlm/read/write matrix, MAT load/save, REPL commands) now validate the abstraction automatically while still using the native provider underneath.
- **Provider permutations:** when new providers appear (browser memory store, remote bridge), their crates will expose feature-gated test suites that call the same builtin helpers after installing the provider via `set_provider`. The plan is to add harness helpers (e.g., `with_fs_provider`) so builtin tests can be parameterized over multiple providers without duplicating assertions.
- **CI matrix:** native builds continue to run all builtin tests. Browser/wasm CI will load the wasm provider tests (wasm-bindgen-driven) once the provider exists, giving us confidence that the same Matlab semantics hold across native, browser, and bridge deployments.

## Telemetry & Permissions
- Browser hosts should prompt the user before enabling persistent storage. Provide helper UI in the JS bindings (e.g., `await requestFsAccess()`).
- Log backend choice and capacity via the telemetry hooks so we can analyze adoption.

## Open Questions / Follow-ups
- Do we need POSIX-like locking semantics? (Probably not initially; can be emulated with advisory locks in IndexedDB.)
- Should remote mode support offline caching + sync? (Future enhancement, but design shouldn’t preclude it.)
- How do we expose directory pickers in browsers? (Likely via the File System Access API; needs UX design.)
