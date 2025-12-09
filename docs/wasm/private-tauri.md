# RunMat Desktop Shell (Tauri) Notes

> This file is intentionally kept outside the public plan; copy it into the private desktop repo and delete it from the OSS tree before releasing.

## Goals
- Ship a desktop shell that reuses the wasm runtime/UI stack but delivers native-class filesystem and snapshot performance.
- Expose the host filesystem (and future virtual drives) through a privileged Rust layer without compromising the sandbox guarantees in the renderer.

## Architecture
1. **Renderer (WebView)**
   - Loads the same `@runmat/wasm` bindings as runmat.org.
   - Passes a `FsBindings` object to `initRunMat` that forwards every call to the Tauri plugin via `window.__TAURI__.invoke` / `invoke_binary`.
   - For bulk transfers, the bindings expose handle-based helpers:
     ```ts
     const handle = await fs.openFileHandle(path, "read");
     const chunk = await fs.readChunk(handle, offset, length); // returns Uint8Array
     await fs.closeHandle(handle);
     ```

2. **Tauri Plugin (Rust)**
   - Lives in `src-tauri/`, links directly to `runmat-runtime` so we have the same structs available.
   - Implements a native `FsProvider`:
     ```rust
     struct NativeFsProvider;
     impl FsProvider for NativeFsProvider {
         fn open(&self, path: &Path, mode: OpenMode) -> Result<FsHandle>;
         fn read(&self, handle: FsHandle, buf: &mut [u8]) -> Result<usize>;
         fn write(&self, handle: FsHandle, data: &[u8]) -> Result<usize>;
         // ...
     }
     ```
   - The plugin also exposes high-throughput streaming channels. Preferred order:
     1. `invoke_binary` (Tauri 2) to pass raw bytes.
     2. If not available, set up a shared-memory buffer via `tauri-plugin-fs-extra`.
     3. Fallback to chunked base64 (only for debugging).

3. **Direct Attach Mode**
   - For CLI-equivalent workflows (snapshot builds, CI workloads) we can run `runmat-core` directly inside the privileged Rust process and bypass wasm entirely.
   - The renderer simply posts commands (execute script X, stream output Y). Useful for long-running scripts where IPC overhead would dominate.

## Performance Notes
- Opens should return lightweight opaque handles (`u64` ids) so wasm can issue overlapping reads/writes without reopening files.
- Implement opportunistic caching: frequently-read files (snapshots, stdlib) can stay memory-mapped on the native side; we just stream slices to wasm when needed.
- For writes, prefer write-behind buffers so short bursts don’t block the UI thread.

## Security & Permissions
- File dialogs and sandbox access go through Tauri’s permission system. We should ship with a “workspace directory” concept: user chooses a root folder; we store the allowlist in `tauri.conf.json`.
- Expose telemetry hooks so we know when the desktop shell falls back to slower IPC modes.

## Open Tasks
- [ ] Define the IPC message schema (probably Cap’n Proto or flat binary with a leading opcode + payload length).
- [ ] Implement the handle table + LRU cache on the native side.
- [ ] Add integration tests that stress 1GB reads/writes to ensure throughput stays near native (~GB/s on NVMe).
