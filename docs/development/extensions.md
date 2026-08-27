# Native Extension ABI

RunMat's native adapters share a versioned C ABI. The boundary is defined by the dependency-free `runmat-extension-abi` crate and its checked-in C header, [`runmat_extension.h`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-extension-abi/include/runmat_extension.h). It is the common base for native FFI, MEX, Java, and future language adapters; each adapter adds its own source-level API above this layer.

The ABI exposes opaque value and foreign-resource handles rather than Rust structs. Extensions access values through host functions, return structured status records, and declare the capabilities they require and provide. Loading succeeds only when the ABI major version is compatible and the host can satisfy every required capability. The negotiated capability set is the intersection of the host and extension declarations.

Foreign resources are registered with a host and generation. Cloning a RunMat foreign value retains the same managed lease, and the final clone releases the registered resource once. Restarting or removing a host invalidates its outstanding handles. Runtime checks also enforce the declared type, ownership, lifetime, capability, and thread or process affinity before a resource is used.

Buffer transfer is ownership-explicit. ABI 1.1 extensions use `borrow_buffer_lease` to receive a read-only view and a generation-fenced lease handle. The data and shape pointers remain valid until `release_buffer`, even if the source value handle is released first. Value operations and lease creation stay on the originating runtime thread; a worker-thread call returns an affinity error. A created lease owns only thread-safe host storage, so native extensions may read it and release it from a worker without moving a general RunMat value or garbage-collected handle across threads. The host advertises `BUFFER_LEASES` only when it can uphold that contract. Incompatible layouts and isolated-process boundaries use an explicit conversion or snapshot rather than borrowing process-local memory.

## C MEX modules

Native RunMat sessions can build and call C MEX source through the MATLAB-facing `mex` function or the shell command:

```matlab
mex("-R2018a", "gateway.c", "support.c", "-output", "native_gateway")
```

```bash
runmat mex --R2018a -o native_gateway gateway.c support.c
```

The default is the `-R2017b` separate-complex, large-array API. `-R2018a` selects the interleaved-complex API. The legacy `-largeArrayDims` and `-compatibleArrayDims` selections are also supported; the latter exposes 32-bit `mwSize` and `mwIndex` while RunMat checks and translates values at its private native-width host boundary. These four API selections are mutually exclusive.

The checked-in [`C_MATRIX_API` and `C_MEX_API` catalogs](https://github.com/runmat-org/runmat/blob/main/crates/runmat-mex/src/compatibility.rs) list the source-compatible C entrypoints in each API generation. The boundary preserves dense numeric classes, logicals, separate and interleaved complex values, sparse matrices, UTF-16 character arrays, cells, structs, and value objects with public properties. Compatible dense numeric and logical arrays, `-R2018a` floating-complex arrays, and native-width sparse buffers retain their host allocation across an in-process call. Floating-point, logical, interleaved floating-complex, and native-width sparse-index storage transferred through `mxSetData`, `mxSetIr`, or `mxSetJc` is adopted when RunMat's allocation registry proves its origin, size, capacity, alignment, element layout, and release function. Fixed-width integer setter storage currently takes one classified conversion into RunMat's exact integer owner; ordinary integer inputs, outputs created through the Matrix API, and unchanged integer outputs still retain their allocation. A registered allocation with another incompatible layout also uses one checked conversion; an unregistered pointer is rejected because the host cannot prove its size or release function. Separate/interleaved complex changes, 32-bit sparse-index pins, character encoding, and device placement are explicit conversion boundaries as well. A MEX module receives the documented C layout, never a Rust collection layout.

Input pointers are invocation-scoped read-only aliases. Mutating an input obtained through a compatibility API violates the extension contract. Writable outputs and explicit duplicates apply RunMat's copy-on-write value semantics, and no pointer may outlive the array or lease that owns it.

Each successful build writes a canonical `.runmat.json` manifest beside the platform MEX module. The manifest binds the module name and bytes to its target triple, Matrix API selection, compiler family, embedded SDK revision, and RunMat MEX host ABI. Its content identity can be carried in executable and package interop manifests for capability admission and cache validation. Moving a module does not change its identity, while changing the module or its compatibility contract does.

Loaded modules belong to the current session. `clear mex`, `clear functions`, `clear all`, and named `clear` requests unload eligible modules and run registered `mexAtExit` handlers. A locked or currently executing module stays loaded. Native MEX loading is not available in a browser/WASM runtime; capability checks report that boundary before native execution.

RunMat-built modules carry the exact manifest described above and run in the owning process. A supported module built for RunMat's compatibility interface can also run without that manifest in the same `runmat` executable's isolated extension-host mode. The driver and host authenticate each other over bounded local pipes; large canonical values move through verified, session-owned snapshots; callbacks return to the originating runtime context. A native crash, cancellation, or configured timeout terminates the host without terminating the driver. Set `unmanifested = "deny"` under `[runtime.foreign.mex]` to reject this tier before host startup.

| Binary tier | Admission | Execution | Supported targets |
| --- | --- | --- | --- |
| Exact RunMat artifact | The `.runmat.json` manifest matches the module bytes, target, API selection, compiler family, SDK revision, and private host ABI. | Owning process | macOS arm64 `.mexmaca64`, macOS x86-64 `.mexmaci64`, Linux x86-64 `.mexa64`, and Windows x86-64 `.mexw64` |
| RunMat-compatible unmanifested artifact | The platform loader accepts the image and its dependencies; the suffix matches the current target; the module exports the current RunMat adapter ABI and a supported Matrix API mode. | Isolated extension-host process | The same four native targets; GNU-like system ABI on macOS and Linux, and MSVC-compatible system ABI on Windows |
| Other native binary | A target, dependency, adapter symbol, or private ABI check fails. | Not invoked | Rebuild from source with `runmat mex` for one of the supported targets. |

Isolation contains native crashes and enables termination on timeout or cancellation. It is not a security sandbox: an admitted module runs native code with the extension-host process's operating-system permissions. The unmanifested tier does not claim compatibility with binaries built for another host's private exports. Suffix, architecture, adapter, API, and dependency failures produce structured diagnostics before module code is invoked whenever the platform loader can report them safely.

Calls made through `mexCallMATLAB`, `mexEvalString`, and the workspace APIs re-enter the exact RunMat session and active workspace that invoked the gateway. A callback may invoke other functions, including other MEX modules. Recursive entry into the same C MEX module is rejected with a structured error instead of deadlocking its persistent state. Cancellation is checked before RunMat enters native code and again when a gateway calls back into RunMat. An in-process C gateway cannot be preempted safely while it is executing; timeout and crash containment belong to the isolated extension-host policy.

`mexLock` blocks interactive clearing, but it does not extend a module beyond its owning session. Session and standalone-program shutdown force final teardown and run `mexAtExit` with valid callback services. Invocation-only workspace frames are released after each gateway call rather than being retained by persistent native state.

## Native and browser products

Executable and package products carry a deterministic interoperability manifest. Native hosts validate the manifest against registered adapter versions, capabilities, and artifact identities before execution.

WebAssembly products carry the same manifest, so a browser can reject unavailable requirements before starting a program. Portable browser adapters may run in-process. A host-bridge requirement runs only when the browser host and adapter both advertise that bridge. Native-only requirements produce an explicit capability error; RunMat does not silently emulate or remotely execute them.

## Header and compatibility

The public query symbol is `runmat_extension_query_v1`. Both host and extension vtables begin with an ABI version and structure size so later compatible revisions can append fields. ABI 1.1 appends the leased-buffer callbacks after the complete 1.0 host-vtable prefix; the original `borrow_buffer` slot remains in place for binary layout compatibility. Extension code should use the constants and layouts in the public header and must treat every context, instance, cancellation token, value handle, buffer lease, and foreign-resource handle as opaque.

The Rust crate contains layout, version, query-symbol, header-parity, and capability-negotiation tests. The repository architecture check also keeps the ABI crate dependency-free and prevents references to runtime, value, VM, JIT, or AOT internals from entering the public boundary.
