# Native Extension ABI

RunMat's native adapters share a versioned C ABI. The boundary is defined by the dependency-free `runmat-extension-abi` crate and its checked-in C header, [`runmat_extension.h`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-extension-abi/include/runmat_extension.h). It is the common base for native FFI, MEX, Java, and future language adapters; each adapter adds its own source-level API above this layer.

The ABI exposes opaque value and foreign-resource handles rather than Rust structs. Extensions access values through host functions, return structured status records, and declare the capabilities they require and provide. Loading succeeds only when the ABI major version is compatible and the host can satisfy every required capability. The negotiated capability set is the intersection of the host and extension declarations.

Foreign resources are registered with a host and generation. Cloning a RunMat foreign value retains the same managed lease, and the final clone releases the registered resource once. Restarting or removing a host invalidates its outstanding handles. Runtime checks also enforce the declared type, ownership, lifetime, capability, and thread or process affinity before a resource is used.

Data transfer is copy-first. An adapter may borrow or adopt storage only when layout compatibility, ownership, process boundaries, and provider residency establish that the operation is safe for the complete lifetime. Isolated adapters use a snapshot transfer rather than borrowing process-local memory.

## C MEX modules

Native RunMat sessions can build and call C MEX source through the MATLAB-facing `mex` function or the shell command:

```matlab
mex("-R2018a", "gateway.c", "support.c", "-output", "native_gateway")
```

```bash
runmat mex --R2018a -o native_gateway gateway.c support.c
```

The default is the `-R2017b` separate-complex, large-array API. `-R2018a` selects the interleaved-complex API. The legacy `-largeArrayDims` and `-compatibleArrayDims` selections are also supported; the latter exposes 32-bit `mwSize` and `mwIndex` while RunMat checks and translates values at its private native-width host boundary. These four API selections are mutually exclusive.

The checked-in [`C_MATRIX_API` and `C_MEX_API` catalogs](https://github.com/runmat-org/runmat/blob/main/crates/runmat-mex/src/compatibility.rs) list the source-compatible C entrypoints in each API generation. The boundary preserves dense numeric classes, logicals, separate and interleaved complex values, sparse matrices, UTF-16 character arrays, cells, structs, and value objects with public properties. RunMat copies values at the boundary unless an ownership rule explicitly permits adoption; a MEX module never receives a Rust value layout.

Each successful build writes a canonical `.runmat.json` manifest beside the platform MEX module. The manifest binds the module name and bytes to its target triple, Matrix API selection, compiler family, embedded SDK revision, and RunMat MEX host ABI. Its content identity can be carried in executable and package interop manifests for capability admission and cache validation. Moving a module does not change its identity, while changing the module or its compatibility contract does.

Loaded modules belong to the current session. `clear mex`, `clear functions`, `clear all`, and named `clear` requests unload eligible modules and run registered `mexAtExit` handlers. A locked or currently executing module stays loaded. Native MEX loading is not available in a browser/WASM runtime; capability checks report that boundary before native execution.

Calls made through `mexCallMATLAB`, `mexEvalString`, and the workspace APIs re-enter the exact RunMat session and active workspace that invoked the gateway. A callback may invoke other functions, including other MEX modules. Recursive entry into the same C MEX module is rejected with a structured error instead of deadlocking its persistent state. Cancellation is checked before RunMat enters native code and again when a gateway calls back into RunMat. An in-process C gateway cannot be preempted safely while it is executing; timeout and crash containment belong to the isolated extension-host policy.

`mexLock` blocks interactive clearing, but it does not extend a module beyond its owning session. Session and standalone-program shutdown force final teardown and run `mexAtExit` with valid callback services. Invocation-only workspace frames are released after each gateway call rather than being retained by persistent native state.

## Native and browser products

Executable and package products carry a deterministic interoperability manifest. Native hosts validate the manifest against registered adapter versions, capabilities, and artifact identities before execution.

WebAssembly products carry the same manifest, so a browser can reject unavailable requirements before starting a program. Portable browser adapters may run in-process. A host-bridge requirement runs only when the browser host and adapter both advertise that bridge. Native-only requirements produce an explicit capability error; RunMat does not silently emulate or remotely execute them.

## Header and compatibility

The public query symbol is `runmat_extension_query_v1`. Both host and extension vtables begin with an ABI version and structure size so later compatible revisions can append fields. Extension code should use the constants and layouts in the public header and must treat every context, instance, cancellation token, value handle, and foreign-resource handle as opaque.

The Rust crate contains layout, version, query-symbol, header-parity, and capability-negotiation tests. The repository architecture check also keeps the ABI crate dependency-free and prevents references to runtime, value, VM, JIT, or AOT internals from entering the public boundary.
