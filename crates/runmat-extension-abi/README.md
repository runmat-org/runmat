# RunMat extension ABI

`runmat-extension-abi` defines the versioned C boundary between RunMat and native extensions. It contains opaque handles, value views, capability negotiation, invocation records, and host/extension vtables. The checked-in header at `include/runmat_extension.h` is the C-facing form of the same contract.

The crate deliberately has no dependency on `runmat-value`, the VM, JIT/AOT internals, runtime services, or adapter implementations. MEX, native FFI, Java, and future language adapters compose this ABI through the runtime-owned foreign subsystem rather than exposing Rust layouts.

ABI 1.1 adds explicit buffer leases. `borrow_buffer_lease` returns a read-only view plus a generation-fenced handle; the data and shape pointers remain valid until the matching `release_buffer` succeeds, independently of the source value handle. The original 1.0 `borrow_buffer` slot is retained in place for layout compatibility and hosts may leave it unset. New zero-copy consumers should require ABI 1.1 and the `BUFFER_LEASES` capability.

Value inspection, retain/release, and lease creation run on the originating runtime thread. A host returns `RUNMAT_STATUS_AFFINITY_VIOLATION` instead of exposing thread-affine runtime or garbage-collected state to an extension worker. Once created, a read-only lease contains an independent storage owner; RunMat's native host permits `release_buffer` from a worker thread. Extensions must stop their callbacks and worker access before host shutdown.
