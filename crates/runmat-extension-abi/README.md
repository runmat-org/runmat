# RunMat extension ABI

`runmat-extension-abi` defines the versioned C boundary between RunMat and native extensions. It contains opaque handles, value views, capability negotiation, invocation records, and host/extension vtables. The checked-in header at `include/runmat_extension.h` is the C-facing form of the same contract.

The crate deliberately has no dependency on `runmat-value`, the VM, JIT/AOT internals, runtime services, or adapter implementations. MEX, native FFI, Java, and future language adapters compose this ABI through the runtime-owned foreign subsystem rather than exposing Rust layouts.
