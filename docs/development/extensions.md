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

### Modern C++ MEX and Data API

Passing a `.cpp`, `.cc`, `.cxx`, or `.c++` source to `mex` selects a C++17 driver and the interleaved-complex API by default. `CXX` and `CXXFLAGS` select and configure the C++ toolchain; an explicit release pin still takes precedence. Modern gateways include `mex.hpp` and `mexAdapter.hpp`, derive `MexFunction` from `matlab::mex::Function`, and receive `matlab::mex::ArgumentList` inputs and outputs.

The bundled Data API uses shared copy-on-write array controls over the same host buffers as the C Matrix API and ordinary RunMat values. Copying an `Array`, assigning an input to an output, placing an array inside a cell or structure, or passing it through an in-process engine callback retains the underlying allocation. The first mutable access detaches only that value. `ArrayFactory::createBuffer` uses the MEX host allocator, and `createArrayFromBuffer` transfers compatible real, logical, or floating-complex column-major storage into the result without copying. A row-major buffer is reordered once and reported as a memory-layout conversion. The same API provides typed real, logical, character, string, floating- and fixed-width-complex arrays; missing string values; cells; structures; real double and logical sparse arrays; value and handle-object properties; enumerations; engine callbacks; and C++ exception translation. Character encoding and sparse coordinate-to-CSC construction remain explicit conversion boundaries, while ordered sparse value and row-index buffers are adopted directly.

`RunMatEngine` provides synchronous and asynchronous function calls, evaluation, workspace access, and indexed object-property access. The established `MATLABEngine` spelling remains available as a source-compatibility alias. Function calls accept Data API arrays without copying their payloads. Typed overloads also accept and return supported native C++ scalars, vectors, complex values, and UTF strings; those overloads create or extract the requested native representation at the call boundary. `FutureResult` and `SharedFutureResult` support waiting, timed waiting, result sharing, and cooperative cancellation. Native work runs on the session's MEX lane; callbacks return to the runtime task that owns the workspace and object handles. A future may outlive the gateway that created it, and the session retains the module, callback services, and array leases until that work finishes. Supplied UTF-16 output and error stream buffers receive the corresponding captured console streams. Without supplied buffers, console output follows the session's normal output path. Callback failures preserve their identifier and message, and accepted cancellation is reported as `CancelException`.

The modern Data API and the C Matrix API are separate source interfaces. A C++ gateway should not use C Matrix functions to mutate an object managed by a Data API wrapper. The adapter itself uses the private host boundary to implement shared copies, allocator transfer, callbacks, and output publication while keeping those mechanics out of extension source.

### Fortran MEX modules

Passing a `.f`, `.for`, `.f77`, `.f90`, `.f95`, `.f03`, or `.f08` source selects the Fortran gateway. Uppercase suffixes are accepted as well. `FC` selects the compiler, `F77` is its fallback, and `FFLAGS` applies only to Fortran translation units. `CFLAGS` and `CXXFLAGS` remain scoped to C and C++ helper sources in a mixed build. RunMat currently qualifies GNU Fortran drivers whose executable name begins with `gfortran`; another driver is rejected before compilation rather than being given incompatible flags. Fixed- and free-form sources are both supported.

Fortran gateways include `fintrf.h` and use the conventional `mexFunction` subroutine:

```fortran
#include "fintrf.h"
      subroutine mexFunction(nlhs, plhs, nrhs, prhs)
      implicit none
      integer nlhs, nrhs
      mwPointer plhs(*), prhs(*)
      mwPointer mxCreateDoubleScalar

      plhs(1) = mxCreateDoubleScalar(42.0d0)
      return
      end
```

The default remains the `-R2017b` separate-complex, large-array API. `-R2018a` selects interleaved complex storage, and the large- and compatible-array-dimension pins have the same meaning as they do for C. `mwPointer` follows the process pointer width under every pin. `mwSize` and `mwIndex` become 32-bit only under `-compatibleArrayDims`; RunMat range-checks those dimensions and converts sparse indices at the native-width host boundary.

Fortran arrays use the same canonical copy-on-write host buffers, invocation leases, allocation registry, callbacks, diagnostics, workspace, lock, persistence, and teardown paths as C and C++. Passing an input through unchanged retains its compatible dense, interleaved floating-complex, or native-width sparse allocation. Copy routines such as `mxCopyPtrToReal8` perform the explicit copy their names request. Legacy separate-complex and 32-bit sparse-index pins remain explicit representation conversions.

Cell, structure, and object element indices use the Fortran interface's one-based convention, including field numbers returned by `mxGetFieldNumber`. The raw CSC arrays returned by `mxGetIr` and `mxGetJc` remain zero-based storage: row indices start at zero, and the first column pointer is zero. This distinction lets existing Fortran gateways inspect sparse storage without changing its representation.

RunMat compiles the gateway and any helper sources separately, compiles one bundled native-support translation unit, and links the result with the Fortran driver so the language runtime is retained. Extension authors do not need to compile or order RunMat's private support files. The generated manifest records Fortran as the source boundary and otherwise uses the same identity, API-pin, admission, package, and isolation contracts as C and C++ modules. Native Fortran modules are unavailable in browser/WASM sessions; the runtime reports that capability boundary before execution.

Each successful build writes a canonical `.runmat.json` manifest beside the platform MEX module. The manifest binds the module name and bytes to its target triple, Matrix API selection, compiler family, embedded SDK revision, and RunMat MEX host ABI. Its content identity can be carried in executable and package interop manifests for capability admission and cache validation. Moving a module does not change its identity, while changing the module or its compatibility contract does.

Loaded modules belong to the current session. A native library image may contain process-global state, so one RunMat session owns each canonical image in-process at a time. A concurrent session uses an exact manifest-admitted isolated extension host, which gives it independent module state instead of rebinding the first image's globals. Releasing the owning session makes that image eligible for another in-process owner. `clear mex`, `clear functions`, `clear all`, and named `clear` requests unload eligible modules and run registered `mexAtExit` handlers. A locked, currently executing, or asynchronously active module stays loaded. Native MEX loading is not available in a browser/WASM runtime; capability checks report that boundary before native execution.

RunMat-built modules carry the exact manifest described above and run in the owning process. A supported module built for RunMat's compatibility interface can also run without that manifest in the same `runmat` executable's isolated extension-host mode. The driver and host authenticate each other over bounded local pipes; large canonical values move through verified, session-owned snapshots; callbacks return to the originating runtime context. A native crash, cancellation, or configured timeout terminates the host without terminating the driver. Set `unmanifested = "deny"` under `[runtime.foreign.mex]` to reject this tier before host startup.

| Binary tier | Admission | Execution | Supported targets |
| --- | --- | --- | --- |
| Exact RunMat artifact | The `.runmat.json` manifest matches the module bytes, target, API selection, compiler family, SDK revision, and private host ABI. | Owning process; exact isolated host when another session owns the same native image | macOS arm64 `.mexmaca64`, macOS x86-64 `.mexmaci64`, Linux x86-64 `.mexa64`, and Windows x86-64 `.mexw64` |
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
