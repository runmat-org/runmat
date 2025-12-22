# FFI & Embedding Handoff Document

**Date:** 2024-12-22  
**Previous Shift:** Completed Milestones 0–4 of RFC-0001  
**Status:** All planned milestones complete

---

## Summary of Completed Work

### Three Branches with Pending PRs

| Branch | PR | Description |
|--------|-----|-------------|
| `blas-opt-in` | #103 | Make BLAS/LAPACK opt-in for Windows compatibility |
| `ffi-embed` | #104 | C ABI embedding (`runmat-abi` + `runmat-embed`) |
| `ffi` | #105 | FFI native calls (`runmat-ffi`) |

### New Crates Created

**On `ffi-embed` branch:**
- `crates/runmat-abi/` — Shared C ABI types (`RmStatus`, `RmError`, `RmArrayF64`, `RmContext`, `RmValue`)
- `crates/runmat-embed/` — C embedding API with header at `include/runmat.h`
  - Builds to `runmat_embed.dll` (Windows) / `librunmat_embed.so` (Linux)
  - Example: `examples/hello.c`

**On `ffi` branch:**
- `crates/runmat-ffi/` — Builtins for calling native C libs from RunMat
  - `ffi_load("libname")` — Load a shared library
  - `ffi_call("libname", "funcname", args...)` — Call a function
  - `ffi_unload("libname")` — Unload a library
  - Test library: `tests/native/mymath.c` → `mymath.dll`

---

## What Works Now

### Embedding (runmat-embed)
```c
#include "runmat.h"

rm_context* ctx = rm_context_new();
rm_value** out; size_t nout;
rm_error err = {0};

rm_eval(ctx, "A = [1 2; 3 4]; A * A", &out, &nout, &err);
// Extract result with rm_value_to_array_f64()

rm_values_free(ctx, out, nout);
rm_context_free(ctx);
```

### FFI (runmat-ffi)
```matlab
% RunMat code
result = ffi_call("mymath", "add", 2.0, 3.0);  % Returns 5.0
y = ffi_call("mymath", "square", 4.0);          % Returns 16.0
pi_val = ffi_call("mymath", "get_pi");          % Returns 3.14159...
```

Supported signatures:
- Nullary: `double func()`
- Unary: `double func(double)`
- Binary: `double func(double, double)`
- Up to 5 args: `double func(double, double, double, double, double)`
- Unary array: scaffolded but needs testing

---

## Milestone 3: `.ffi` Signature Language — COMPLETED

**Implemented:**
- `FfiType` enum with pointer types: `F64`, `F32`, `I32`, `I64`, `U32`, `Usize`, `Ptr(Box<FfiType>)`, `PtrMut(Box<FfiType>)`, `Void`
- `FfiSignature` struct with `name`, `args`, `ret`
- `SignatureFile` parser for `.ffi` files
- `ffi_load("libname", "libname.ffi")` syntax
- `ffi_call` validates argument count against signature

**Signature File Format:**
```
# mymath.ffi
add: (f64, f64) -> f64
square: (f64) -> f64
get_pi: () -> f64
scale_array: (ptr<f64>, usize, usize, ptr_mut<f64>, ptr_mut<usize>, ptr_mut<usize>, f64) -> i32
```

**Files added/modified:**
- `crates/runmat-ffi/src/parser.rs` — NEW: Signature file parser
- `crates/runmat-ffi/src/types.rs` — Enhanced with pointer types, Display impl
- `crates/runmat-ffi/src/registry.rs` — Extended with `LibraryEntry`, signature storage
- `crates/runmat-ffi/src/builtins/ffi_load.rs` — Accepts optional `.ffi` path
- `crates/runmat-ffi/src/builtins/ffi_call.rs` — Uses signatures for validation/dispatch
- `crates/runmat-ffi/tests/native/mymath.ffi` — NEW: Test signature file
- `crates/runmat-ffi/tests/integration_test.rs` — Added signature file tests

---

---

## Milestone 4: MEX-lite Compatibility Layer — COMPLETED

**Implemented on `ffi-embed` branch:**
- `rm_mex.h` header with MATLAB MEX-like API
- `MxArray` struct wrapping RunMat Tensor directly
- Matrix creation: `mxCreateDoubleMatrix`, `mxCreateDoubleScalar`
- Matrix info: `mxGetM`, `mxGetN`, `mxGetNumberOfElements`, `mxIsEmpty`, `mxIsScalar`, `mxIsDouble`
- Data access: `mxGetPr`, `mxGetScalar`
- Memory: `mxDestroyArray`, `mxDuplicateArray`
- Helpers: `mexErrMsgTxt`, `mexWarnMsgTxt`, `mexPrintf`

**Files added/modified:**
- `crates/runmat-embed/include/rm_mex.h` — NEW: MEX compatibility header
- `crates/runmat-embed/src/mex.rs` — NEW: MEX function implementations
- `crates/runmat-embed/src/lib.rs` — Added mex module
- `crates/runmat-embed/examples/mex_double.c` — NEW: Example MEX function

**Example MEX function usage:**
```c
#include "rm_mex.h"

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
    size_t m = mxGetM(prhs[0]);
    size_t n = mxGetN(prhs[0]);
    double* input = mxGetPr(prhs[0]);

    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
    double* output = mxGetPr(plhs[0]);

    for (size_t i = 0; i < m * n; i++) {
        output[i] = input[i] * 2.0;
    }
}
```

---

## Development Environment

### Windows Build Requirements
```powershell
# Set MSYS2 in PATH for C compilation
$env:PATH = "C:\msys64\ucrt64\bin;C:\msys64\usr\bin;$env:PATH"

# Build release
cargo build --release

# The embed DLL and import lib are at:
#   target/release/runmat_embed.dll
#   target/release/runmat_embed.dll.lib
```

### Testing runmat-embed
```powershell
cd crates/runmat-embed/examples

# Compile hello.c
gcc -o hello.exe hello.c -I../include -L../../../target/release -lrunmat_embed

# Run (needs DLL in PATH)
$env:PATH = "$PWD\..\..\..\target\release;$env:PATH"
./hello.exe
```

### Testing runmat-ffi
```powershell
cd crates/runmat-ffi/tests/native

# Compile test library
gcc -shared -o mymath.dll mymath.c

# Run integration tests
cd ../..
cargo test --release
```

---

## Architecture Notes

### Builtin Registration
Builtins use the `#[runtime_builtin]` macro from `runmat-macros`:
```rust
#[runtime_builtin(
    name = "ffi_call",
    category = "ffi",
    summary = "Call a native function.",
    keywords = "ffi,native,c"
)]
pub fn ffi_call_builtin(lib: Value, func: Value, rest: Vec<Value>) -> Result<Value, String> {
    // ...
}
```

The macro uses `inventory` crate for registration. Functions are collected at link time.

### Linking runmat-ffi into Runtime
**Current state:** `runmat-ffi` builtins are NOT yet linked into `runmat-runtime`.

**To enable FFI builtins at runtime:**
1. Add `runmat-ffi` as dependency in `crates/runmat-runtime/Cargo.toml`
2. Add `pub use runmat_ffi;` in runtime's lib.rs to ensure builtins are linked

### Library Registry
FFI uses a global `Lazy<Mutex<LibraryRegistry>>` in `registry.rs`:
- `load_library(name)` — Loads with platform-specific naming
- `unload_library(name)` — Drops library handle
- Auto-searches: `{name}.dll`, `lib{name}.dll`, `lib{name}.so`, etc.

---

## Known Caveats

1. **Array functions:** Scaffolded in `ffi_call.rs` but incomplete. The `call_unary_array` function needs:
   - Output buffer allocation
   - Proper error handling for native return codes
   - Testing with real array operations

2. **cbindgen failed:** Header was manually written. If you modify `runmat-abi` types, update `runmat.h` manually.

3. **No signature validation:** Currently `ffi_call` infers signatures from arguments. Calling with wrong types will cause undefined behavior or crashes.

4. **Windows-only testing:** Tested on Windows with MSYS2. Linux/macOS builds should work but haven't been verified for FFI.

---

## File Locations Quick Reference

| Component | Location (on branch) |
|-----------|---------------------|
| C ABI types | `ffi-embed:crates/runmat-abi/src/lib.rs` |
| Embed API | `ffi-embed:crates/runmat-embed/src/lib.rs` |
| C header | `ffi-embed:crates/runmat-embed/include/runmat.h` |
| C example | `ffi-embed:crates/runmat-embed/examples/hello.c` |
| FFI builtins | `ffi:crates/runmat-ffi/src/builtins/` |
| Library registry | `ffi:crates/runmat-ffi/src/registry.rs` |
| Test native lib | `ffi:crates/runmat-ffi/tests/native/mymath.c` |
| Integration tests | `ffi:crates/runmat-ffi/tests/integration_test.rs` |

---

## Suggested Approach for Milestone 3

1. **Start on `ffi` branch** — it has all FFI code
2. **Add signature parser** — Use `nom` or hand-written parser for `.ffi` files
3. **Store signatures in registry** — Extend `LibraryRegistry` to hold `HashMap<String, FfiSignature>`
4. **Update dispatch** — In `ffi_call`, check registry for signature before inferring
5. **Test with mymath.ffi** — Create signature file for the test library

Good luck! 🚀
