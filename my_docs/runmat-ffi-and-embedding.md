# RFC-0001: RunMat Native FFI and C ABI Embedding

- **Status**: Milestone 0-1 Implemented
- **Authors**: RunMat contributors
- **Target Version**: RunMat ≥ 0.x
- **Created**: 2025-xx-xx
- **Discussion**:
  - MEX equivalent & FFI: https://github.com/orgs/runmat-org/discussions/93
  - Embedding / C ABI: https://github.com/runmat-org/runmat/issues/87

---

## 1. Summary

This RFC proposes a unified native interoperability design for RunMat that:

1. Replaces MATLAB MEX with a **typed, ABI-stable Foreign Function Interface (FFI)** for calling native code from RunMat.
2. Introduces a **C ABI embedding interface** that allows RunMat to be used as a library from C/C++ and other languages.
3. Uses a **single shared type system** for:
   - runtime argument checking
   - native ABI layout
   - future ahead-of-time specialization

The goal is **not** to reimplement MATLAB’s `mxArray` API, but to provide a modern, explicit, and maintainable alternative that supports both performance and portability.

---

## 2. Motivation

### 2.1 Why not MEX?
MEX has several structural issues:
- Heavy reliance on `mxArray`, which is dynamically typed and opaque
- Poor separation between ABI, type system, and memory ownership
- Difficult static analysis and specialization
- Tight coupling to MATLAB internals

However, MEX is widely used, and a large ecosystem of numeric kernels exists.

### 2.2 Why a unified FFI + embedding story?
RunMat should support:
- Calling native code **from RunMat**
- Calling RunMat **from native code**

Other modern technical computing systems (e.g. Julia, Python) treat these as two sides of the same interoperability story. RunMat should do the same.

---

## 3. Design Goals

### 3.1 Primary goals
- Stable C ABI
- Explicit memory ownership rules
- Zero-copy numeric arrays when possible
- Optional but enforceable type annotations
- First-class support for dense numeric workloads

### 3.2 Non-goals (initially)
- Full MATLAB `mxArray` compatibility
- Complete coverage of MATLAB types (cell, struct, sparse, object system)
- Automatic translation of arbitrary MEX code

---

## 4. High-Level Architecture

The proposal consists of **three layers**, all sharing a common type model.

```
+----------------------------+
|        RunMat Core         |
+----------------------------+
        ↑            ↓
+---------------+  +----------------+
| runmat-ffi    |  | runmat-embed   |
| (RunMat → C)  |  | (C → RunMat)   |
+---------------+  +----------------+
        ↑
+----------------------------+
| Native libraries (C/C++)   |
+----------------------------+
```

---

## 5. Layer A: C ABI Embedding (`runmat-embed`)

### 5.1 Purpose
Allow RunMat to be embedded into:
- C/C++ applications
- Other language runtimes
- Simulation and modeling environments

### 5.2 Deliverables
- `librunmat` (static + dynamic)
- `runmat.h` public C header
- Minimal example applications

### 5.3 Core API (draft)

```c
typedef struct rm_context rm_context;
typedef struct rm_value   rm_value;
typedef struct rm_error   rm_error;

/* lifecycle */
rm_context* rm_context_new(void);
void        rm_context_free(rm_context*);

/* evaluation */
int rm_eval_utf8(
    rm_context* ctx,
    const char* code,
    rm_value*** out,
    size_t* nout,
    rm_error* err
);

int rm_run_file(
    rm_context* ctx,
    const char* path,
    rm_value*** out,
    size_t* nout,
    rm_error* err
);

/* calling functions */
int rm_call(
    rm_context* ctx,
    const char* function_name,
    const rm_value** args,
    size_t nargs,
    rm_value*** out,
    size_t* nout,
    rm_error* err
);
```

### 5.4 Error handling
- All APIs return status codes
- `rm_error` contains:
  - error code
  - UTF-8 message
  - optional stack trace

No exceptions or `longjmp` across the ABI boundary.

---

## 6. Layer B: RunMat → Native FFI (`runmat-ffi`)

### 6.1 Purpose
Replace MEX with:
- a typed, explicit ABI
- predictable performance
- clean separation of interface and implementation

### 6.2 RunMat surface API (illustrative)

```matlab
y = ffi.call("mylib", "fir", x, coeffs);
```

### 6.3 Native function ABI (example)

```c
int fir_f64(
    rm_array_f64 x,
    rm_array_f64 coeffs,
    rm_array_f64* y,
    rm_context* ctx
);
```

Key properties:
- No `mxArray`
- No hidden allocation
- Outputs explicitly allocated via context or returned handles

---

## 7. Layer C: Signature Language (`.ffi` / `.rmi`)

### 7.1 Motivation
MATLAB MEX signatures are implicit.
Codegen types live outside the function definition.

RunMat introduces a **first-class interface specification**.

### 7.2 Example

```text
module mylib

  fir(
    x: f64[:,1],
    coeffs: f64[1,:]
  ) -> f64[:,1]
    = fir_f64

  fir(
    x: f32[:,1],
    coeffs: f32[1,:]
  ) -> f32[:,1]
    = fir_f32

end
```

### 7.3 Type system (initial)

Base types:
- `f64`, `f32`
- `i32`, `i64`, `u8`
- `bool`
- `complex<f32>`, `complex<f64>`

Shapes:
- fixed: `T[m,n]`
- partially dynamic: `T[:,n]`, `T[m,:]`
- fully dynamic: `T[:,:]`

### 7.4 Semantics
- Types are optional but authoritative when present
- Used for:
  - runtime validation
  - overload dispatch
  - ABI layout
  - future specialization

---

## 8. Memory Ownership Model

- Inputs are borrowed unless explicitly copied
- Outputs are owned by RunMat
- Native code may request allocation via context
- No hidden reference counting across ABI

Rule of thumb:
> If you didn’t allocate it, you don’t free it.

---

## 9. Migration Strategy (MEX-lite)

### 9.1 Goals
- Enable gradual migration
- Support common numeric MEX patterns
- Avoid full `mex.h` reimplementation

### 9.2 Approach
Provide a small compatibility header:

```c
#include <rm_mex.h>
```

Features:
- numeric array helpers
- argument count checking
- error reporting
- mapping to `rm_value`

Explicitly unsupported (initially):
- deep MATLAB API hooks
- graphics, Java, engine calls

---

## 10. Tooling & Packaging

### 10.1 Native packages
Native code is packaged as a RunMat package:

```
mylib/
 ├─ mylib.ffi
 ├─ src/
 ├─ CMakeLists.txt
 └─ runmat.toml
```

### 10.2 CLI support
```bash
runmat package init --native mylib
runmat package build
runmat package test
```

---

## 11. Roadmap & Milestones

### Milestone 0
- Freeze C ABI v1
- Dense numeric arrays only

### Milestone 1
- `librunmat` embedding
- C examples and documentation

### Milestone 2
- `ffi.call` MVP
- One native library loaded dynamically

### Milestone 3
- `.ffi` parser and validation
- Overload dispatch

### Milestone 4
- MEX-lite compatibility layer

---

## 12. Alternatives Considered

### 12.1 Reimplement `mxArray`
Rejected:
- large surface area
- legacy design constraints
- discourages better APIs

### 12.2 Rust-only FFI
Rejected:
- C ABI is required for ecosystem reach
- Rust remains an implementation detail

---

## 13. Open Questions

- Sparse array ABI
- Cell/struct representation
- Threading model for native calls
- JIT vs AOT specialization triggers

---

## 14. Conclusion

This RFC defines a modern interoperability foundation for RunMat:
- principled
- minimal
- extensible

It enables RunMat to replace MEX-style workflows while also becoming a reusable numerical runtime embeddable in larger systems.

---
