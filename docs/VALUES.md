---
title: "Runtime Values & Type Model"
category: "Runtime Values"
section: "10.0"
last_updated: "July 30, 2026"
---

# Runtime Values & Type Model

`Value` is the concrete runtime representation used for values produced, stored, and passed around during RunMat execution. The VM stack, VM variables, builtin calls, workspace state, session results, GC roots, GPU residency paths, and WASM wire adapters all exchange this value type.

The compiler does not execute `Value` directly. HIR and MIR use static facts to approximate value type, shape, flow, and async state before bytecode runs. At execution time, those facts become concrete `Value` instances moving through the VM and runtime.

## Value Families

The `Value` enum groups loosely into scalars, dense arrays, aggregates, objects/handles, callables, the GPU handle, and a couple of internal execution helpers:

```rust
pub enum Value {
    // Scalars
    Int(IntValue), Num(f64), Complex(f64, f64), Bool(bool), String(String),
    // Dense arrays
    Tensor(Tensor), ComplexTensor(ComplexTensor),
    LogicalArray(LogicalArray), StringArray(StringArray), CharArray(CharArray),
    // Aggregates
    Cell(CellArray), Struct(StructValue),
    // Objects and handles
    Object(ObjectInstance), HandleObject(HandleRef), Listener(Listener),
    ClassRef(String), MException(MException),
    // Callables
    FunctionHandle(String), ExternalFunctionHandle(String),
    MethodFunctionHandle(String), BoundFunctionHandle { name: String, function: usize },
    Closure(Closure),
    // Acceleration
    GpuTensor(GpuTensorHandle),
    // Execution helper (internal multi-output/destructuring)
    OutputList(Vec<Value>),
}
```

| Family | Runtime variants | Notes |
| --- | --- | --- |
| Scalars | `Int`, `Num`, `Complex`, `Bool`, `String` | Scalar `Num` is a MATLAB double. Integer scalars preserve their integer class through `IntValue`. |
| Dense arrays | `Tensor`, `ComplexTensor`, `LogicalArray`, `StringArray`, `CharArray` | Dense array payloads own Rust buffers directly. Shapes follow MATLAB column-major semantics. |
| Aggregates | `Cell`, `Struct` | Cells own `Value` elements directly. Struct fields preserve insertion order through `IndexMap`. |
| Objects and handles | `Object`, `HandleObject`, `Listener`, `ClassRef`, `MException` | These carry class, identity, event, metaclass, or exception semantics for object-oriented and diagnostic paths. |
| Callables | `FunctionHandle`, `ExternalFunctionHandle`, `MethodFunctionHandle`, `BoundFunctionHandle`, `Closure` | Callable values preserve different resolution policies for builtins, semantic functions, methods, closures, and external-boundary calls. |
| Acceleration | `GpuTensor` | GPU-resident tensor handle owned by an acceleration provider. Host materialization happens only when an operation requires it. |
| Execution helpers | `OutputList` | Internal multi-output/destructuring helper used while shaping results. |

The enum lives in `runmat-builtins` because builtins, VM dispatch, runtime services, GC, session state, and WASM all need the same value vocabulary. `runmat-runtime` owns most operations over values, while `runmat-vm` owns instruction-level movement and mutation.

## Dense Arrays And Shape

RunMat stores dense numeric arrays as `Tensor` or `ComplexTensor`. During the authoritative-storage migration, the current `Tensor` owns both the ordinary floating representation and, for integer arrays, an authoritative exact backing store:

```rust
pub struct Tensor {
    pub data: Vec<f64>,                       // compatibility view
    pub integer_data: Option<IntegerStorage>, // authoritative for integer dtype
    pub shape: Vec<usize>,                    // MATLAB-visible N-D shape
    pub rows: usize,                          // cached 2-D dimensions
    pub cols: usize,
    pub dtype: NumericDType,
}
```

| Field | Meaning |
| --- | --- |
| `data` | Contiguous column-major floating data. For integer tensors this is a compatibility mirror only and can be inexact for `int64`/`uint64`. |
| `integer_data` | Exact homogeneous `i8`/`i16`/`i32`/`i64`/`u8`/`u16`/`u32`/`u64` storage. When present, it is authoritative for values and dtype. |
| `shape` | MATLAB-visible N-D shape. |
| `rows` / `cols` | Cached 2-D dimensions for common matrix paths and interop. |
| `dtype` | MATLAB-visible numeric class. Integer dtypes must agree with `integer_data`. |

`IntegerStorage` is not optional optimization metadata. Integer-aware code must read it before `data`; mutation must update exact storage and then repair the mirror. The mirror exists for legacy algorithms and intentional conversions to a floating computation domain. It must not be used for integer comparison, ordering, hashing, indexing, assignment into integer storage, class-preserving arithmetic, serialization, or exact host/device transfer.

`ComplexTensor` follows the same rule. Ordinary complex values use `Vec<(f64,f64)>`; typed complex integers additionally carry authoritative paired `IntegerStorage` values for their real and imaginary components.

`SparseTensor` stores CSC structure plus a floating `values` compatibility view. Typed sparse integers carry authoritative `IntegerStorage` for stored nonzeros. Exact consumers use `integer_storage`/`integer_at`, not the legacy floating `get` path.

Column-major shape semantics are preserved across tensor construction, indexing, builtin dispatch, workspace inspection, and host materialization. Code that reports memory footprint must account for both the compatibility view and native storage while both are retained.

### Authoritative-storage target

The compatibility mirror is transitional, not the target value model. Dense real numeric tensors are moving to one private homogeneous storage enum with native variants for `f64`, `f32`, and all eight integer classes. Dtype will be derived from that storage, `single` will use native `f32`, and no integer tensor will retain an eager or persistent `f64` mirror.

`NumericStorage`, `NumericScalar`, `NumericStorageView`, and `NumericStorageViewMut` define that exhaustive native storage contract in `runmat-builtins`. Exact scalar access, same-class mutation, zero/one allocation, shape-validated clone, gather, and reorder preserve the storage variant; `materialize_f64` and `materialize_f32` are the explicitly named potentially lossy boundaries. During Phase 1 these primitives coexist with the current public `Tensor` fields; the later field-privatization migration moves `Tensor` ownership onto this contract and uses compiler diagnostics to enumerate every consumer.

`Tensor::from_numeric_storage` is the Phase 2 construction boundary for all ten native classes, and `Tensor::into_numeric_storage` is the transitional consuming ownership bridge. Until field privatization completes, the constructor derives legacy public compatibility fields from its native input; new construction paths should enter through this boundary rather than assemble those fields independently.

Sparse values, complex values, provider transfer views, and GPU handle metadata will use the same authoritative element-type contract while retaining container/backend-appropriate physical layouts. In particular, “unified” does not require sparse CSC values and packed WGPU words to share an in-memory layout.

### Numeric boundary rule

Until the migration removes `Tensor::data`, a consumer that reads it from a value that may be integer must fall into one of these categories:

1. **Exact consumer:** branch on `integer_storage` and operate on `IntegerStorage`/`IntValue`.
2. **Intentional floating boundary:** the documented operation converts integer input to single/double output or a floating algorithm domain. The conversion is explicit and any loss above `flintmax` is part of that public conversion.
3. **Validated scalar parameter:** convert only after proving the exact integer is in the destination domain, such as `usize`, `u32`, or the exact-double interval.
4. **Unsupported integer input:** reject before reading the mirror.

A direct mirror read without one of these justifications is an integer threading defect. Poisoned-mirror tests should replace `data`/`values` with invalid sentinels while retaining exact storage and verify that exact consumers still produce the right result. The repository census script remains a discovery and regression aid; completion is established by private storage, exhaustive typed dispatch, compiler diagnostics, and semantic tests.

Logical arrays use `LogicalArray`. Logical scalars use `Bool`, while logical N-D arrays store normalized `0` or `1` bytes with an explicit shape.

Text has three representations:

| Runtime value | MATLAB concept |
| --- | --- |
| `String` | Scalar string value. |
| `StringArray` | N-D string array. |
| `CharArray` | 2-D character array for single-quoted text and char-matrix behavior. |

## Identity And GC

Most `Value` payloads are ordinary Rust-owned data. They are cloned, moved through the VM stack, stored in workspace maps, and dropped by normal Rust ownership.

Values that need stable identity, cycle reachability, finalizers, or bridge identity use opaque `GcHandle` tokens. The main cases are handle-object targets, listener targets/callbacks, selected object/struct payloads, provider-owned resources that need finalizers, and bridge values that must remain address-stable while runtime code holds references. Cell arrays own their elements as ordinary `Value`s; a cell element may contain a handle, but cells do not GC-allocate every element.

The GC owns the outer `Value` allocation. Nested buffers such as tensor data, strings, vectors, and maps remain owned by Rust values inside that allocation. The collector is non-moving, so surviving `GcHandle` identities stay stable. A `GcHandle` is not a Rust reference; value access goes through checked GC APIs and guarded `GcValueRef` / `GcValueMut` borrows.

For details on allocation, roots, barriers, and finalizers, see [Memory Management](/docs/runtime/gc).

## GPU Residency

`Value::GpuTensor` is a handle to provider-owned device data. The handle itself contains shape, device identity, and buffer identity. Provider/API registries carry precision, real/complex layout, logical status, and exact integer element type.

Runtime and builtin paths gather GPU tensors only when host materialization is required. Device-capable builtins and fusion paths can keep data resident and return another `Value::GpuTensor`. Host-only builtins gather explicitly before operating.

Exact integer transfers use the provider's `upload_integer` and `download_integer` methods. Providers that cannot preserve native integer storage must reject those methods; they must not substitute the floating upload/download path. The WGPU provider uses packed `u32` words, with two words per `int64`/`uint64` element.

The current handle metadata does not distinguish a tensor created by explicit `gpuArray` syntax from one created by automatic promotion. Host fallback is therefore an acceleration policy over the shared representation, not proof of MATLAB `gpuArray` unsupported-call parity.

The migration target embeds authoritative numeric element metadata in the durable handle/provider state and converges floating and integer host transfer views on one exhaustive numeric type contract. Backend storage remains specialized.

For details on residency and fusion planning, see [GPU Acceleration & Fusion Engine](/docs/runtime/gpu).

## Callables And Multi-Output Values

Function-like values preserve the policy needed to call them later:

| Value | Purpose |
| --- | --- |
| `FunctionHandle` | Name-shaped function handle that can resolve through normal callable lookup. |
| `ExternalFunctionHandle` | Handle whose resolution must stay at the external boundary. |
| `MethodFunctionHandle` | Handle that preserves typed method identity. |
| `BoundFunctionHandle` | Handle already bound to a semantic function ID by the compiler/session. |
| `Closure` | Callable plus captured runtime values. |

`OutputList` is different. It is an internal value used to carry multiple outputs through bytecode, builtin dispatch, and destructuring. Session outcome assembly turns public results into `RuntimeFlow` shapes such as single value, output list, comma list, dynamic list, or no value.

## Static Facts

Compile-time type information is deliberately separate from runtime `Value`.

| Layer | Representation | Purpose |
| --- | --- | --- |
| HIR/MIR facts | `TypeFact`, `ShapeFact`, `ValueFlowFact`, `AsyncValueFact` | Dataflow reasoning, diagnostics, spawn safety, and lowering decisions before execution. |
| Builtin metadata | `runmat_builtins::Type` and type resolvers | Describes builtin signatures and inferred outputs for tooling and validation. |
| Runtime execution | `runmat_builtins::Value` | Concrete values passed through the VM, runtime, session, GC, GPU, and host adapters. |

Static facts may say that a local is a numeric tensor with a known shape. The runtime value might then be a host `Tensor`, a `ComplexTensor`, or a `GpuTensor` depending on execution path and residency. Compiler facts should guide checks and optimization, but runtime code must still validate actual `Value` variants at boundaries.

For details on static facts, see [MIR & Static Analysis](/docs/runtime/compiler/static-analysis). For builtin authoring rules, see [Authoring Builtins](/docs/runtime/builtins/authoring).

## Host Metadata

Hosts usually do not need the full internal value graph for presentation layers, such as a variable inspector. Session and WASM APIs derive host-facing metadata from `Value`:

| Metadata | Source |
| --- | --- |
| MATLAB class name | `matlab_class_name(value)` maps variants to labels such as `double`, `logical`, `cell`, `struct`, `gpuArray`, or `function_handle`. |
| Shape | `value_shape(value)` reads scalar, array, cell, string, object, and GPU shapes when available. |
| Numeric dtype | `numeric_dtype_label(value)` reports scalar and tensor numeric classes. |
| Size estimate | `approximate_size_bytes(value)` estimates directly owned host payload bytes when meaningful. |
| Preview | `preview_numeric_values(value, limit)` extracts bounded numeric previews for workspace inspection. |

Workspace inspection uses those helpers to avoid materializing large values unnecessarily. GPU tensors are previewed through provider-aware gather paths so hosts can inspect slices without downloading an entire device buffer.

## Where Values Flow

```mermaid
flowchart TD
  HIR["HIR / MIR facts<br/>TypeFact, ShapeFact, ValueFlowFact"]
  Bytecode["VM bytecode"]
  VM["VM stack and variables<br/>Vec<Value>"]
  Builtins["runtime builtins<br/>Value inputs and outputs"]
  Workspace["session workspace<br/>workspace_values"]
  GC["GC-managed identity<br/>GcHandle"]
  GPU["Accelerate provider<br/>Value::GpuTensor"]
  Host["host ABI<br/>ExecutionOutcome / WASM wire"]

  HIR --> Bytecode --> VM
  VM <--> Builtins
  VM <--> Workspace
  VM <--> GC
  Builtins <--> GPU
  Workspace --> Host
  Builtins --> Host
```
