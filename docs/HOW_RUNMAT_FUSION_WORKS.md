# Deep Dive: RunMat Fusion Planner & Execution

This document explains how RunMat detects fusible sub-graphs, generates GPU kernels, and executes them through the acceleration provider. Understanding fusion is essential when extending the builtin catalogue or optimising new kernels.

## 1. Goals & Scope

The fusion subsystem in `crates/runmat-accelerate/src/fusion.rs` turns stretches of MATLAB operations into a single GPU dispatch. It focuses on two fusion kinds today:

- **Elementwise chains** – sequences of elementwise primitives / builtins that share compatible shapes.
- **Reductions** – single reduction nodes where the surrounding context can reuse the GPU output.

By consolidating operations, RunMat reduces kernel launch overhead, minimises host ↔ device transfers, and unlocks more opportunities for `NativeAutoOffload` to keep data on the GPU.

## 2. Graph Representation (`AccelGraph`)

The planner operates on an `AccelGraph` constructed by upstream compilation passes (`crates/runmat-accelerate/src/graph.rs`). Each node (`AccelNode`) records:

- A unique `NodeId`.
- A label (`AccelNodeLabel`) that distinguishes primitive operations from builtin calls.
- A category (`AccelOpCategory`) such as `Elementwise` or `Reduction`.
- Lists of input and output `ValueId`s plus spans into the original instruction stream, enabling positional lookups.

`ValueInfo` entries capture data flow: origin (variable, constant, node output), runtime type, shape information (`ShapeInfo`), and optional constant payloads. The fusion detector uses this metadata to enforce broadcasting and shape compatibility.

## 3. Detecting Fusion Groups

`detect_fusion_groups` performs a forward scan over the graph:

1. Builds a consumer map (`build_consumer_map`) that records which nodes use each value.
2. For each elementwise node that has not been assigned to another group, attempts to extend a chain while:
   - Requiring single-consumer edges (`find_next_elementwise`) to avoid duplicating results.
   - Verifying shape compatibility via `ShapeInfo::unify`.
   - Rejecting unknown shapes or cycles.
3. Registers the chain as a `FusionGroup` (kind `ElementwiseChain`) when more than one node participates. The span conservatively covers the first and last instruction touched.
4. Adds remaining standalone reduction nodes as single-node groups (`FusionKind::Reduction`).

The output is a list of `FusionGroup` descriptors containing node IDs, aggregate shapes, spans, and a monotonically increasing group ID.

## 4. Building Fusion Plans

`FusionPlan::from_graph` transforms groups into `FusionGroupPlan` structures used at runtime. The constructor (`FusionGroupPlan::new`) performs several tasks:

- Collects external inputs, assigns them stable indices, and gathers literal constants into a map. For elementwise groups the `stack_pattern` preserves the order in which constants should be pushed onto the expression stack inside the generated shader.
- For reduction groups, recognises known reduction builtins (`sum`, `mean`) and keeps only the data operand in `inputs`, storing constants (e.g., `omitnan` flags) separately. This is important because the reduction shaders assume a data buffer plus uniform parameters rather than N arbitrary operands.
- Emits an operation sequence (`Vec<FusionOp>`) where each entry records whether the original node was a primitive or a named builtin.
- Computes a kernel specification (`FusionKernelSpec`) that currently flags whether the plan is supported. Plans call into `generate_wgsl` or `generate_reduction_wgsl` to probe whether the expression can be rendered for the active scalar type.

`FusionGroupPlan::element_count` and `constant_shape` provide metadata used by both auto-offload heuristics and argument preparation.

## 5. WGSL Generation

Elementwise plans call `generate_wgsl` to produce shader source tailored to the provider’s numeric precision:

- The planner seeds an expression map (`HashMap<ValueId, String>`) with input bindings named `input{n}`.
- Each `FusionOp` is translated into WGSL via `primitive_expr` (for arithmetic primitives) or `builtin_expr` (for named builtins such as `log1p`, `atan2`, `cosh`). The helpers implement a widening policy for constants (`cast_literal`) to support both `f32` and `f64` code generation.
- The function emits a complete shader module with bindings for each argument, the output storage buffer, and uniforms payload (`Params`) containing loop bounds.

Reduction plans call `generate_reduction_wgsl`. The generator emits tiling logic that loads slices into `var<workgroup>` memory, performs the reduction, applies builtin-specific post-processing (e.g., dividing by `reduce_len` for `mean`), and writes results back to the output buffer. Optional two-pass thresholds and workgroup sizes specified in the builtin GPU metadata influence how the provider launches the shader.

If either generator returns `None`, the kernel is marked unsupported and execution falls back to the CPU path.

## 6. Plan Activation & Instrumentation

Fusion plans are cached per graph pointer in `PLAN_CACHE` (a `Lazy<RwLock<HashMap<usize, Weak<FusionPlan>>>>`). At runtime:

1. `prepare_fusion_plan` upgrades an existing plan or constructs a new one.
2. `activate_fusion_plan` installs the plan in a thread-local (`ACTIVE_PLAN`) so interpreter threads can access fusion metadata without recomputing.
3. `set_current_pc` updates the active group based on the program counter. Debuggers and telemetry endpoints can call `active_fusion` to inspect the current group’s kind, span, estimated element count, and support flag. This hook feeds the auto-offload engine via `NativeAutoOffload::promote_binary`.

The thread-local also supports duplicating the active group plan (`active_group_plan_clone`) so downstream code can inspect constants or rebuild WGSL without mutating shared state.

## 7. Executing Fused Kernels

`crates/runmat-accelerate/src/fusion_exec.rs` translates `FusionGroupPlan` structures into provider calls.

- **Input preparation** – `execute_elementwise` and `execute_reduction` walk the plan’s inputs and allocate `PreparedInput` records. Host tensors are uploaded to temporary device buffers (`provider.upload`); scalars are materialised as dense tensors so providers see a uniform interface. GPU inputs pass through untouched and maintain residency.
- **Shader generation** – the functions query the provider for its precision (`ProviderPrecision`) to choose `f32` vs `f64` WGSL, then call the plan’s generator. A missing shader results in an error so the caller can fall back.
- **Dispatch** – for elementwise plans the provider’s `fused_elementwise` hook receives the WGSL source, device handles, resolved output shape, and total element count. Reductions call `fused_reduction` with explicit `reduce_len`, `num_slices`, and workgroup configuration (using provider defaults when none are supplied). The provider returns a new `GpuTensorHandle`; the execution helper marks it resident via `fusion_residency::mark`.
- **Cleanup** – any temporary uploads are freed (`provider.free`) before returning the GPU handle wrapped in `Value::GpuTensor`.

`infer_output_shape` cross-checks `FusionGroup.shape` to produce the column-major shape vector expected by WGSL helpers.

## 8. Residency Tracking

`crates/runmat-accelerate/src/fusion_residency.rs` maintains a process-wide `HashSet<u64>` of resident buffer IDs. Providers call `mark` after successful fused dispatches, and `runmat_accelerate_api::clear_residency` (registered during provider initialisation) removes handles when the runtime gathers a tensor back to the host. This bookkeeping helps the planner avoid redundant re-uploads and lets debugging tools highlight which tensors still live on the GPU.

## 9. Integration with Builtins

Fusion relies on declarative metadata supplied by the new builtin set in `crates/runmat-runtime/src/builtins`:

- Each builtin that supports fusion registers a `BuiltinFusionSpec` describing shape requirements, constant embedding strategy, and WGSL builders (`FusionKernelTemplate`). For example, `math/elementwise/exp.rs` supplies a unary elementwise template, while `math/reduction/sum.rs` exposes a reduction template with omit-NaN hints.
- When the interpreter lowers builtin calls into the acceleration graph, it tags nodes with `AccelGraphTag::Elementwise` or `AccelGraphTag::Reduction`, enabling the detector to recognise patterns.
- During execution the builtin consults `fusion::active_group_plan_clone()`. If the plan is supported, the builtin forwards to `fusion_exec`. Otherwise it falls back to direct provider hooks or host code.

This arrangement lets builtin authors focus on expressing the scalar semantics while RunMat handles code generation, broadcasting, and residency.

## 10. Provider Support & Pipeline Caching

The wgpu backend compiles the WGSL source emitted by fusion through `WgpuPipelines` (`crates/runmat-accelerate/src/backend/wgpu/pipelines.rs`). The provider hashes shader strings (using layouts such as `runmat-fusion-layout-{arity}`) to reuse compute pipelines. Warm-up dispatches during provider registration populate caches and surface hit/miss metrics (`WgpuProvider::fused_cache_counters`) for diagnostics exposed by `provider_cache_stats()`.

Because the fusion generator produces deterministic WGSL, pipelines remain hot across executions, keeping amortised dispatch latency low even for short expressions.

## 11. Interaction with Auto-Offload

`NativeAutoOffload` inspects `fusion::active_fusion()` when evaluating promotion candidates. If the current group is an elementwise chain with a known element count, the planner can confidently decide to keep operands on the GPU even if individual operands look small. This tight coupling ensures auto-offload decisions match what the fusion engine will ultimately execute.

## 12. Typical Execution Flow

1. The interpreter records builtin calls into an `AccelGraph`.
2. Before execution, `detect_fusion_groups` identifies fusible regions and builds a `FusionPlan`.
3. The interpreter activates the plan and updates the program counter as it steps through instructions.
4. When evaluating a builtin that participates in the active group, the builtin pulls the `FusionGroupPlan` and calls into `fusion_exec`.
5. `fusion_exec` uploads any host operands, generates WGSL, and submits it to the provider. The resulting GPU tensor remains resident for subsequent consumers.
6. If a later builtin requires CPU access (e.g., printing results), the dispatcher gathers the tensor, which in turn clears residency metadata.

By separating detection, planning, and execution, RunMat keeps fusion extensible. Adding support for new operations usually involves updating builtin fusion templates rather than reworking the planner itself.
