# RunMat Accelerate WGPU Provider Refactor

## Goal

Refactor `runmat/crates/runmat-accelerate/src/backend/wgpu/provider_impl/mod.rs` into a clearer module tree without changing behavior, public APIs, fallback choices, buffer semantics, or kernel execution order.

This is a logic cleanup only refactor.

## Current State

The WGPU backend currently has this layout:

```text
backend/wgpu/
  provider.rs
  provider_impl/
    mod.rs
    constructors.rs
    elementwise.rs
    fft.rs
    indexing.rs
    polynomial.rs
    reduction.rs
    rnd.rs
    solve.rs
    tensor.rs
    window.rs
  pipelines.rs
  resources.rs
  dispatch/*
  shaders/*
```

`provider_impl/mod.rs` is the center of gravity for the backend. It currently mixes:

- provider construction and initialization
- device/adapter selection
- workgroup and autotune configuration
- pipeline/bootstrap/warmup logic
- GPU buffer lifecycle and handle registration
- upload/download/readback/free
- telemetry and cache reporting
- inline WGSL shader strings for logical/comparison kernels
- a very large `impl AccelProvider for WgpuProvider`
- many private execution helpers across unrelated operation families

The file is already partially dissolved into operation-family submodules (`elementwise`, `reduction`, `indexing`, `tensor`, `constructors`, `polynomial`) plus existing focused modules (`fft`, `rnd`, `solve`, `window`), and the top-of-file comment in `provider_impl/mod.rs` explicitly indicates that new implementation work should go into submodules.

At this point, `provider_impl/mod.rs` is still effectively monolithic.

As of 2026-05-24, extraction is materially underway and `provider_impl/mod.rs` has been reduced significantly, but it still contains mixed concerns (remaining operation families + provider lifecycle).

## Drift Notes (Repo-Verified)

These updates reflect current repository reality:

- `backend/wgpu/provider.rs` already exists and is the public registration/facade module.
- Renaming `provider_impl/mod.rs` directly to `provider/mod.rs` would collide with that existing module unless the facade is renamed in the same change.
- Multiple tests and internal imports currently reference `backend::wgpu::provider_impl::*`; a package rename is therefore not a pure local file move.
- `rnd.rs` already exists as a partial random-family extraction.

## Refactor Principles

The refactor should preserve all semantics.

Do:

- move code into coherent modules
- extract small local helpers where duplication is obvious
- keep the `WgpuProvider` type and core lifecycle easy to find
- preserve existing dispatch behavior and cleanup ordering
- preserve current host fallback behavior
- preserve current buffer metadata behavior
- preserve current telemetry and cache behavior

Do not:

- redesign the provider trait surface
- change which operations are GPU-backed vs host-backed
- change shader math, kernel parameters, or workgroup choices
- change caching semantics
- change submission ordering or readback behavior
- introduce broad generic abstractions that make control flow harder to follow

## Naming Recommendation

Use a single canonical `provider` module tree as the final architecture.

- `backend/wgpu/provider.rs` remains the parent module.
- implementation modules live under `backend/wgpu/provider/*`.
- no permanent split between `provider` (facade) and `provider_impl` (implementation package).

Transition strategy:

- During extraction, keep paths stable as needed.
- Finalization includes a coordinated rename from `provider_impl` into `provider/*` with import/test updates in one pass.

Why not rename directly to `ops/`?

Because a large portion of the current file is not operations code. It includes provider bootstrapping, buffer lifecycle, upload/download/free, warmup, metrics, and shared infrastructure. An `ops/` name would describe only part of the implementation.

Use `provider/ops/` as the stable long-term operations layer; the sequencing still starts with extraction, not immediate large-scale path churn.

## Target Layout

Recommended target layout:

```text
backend/wgpu/
  mod.rs
  provider.rs
  provider/
    mod.rs
    init.rs
    core.rs
    helpers.rs
    ops/
      elementwise.rs
      reduction.rs
      tensor.rs
      indexing.rs
      random.rs
      constructors.rs
      polynomial.rs
      image.rs
      linalg.rs
    fft.rs
    random_dist.rs
    solve.rs
    window.rs
```

This is the intended end state, not necessarily the first landing step.

Design boundaries in the final layout:

- `provider/{init,core,helpers}` own provider lifecycle and cross-cutting infrastructure.
- `provider/ops/*` own operation semantics and provider-level orchestration by family.
- `dispatch/*` stays kernel-launch plumbing and parameter binding mechanics.
- `shaders/*` stays shader source/constants only.

## What Should Live In Each Module

### `provider/mod.rs`

Primary contents:

- `WgpuProviderOptions`
- `WgpuProvider`
- `BufferEntry`
- `MatrixOperandView`
- `WorkgroupConfig`
- top-level module wiring and imports
- the `impl AccelProvider for WgpuProvider` entrypoints, unless that impl itself later becomes split by family using private helper methods in submodules

This file should become the top-level index for the provider implementation rather than the place where all logic lives.

### `provider/init.rs`

Provider construction and bootstrap logic:

- adapter/device selection
- environment variable parsing
- precision selection
- workgroup normalization/clamping
- pipeline cache path setup
- autotune bootstrap
- warmup bootstrap configuration
- device error handler installation

This code is cross-cutting and should stay close to provider construction.

### `provider/core.rs`

Provider lifecycle and buffer management:

- handle registration
- `get_entry`
- storage buffer creation helpers
- uniform buffer helpers that belong directly to provider behavior
- `submit`
- readback staging helpers
- upload/download
- `read_scalar`
- `free`
- metadata cleanup
- shared core validation helpers that directly depend on buffer tables

This is the highest-risk behavior and should remain easy to inspect.

### `provider/helpers.rs`

Small non-semantic helper utilities that reduce repetition:

- chunk iteration helpers
- empty-output fast path helpers
- pipeline warmup/noop helpers
- common shape/limit guards when they are reused across multiple families
- small dispatch helpers that do not own operation semantics

This file should stay small and practical.

### `provider/ops/elementwise.rs`

Operation family:

- unary ops
- scalar ops
- binary ops
- broadcast binary ops
- logical/comparison ops
- fused elementwise
- `map_nan_to_zero`
- `not_nan_mask`

This is the best first extraction candidate because the family is large, internally coherent, and has a lot of low-risk repetition.

### `provider/ops/reduction.rs`

Operation family:

- global reductions
- dimension reductions
- min/max with indices
- std/mean helpers
- nd mean
- nd moments
- fused reduction dispatch
- reduction tuning/autotune helpers that are only used here

This is a major cohesion win after `elementwise.rs`.

### `provider/ops/tensor.rs`

Operation family:

- transpose
- permute
- flip
- circshift
- repmat
- kron
- cat helpers
- diag
- tril/triu
- reshape-related helper paths that are operational rather than lifecycle-related

### `provider/ops/indexing.rs`

Operation family:

- `find`
- `sub2ind`
- `ind2sub`
- scatter row/column
- gather/index-select related helpers

### `provider/ops/random.rs`

Operation family:

- random uniform
- random normal
- random integer range
- random permutation
- RNG synchronization helpers
- stochastic evolution

### `provider/ops/constructors.rs`

Operation family:

- `fill`
- `eye`
- `linspace`
- `peaks`
- `peaks_xy`
- `fspecial`
- other pure allocation/creation style kernels

`window.rs` remains separate because it is already clean and self-contained.

### `provider/ops/polynomial.rs`

Operation family:

- `polyval`
- `polyder`
- `polyint`

This can stay small.

### `provider/ops/image.rs`

Operation family:

- `imfilter`
- provider-facing image normalize helpers
- operation-level orchestration for image kernels

Backend-tuning and low-level pipeline concerns stay in `dispatch/*` and are called from this ops layer.

### `provider/ops/linalg.rs`

Operation family:

- matmul helpers that are not already separated elsewhere
- pagefun mtimes
- centered gram
- `syrk`
- QR / QR power iteration
- covariance / corrcoef
- eig and matrix structure helpers
- `issymmetric`, `ishermitian`, `bandwidth`, `sym_rcm`

This should not be the first extraction because it is broad and more tightly coupled.

### Existing modules to keep

- `provider/fft.rs`
- `provider/solve.rs`
- `provider/window.rs`
- `provider/random_dist.rs` (distribution-specific helpers; fold/route cleanly into `ops/random.rs`)

These already match the desired direction.

## Exact Duplicate Patterns Worth Factoring Out

These should be treated as local cleanup opportunities during extraction, not as a separate abstraction project.

### 1. Chunked dispatch loops

Repeated shape:

```rust
let chunk_capacity = (MAX_DISPATCH_WORKGROUPS as usize) * WORKGROUP_SIZE as usize;
let mut offset = 0usize;
while offset < len {
    let chunk_len = (len - offset).min(chunk_capacity).max(1);
    // build params
    // create bind group
    // dispatch
    offset += chunk_len;
}
```

This appears across multiple constructors, transforms, random ops, and fused execution helpers.

Safe extraction:

- `for_each_chunk(len, |offset, chunk_len| -> Result<()>)`

### 2. Pipeline warmup / noop / flush sequencing

Several operations repeat a near-identical “touch pipeline and flush submission” sequence.

Safe extraction:

- `warm_pipeline_once(...)`
- `flush_queue_gap(...)`

### 3. Allocate-output-and-return-empty pattern

Repeated shape:

```rust
let out_buffer = self.create_storage_buffer_checked(len, label)?;
if len == 0 {
    return Ok(self.register_existing_buffer(out_buffer, shape.to_vec(), 0));
}
```

Safe extraction:

- `create_output_buffer_or_empty(...)`
- or a smaller helper returning an early result option

### 4. Logical/comparison wrappers

Nearly identical wrappers exist for:

- `eq`, `ne`, `lt`, `le`, `gt`, `ge`
- `logical_and`, `logical_or`, `logical_xor`, `logical_not`
- `isfinite`, `isnan`, `isinf`

Differences are mostly shader choice, arity, and logical result annotation.

Safe extraction:

- `run_unary_shader_op(...)`
- `run_binary_shader_op(...)`

### 5. Reverse scan via flip-forward-flip

This pattern is repeated for reverse `cumsum` / `cumprod` style logic.

Safe extraction:

- `reverse_scan_via_flip(...)`

### 6. Inline WGSL constants for logical/comparison kernels

The top of `provider_impl/mod.rs` contains a large block of embedded WGSL strings for logical and comparison kernels in both `f32` and `f64` forms.

These should move into `backend/wgpu/shaders/`.

Recommended destination:

- `backend/wgpu/shaders/logical.rs`
- or `backend/wgpu/shaders/comparison.rs` and `backend/wgpu/shaders/logical.rs`

This is a readability win and reduces the non-provider noise near the top of the implementation.

## First Extraction Recommendation

The best first extraction is `elementwise.rs`.

Why:

- large amount of code moved for relatively low risk
- high internal cohesion
- clear duplication cleanup opportunities
- minimal impact on provider core lifecycle logic
- natural place to move inline logical/comparison WGSL constants out of the monolith

Suggested contents for the first extraction:

- unary op execution helpers
- scalar op execution helpers
- binary and broadcast binary helpers
- fused elementwise helpers
- logical/comparison wrappers
- NaN mapping helpers
- any small local helper functions used only by this family

## Second Extraction Recommendation

After `elementwise.rs`, extract `reduction.rs`.

Why:

- another large cohesive family
- a major readability improvement
- reduces noise in the trait implementation

But this should come after elementwise because reduction code is more coupled to tuning and performance heuristics.

## Third Extraction Recommendation

Extract the tensor operations family (`provider/ops/tensor.rs`).

Why:

- operations are conceptually related
- many of them use similar chunking/output allocation patterns
- keeps reshaping and structural tensor transforms together

## What Should Not Move First

These areas are riskier and should stay put until the easier operation families are extracted:

- provider construction and bootstrap
- upload/download/free
- submission and readback synchronization
- buffer handle table behavior
- QR power iteration internals
- fused reduction tuning internals
- complex linalg fallback bridges

These parts are central to correctness and easy to destabilize during a purely structural refactor.

## Proposed Method-to-Module Mapping

This mapping is intentionally approximate. It is meant to guide extraction boundaries, not require a giant one-shot move.

### `provider/core.rs`

- handle registration helpers
- `get_entry`
- `create_storage_buffer*`
- `register_existing_buffer`
- `uniform_buffer`
- `submit`
- `map_readback_bytes_sync`
- upload/download/read scalar/free
- metadata cleanup helpers
- exported device/context helpers
- telemetry snapshot/reset (kept in core alongside cache ownership)

### `provider/ops/elementwise.rs`

- unary execution helpers
- scalar execution helpers
- binary execution helpers
- binary broadcast execution helpers
- logical/comparison execution helpers
- `fused_elementwise`
- `fused_elementwise_multi`
- `map_nan_to_zero`
- `not_nan_mask`
- inline WGSL logical/comparison shader references after they move to `shaders/`

### `provider/ops/reduction.rs`

- `reduce_global_exec`
- `reduce_dim_sum_mean_exec`
- `reduce_dim_minmax_exec`
- `reduce_std_exec`
- `reduce_std_dim_exec`
- `reduce_nd_mean_exec`
- `reduce_moments_nd_exec`
- `fused_reduction`
- reduction wrapper entrypoints in `impl AccelProvider`

### `provider/ops/tensor.rs`

- transpose / permute / reshape-related execution helpers
- flip / circshift
- repmat / kron
- diag helpers
- tril / triu
- cat-related helpers

### `provider/ops/indexing.rs`

- `find_exec`
- `scatter_column_exec`
- `scatter_row_exec`
- `sub2ind_exec`
- `ind2sub_exec`

### `provider/ops/random.rs`

- random uniform / normal / integer / permutation execution helpers
- RNG state setting
- stochastic evolution

### `provider/ops/constructors.rs`

- `fill`
- `eye`
- `linspace`
- `peaks`
- `peaks_xy`
- `fspecial`

### `provider/ops/polynomial.rs`

- `polyval`
- `polyder`
- `polyint`

### `provider/ops/image.rs`

- `imfilter`
- image normalize provider-facing helpers

### `provider/ops/linalg.rs`

- matmul helpers that remain provider-level orchestration
- pagefun mtimes
- centered gram
- `syrk`
- QR / QR power iter helpers
- covariance / corrcoef
- eig / structure checks / bandwidth / symrcm style helpers

## Proposed Refactor Sequence

All phases below are plan-of-record, not optional.

### Phase 0: Path stabilization without public path churn

1. Keep current public API/paths stable while extraction starts.
2. Normalize monolith file layout to module-friendly form (same module path; no behavior changes).
3. Keep code otherwise unchanged except local path normalization.

This creates the right structural landing zone before more extraction, without forcing import-path updates.

Status: completed on 2026-05-24.

### Phase 1: Extract `elementwise.rs`

1. Move elementwise and logical/comparison execution helpers.
2. Move inline WGSL logical/comparison shader strings into `shaders/`.
3. Extract only small local helper functions needed to reduce repetition.
4. Place extracted logic under `provider/ops/elementwise.rs` in the final tree.

Expected result:

- large monolith reduction
- minimal semantic risk
- immediate readability win

Status: completed on 2026-05-24.

### Phase 2: Extract `reduction.rs`

1. Move reduction execution helpers.
2. Keep provider-core synchronization behavior unchanged.
3. Keep tuning and thresholds exactly as they are.
4. Place extracted logic under `provider/ops/reduction.rs` in the final tree.

Expected result:

- another large monolith reduction
- clearer separation between elementwise and reduction families

Status: completed on 2026-05-24.

### Phase 3: Extract `tensor` and `indexing` operation families

1. Move structural tensor transforms.
2. Move indexing/scatter/find helpers.
3. Place extracted logic under `provider/ops/tensor.rs` and `provider/ops/indexing.rs`.

Expected result:

- provider root becomes much easier to scan

Status: completed on 2026-05-24.

### Phase 4: Extract `random.rs`, `constructors.rs`, `polynomial.rs`

1. Move creation and math utility families.
2. Fold existing random distribution logic (`rnd`) into final random-family structure.
3. Keep `window.rs` as-is.
4. Place extracted logic under `provider/ops/{random,constructors,polynomial}.rs`.

Status: completed on 2026-05-24.

### Phase 5: Extract remaining operation families from `provider_impl/mod.rs`

1. Extract signal/transform-style operations into a dedicated module (for example, `signal.rs`) so `conv1d`/`iir`/`diff`/`gradient` and related helpers are co-located.
2. Extract image-oriented provider orchestration (`imfilter` and provider-facing image normalization helpers) into `image.rs`.
3. Extract higher-coupling linear algebra and matrix-structure operations into `linalg.rs`.
4. Preserve exact fallback behavior, dispatch ordering, and kernel parameterization while moving.

This phase is still logic-preserving refactor only; no behavior changes.

### Phase 6: Split provider root into explicit lifecycle modules

Split provider root so non-operational provider lifecycle code is explicitly organized into:

- `init.rs`
- `core.rs`

This happens after the larger operation families are extracted.

### Phase 7: Coordinated finalization to canonical `provider` tree

After structural extraction is stable:

1. Move implementation package into `provider/*` under a single module root.
2. Preserve `provider.rs` as the public parent module and re-export surface.
3. Update all internal and test imports in one coordinated pass.

This is part of the plan of record for final naming clarity and long-term maintainability.

## Verification Strategy

Because the goal is structural cleanup only, every phase should be validated with no expected behavioral deltas.

At minimum after each phase:

- compile `runmat-accelerate`
- run relevant backend tests for WGPU
- run targeted tests covering the extracted family
- compare generated public behavior only through existing tests rather than new semantics

Important:

- no fallback changes
- no shader parameter changes
- no cleanup ordering changes
- no host/device result changes

## Bottom Line

The right cleanup is to keep behavior fixed while finishing extraction by operation family first, and then complete lifecycle split + naming cleanup to reach the final resting state.

Execution target during extraction:

- `provider.rs` remains the public facade
- `provider_impl/` remains the implementation package during extraction
- provider core stays near the root
- operation families are extracted one at a time

Final resting state (plan of record):

- single canonical `provider` module tree
- explicit lifecycle split (`init.rs`, `core.rs`, `helpers.rs`)
- explicit operation-family layer (`provider/ops/*`)
- strict domain boundaries:
  - ops semantics in `provider/ops/*`
  - launch plumbing in `dispatch/*`
  - shader sources in `shaders/*`
- normalized random naming (no long-term `rnd` abbreviation drift)

Next move:

1. finish Phase 5 extraction (`signal` + `image` + `linalg`) from `provider_impl/mod.rs`
2. split lifecycle code into explicit `init/core/helpers` ownership (Phase 6)
3. execute the coordinated naming finalization pass (Phase 7)

This keeps the work in clean domain boundaries while preserving current behavior end to end.
