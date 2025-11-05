## Team A Handoff

### 0. Orientation & Artefacts

- **Scope recap:** Team A owns the GPU provider (`crates/runmat-accelerate`), WGSL kernels, residency/buffer lifecycle, and fused GEMM epilogues (see `PROMPT_TEAM_A.md`, `NEXT_PLAN.md`).
- **Inputs reviewed:** `TEAM_A_START.md` (initial marching orders), cumulative notes in `TEAM_A_PROGRESS.md`, system plan in `NEXT_PLAN.md`, and the merged code as of 2025‑11‑05.
- **Verification status:** `cargo test -p runmat-accelerate --features wgpu` and targeted GPU regressions (matmul epilogue suite) are green on the merged tree. No lints outstanding for Team A hotspots.
- **Partner workstreams:** Team B (fusion planner, runtime policy) and Team C (parser/runtime correctness) have merged complementary changes; Team A code now depends on their new fusion descriptors (notably diag extraction detection).

### 1. What We Delivered This Shift

#### 1.1 GEMM Epilogue Extensions (Elementwise + Diag Pack)

- Expanded `MatmulEpilogue` (`crates/runmat-accelerate-api/src/lib.rs`) to cover `clamp_min`, `clamp_max`, `pow_exponent`, and an optional `diag_output` handle for epilogue C.
- Updated the uniform structs (`crates/runmat-accelerate/src/backend/wgpu/params.rs`) with a flags bitmask + diagonal metadata and adjusted WGSL epilogue shaders (`shaders/matmul.rs`) to apply clamp/pow while optionally writing the diagonal.
- Ensured pipeline layout/bind groups allocate a dedicated storage slot for diag buffers (`pipelines.rs`, `provider_impl.rs`) and segregated dummy buffers to avoid WGPU usage conflicts.
- Mirrored the expanded semantics in the CPU fallback provider (`simple_provider.rs`) so parity tests remain representative.
- Added GPU tests (`tests/matmul_epilogue.rs`) that validate clamp/pow and diagonal extraction against CPU references, covering the new path end-to-end.

#### 1.2 SYRK & Small‑k GEMM (from earlier in shift, now merged)

- Implemented SYRK WGSL kernels, dispatch logic, and API integration; added CPU fallback and regression tests.
- Added a small‑k specialized matmul path with heuristics inside `provider_impl.rs`, plus WGSL and tests. Ensure heuristics still match plan expectations (Team B may tune thresholds later).

#### 1.3 Fusion Integration Upgrades

- Reconciled Team B/C changes by extending `fusion_exec::execute_matmul_epilogue` to harvest clamp, pow, and diag intent from the fusion plan, allocate diag buffers on demand, and populate the descriptor before invoking Team A kernels.
- Normalized planner constants (via `value_to_f64`) and ensured residency bookkeeping frees or marks temporary outputs correctly.

#### 1.4 Infrastructure Validation

- Re-ran `cargo test -p runmat-accelerate --features wgpu` post-merge to confirm Team B/C contributions plus our extensions coexist.
- Inspected shader, pipeline, and provider diffs to confirm Team A’s prior buffer-pooling and epilogue work remain intact after the merge.
- Updated `TEAM_A_PROGRESS.md` with the final reconciliation entry so future shifts have a continuous log.

### 2. Current State Snapshot

| Area | Status | Notes |
| --- | --- | --- |
| **Matmul epilogues** | ✅ Elementwise + diag pack wired. | Pipeline key includes mask bits; diag buffers optional. |
| **Fusion planner linkage** | ✅ clamp/min/max/pow/diag extracted. | Planner currently emits clamp/pow/diag for relevant PCA/variance chains. |
| **SYRK** | ✅ Kernel + tests present. | Team B planner still emits standalone SYRK; diag pack integration pending. |
| **Small‑k GEMM** | ✅ Kernel + heuristics. | Trigger threshold is static; may revisit once telemetry flows. |
| **Buffer pooling** | ✅ Size-class pooling with zero-on-alloc (from earlier work). | No conflicts detected post-merge. |
| **Vec4 IO & logical transpose** | ⏳ Not yet addressed this shift. | Remain on our TODO backlog (see §4). |
| **4k fused kernel** | ⏳ Not started. | Planner support partially in place; Team A kernel outstanding. |
| **Telemetry hooks** | ⚠️ Minimal. | `crates/runmat-accelerate/src/telemetry.rs` landed from other teams; Team A integration TBD. |

### 3. Code Hotspots to Know

1. **`crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs`** — central orchestrator for matmul/syrk, buffer pooling, pipeline dispatch. New logic around lines touching `matmul_epilogue` handles diag buffers and uses the bitmask flags.
2. **`crates/runmat-accelerate/src/backend/wgpu/shaders/matmul.rs`** — WGSL epilogue now reads `flags` to gate row/col scaling, clamp/pow, and diag writes; keep masks in sync with `params.rs` constants.
3. **`crates/runmat-accelerate/src/fusion_exec.rs`** — `execute_matmul_epilogue` is the bridge from Team B’s planner output to Team A kernels. Any future planner changes must be reflected here to avoid silent feature drops.
4. **`crates/runmat-accelerate/tests/matmul_epilogue.rs`** — canonical parity coverage for epilogue features; extend here when adding new epilogue operations (bias vectors, axis reductions, etc.).
5. **`TEAM_A_PROGRESS.md`** — reliable operational log; aligns with this handoff and should be maintained next shift.

### 4. Outstanding Work & Suggested Next Steps

#### 4.1 Immediately Actionable (High Priority)

1. **Fused Normalize→Gain→Bias→Clamp→Pow Kernel (4k pipeline)**
   - Implement the fused WGSL kernel (likely reusing Welford stats from Team B’s planner). Hook into provider with a new dispatch entry.
   - Coordinate with Team B to ensure `FusionOp` emission matches expected parameters (mean/var handles, gain/bias vectors, clamp bounds, exponent).
   - Add unit/regression tests for correctness vs. the CPU provider and measure launch-count savings.

2. **Vec4 IO & Logical Transpose Support**
   - Audit matmul and SYRK loads/stores to add `vec4` paths where alignment allows; confirm shader workgroup dimensions remain valid across devices.
   - Implement stride-remapping for logical transpose so we avoid host materialization for `'`/`.'` when shapes align.
   - Add microbenchmarks or telemetry counters to confirm bandwidth improvements.

3. **Telemetry Instrumentation**
   - Integrate the newly-added `telemetry.rs` with matmul/syrk dispatch to record pipeline cache hits, kernel durations, and buffer reuse statistics.
   - Expose counters to the bench harness so Team D can ingest them (goal articulated in `NEXT_PLAN.md`).

#### 4.2 Coordinate with Other Teams

- **Team B (Fusion Planner & Runtime)**
  - Ensure planner emits diag pack requests at the right granularity (currently we detect `diag` builtins; confirm this holds for PCA/variance graphs).
  - Align on shape heuristics for small‑k matmul vs general matmul — they may gather telemetry to tune thresholds; keep our heuristics pluggable.
  - Watch for upcoming fusion patterns (Monte Carlo, NLMS) to reserve descriptor bits and buffer layouts.

- **Team C (Runtime Correctness)**
  - Validate that dtype propagation (`NumericDType`) remains intact after diag pack writes; diag outputs should preserve provider precision.
  - Ensure zero-length edge cases (e.g., diag on empty matrices) are handled gracefully (currently we allocate a `[0,1]` buffer; confirm this doesn’t trigger downstream assertion failures).

- **Team D (Telemetry/Bench)**
  - Provide the schema for the extended epilogue flag mask so telemetry logs can decode which features were active per kernel.
  - Offer warmup recommendations for the new pipelines (matmul_epilogue variants) to minimize first-use latency.

#### 4.3 Medium-Term / Strategic

1. **Epilogue B (per-axis reduce + scale)**
   - Plan the design for mean/std normalization epilogues (row/col reductions feeding broadcast scalers). Will require additional buffers + WGSL support.
2. **Pipeline Cache Persistence & Warmup**
   - Introduce hashed keys that incorporate epilogue flags and precision; coordinate with Team D on on-disk persistence.
3. **Kernel Autotuning Hooks**
   - Add instrumentation to easily tune workgroup sizes/tiles per device (Metal vs Vulkan) as spelled out in `NEXT_PLAN.md` (Milestones M2–M4).

### 5. Risks, Assumptions, Watch Items

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Planner emits new epilogue operations (e.g., bias vectors, axis-specific transforms) without Team A support. | Runtime fallback to CPU or incorrect GPU execution. | Maintain a compatibility check in `execute_matmul_epilogue`; log/early-return to CPU provider if unsupported fields appear. Coordinate weekly with Team B on schema changes. |
| Diag buffer allocation overhead for large batched runs. | Extra allocations per epilogue could offset wins. | Consider reusing pooled buffers keyed by length; monitor telemetry once added. |
| Clamp/pow ordering semantics differ between CPU fallback and GPU. | Parity drift in edge cases. | Tests currently cover simple scenarios; add stress cases (negative values with pow, clamp bounds interplay) to detect mismatches early. |
| Vec4 adoption may misalign with non-multiple-of-4 dimensions. | Incorrect memory loads/stores. | Gate vec4 paths behind alignment checks and fall back to scalar loops; add tests for odd sizes. |
| Telemetry integration lag. | Hard to measure improvements / regressions. | Prioritize hooking in counters before expanding kernel set. |

### 6. Testing & Validation Checklist for Next Shift

1. Keep `cargo test -p runmat-accelerate --features wgpu` as the baseline regression gate before/after major edits.
2. For new kernels (e.g., 4k fused), add GPU + CPU parity tests under `crates/runmat-accelerate/tests/`.
3. When adjusting fusion logic, also run `cargo test -p runmat-accelerate` (without WGPU) to ensure planner-only tests (`fusion_patterns.rs`, `fusion::tests::*`).
4. If telemetry code touches bins/pipelines, consider running the bench harness smoke suite (`benchmarks/.harness/run_suite.py`) once instrumentation is in place.

### 7. Tips for Success & Contextual Knowledge

- **Pipeline key discipline:** The epilogue flags feed into `layout_tag` and pipeline hashing; whenever adding new epilogue features, extend the bitmask constants and update pipeline key strings to avoid cache collisions.
- **Buffer pooling contract:** `create_storage_buffer_checked` must wrap all hot-path allocations to honor device limits. If introducing new scratch buffers, route allocations through this helper or the zero-on-alloc variants.
- **WGSL code style:** Keep workgroup sizes and tile macros consistent (`@MT@` placeholders). When adding new kernels, include comments with tile/workgroup assumptions for future tuning.
- **Residency hooks:** After every provider allocation, mark/finalize residency as done in existing functions. This prevents VRAM leaks and ensures `fusion_residency::mark` remains accurate.
- **Coordination cadence:** Team A should sync with Team B after each planner milestone (see `NEXT_PLAN.md` milestones M1–M3) to consume new fusion opportunities promptly.

### 8. Reference Index

- **Design/Plan**: `PROMPT_TEAM_A.md`, `NEXT_PLAN.md`
- **Progress Log**: `TEAM_A_PROGRESS.md`
- **Key Codepaths**:
  - Provider: `crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs`
  - Kernels: `crates/runmat-accelerate/src/backend/wgpu/shaders/`
  - Fusion bridge: `crates/runmat-accelerate/src/fusion_exec.rs`
  - Tests: `crates/runmat-accelerate/tests/`
  - Telemetry (new): `crates/runmat-accelerate/src/telemetry.rs`
- **Bench harness:** `benchmarks/.harness/`

### 9. Closing Remarks

Team A’s current tree has a solid foundation: buffer pooling is robust, matmul epilogues cover the full “elementwise + diag pack” feature set required for PCA and explained-variance workloads, and SYRK/small‑k kernels are ready for planner adoption. The next shift should focus on fusing the 4k pipeline, expanding vectorized IO, and deepening telemetry so we can quantify future wins. Keep leveraging `TEAM_A_PROGRESS.md` for day-to-day logging; this handoff + that log should let you ramp quickly.

Good luck, and please reach out to Teams B/C/D early when interfaces shift—our success hinges on the fusion planner emitting the patterns our kernels expect.

