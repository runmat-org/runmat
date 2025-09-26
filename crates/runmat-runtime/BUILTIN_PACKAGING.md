# Builtin Packaging & Authoring Blueprint

This document captures the end-state we want for builtin authoring, GPU integration, documentation, and automation. It is the go-to reference when wiring new builtins or extending the RunMat Function Manager tooling.

## Goals
- One Rust source file per builtin containing code, long-form documentation, GPU/fusion specs, and unit tests.
- Inventory-backed metadata that fuels both the runtime (Ignition/Turbine + Accelerate) and authoring tools.
- First-class support for scalar and variadic signatures, GPU offload, fusion planning, and BLAS/LAPACK fallbacks.
- Tooling that can emit structured docs for the Next.js site and drive Codex-based authoring sessions.

## Source Layout
```
crates/runmat-runtime/
  src/
    builtins/
      mod.rs              # category re-exports, shared helpers
      common/             # shared utilities (complex math, GPU helpers, test support)
      math/
        sin.rs
        sum.rs
        ...
      array/
        zeros.rs
        ...
      accel/
        gpu_array.rs
        ...
      ...                 # other categories (io, introspection, strings, etc.)
```
- `builtins/mod.rs` exposes category modules and re-exports existing function symbols to keep downstream code compiling.
- Shared helpers live under `builtins/common/`. They must never perform registration; builtin files call into them explicitly.
- Each builtin file is self-contained: documentation constant, specs, one or more `#[runtime_builtin]` annotated functions, helper routines, and tests.

## Builtin Template Checklist
1. `//!` file doc comment summarising the builtin.
2. `use` statements scoped to required helpers.
3. `pub const DOC_MD: &str = r#"..."#;` containing YAML frontmatter + Markdown (details below).
4. Optional `pub const GPU_SPEC: BuiltinGpuSpec` and `pub const FUSION_SPEC: BuiltinFusionSpec`, registered via helper macros.
5. One or more `#[runtime_builtin(..., doc_md = DOC_MD, ...)]` functions. Variadic signatures use a trailing `Vec<Value>` parameter, e.g. `rest: Vec<Value>`. The runtime macro already detects this pattern and passes the remaining arguments through.
6. Helper functions (private) to keep the annotated functions concise. Host/GPU split helpers are common.
7. `#[cfg(test)] mod tests` covering: scalars, array/broadcast, variadic combinations, GPU provider execution (under `feature = "native-accel"`), and doc example smoke tests via the shared test harness.

## Inline Documentation Expectations
- YAML frontmatter should cover `title`, `category`, `keywords`, `summary`, `references`, `gpu_support`, `fusion`, `tested`, and any flags relevant to BLAS/LAPACK usage.
- Markdown body should explain numerics, broadcasting, error behaviour, GPU semantics (including how Accelerate fuses kernels and manages residency), and provide MATLAB-style examples.
- Encourage users to understand GPU offload: describe gpuArray creation, gather, and the lazy execution model (Ignition + Accelerate detect fusion opportunities, queue kernels, and execute on demand).

## GPU & Fusion Spec Types
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType { F32, F64, I32, Bool }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuOpKind { Elementwise, Reduction, MatMul, Transpose, Custom(&'static str) }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastSemantics { Matlab, ScalarOnly, None }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderHook {
    Unary { name: &'static str },
    Binary { name: &'static str, commutative: bool },
    Reduction { name: &'static str },
    Custom(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstantStrategy { InlineLiteral, UniformBuffer, WorkgroupMemory }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyPolicy { InheritInputs, NewHandle, GatherImmediately }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeRequirements { BroadcastCompatible, Exact(&'static [usize]), Any }

pub struct FusionExprContext<'a> {
    pub scalar_ty: ScalarType,
    pub inputs: &'a [&'a str],
    pub constants: &'a [&'a str],
}

pub type FusionExprBuilder = fn(&FusionExprContext) -> Result<String, FusionError>;

pub struct FusionKernelTemplate {
    pub scalar_precisions: &'static [ScalarType],
    pub wgsl_body: FusionExprBuilder,
}

#[derive(Debug, thiserror::Error)]
pub enum FusionError {
    #[error("missing input {0}")] MissingInput(usize),
    #[error("unsupported precision {0:?}")] UnsupportedPrecision(ScalarType),
    #[error("{0}")] Message(&'static str),
}

pub struct BuiltinGpuSpec {
    pub name: &'static str,
    pub op_kind: GpuOpKind,
    pub supported_precisions: &'static [ScalarType],
    pub broadcast: BroadcastSemantics,
    pub provider_hooks: &'static [ProviderHook],
    pub constant_strategy: ConstantStrategy,
    pub residency: ResidencyPolicy,
    pub notes: &'static str,
}

pub struct BuiltinFusionSpec {
    pub name: &'static str,
    pub shape: ShapeRequirements,
    pub constant_strategy: ConstantStrategy,
    pub elementwise: Option<FusionKernelTemplate>,
    pub reduction: Option<FusionKernelTemplate>,
    pub emits_nan: bool,
    pub notes: &'static str,
}

register_builtin_gpu_spec!(GPU_SPEC);
register_builtin_fusion_spec!(FUSION_SPEC);
```
- Authoritative implementations of these types live in `crates/runmat-accelerate/src/spec.rs` (owned by Accelerate) and the shared helper re-export in `crates/runmat-runtime/src/builtins/common/spec.rs`. The Function Manager links against the same module to stay in sync.
- `register_builtin_*` macros submit the specs into inventory so Accelerate and the Function Manager can discover them.
- Provider hooks map to methods exposed by `runmat-accelerate-api`. For reductions, add `ProviderHook::Reduction { name: "reduce_sum" }`.
- `notes` must stay concise (one or two sentences) and focus on actionable implementation details: provider prerequisites, fallbacks, precision caveats, or residency expectations. Avoid repeating long-form documentation that already exists in `DOC_MD`.

## BLAS / LAPACK Integration Points
- BLAS/LAPACK-backed builtins live under `src/blas.rs` and `src/lapack.rs` guarded by `#[cfg(feature = "blas-lapack")]`.
- When authoring builtins that rely on these crates, mention the feature flag in `DOC_MD` (`requires_feature: blas-lapack`) and in the GPU spec notes if relevant.
- The Function Manager should check cargo features and warn when attempting to run tests that require BLAS/LAPACK but the feature is disabled.

## RunMat Function Manager Snapshot
- Binary crate at `tools/runmatfunc/`.
- Responsibilities: discover builtin manifests, assemble authoring contexts, launch interactive/headless Codex sessions, run targeted tests, emit documentation bundles, and manage job queues.
- CLI surface (initial):
  - `runmatfunc builtin <name> [--headless] [--model ...]`
  - `runmatfunc browse`
  - `runmatfunc docs emit`
  - `runmatfunc queue add/run`
  - `runmatfunc list`
- Documentation export writes `docs/generated/builtins.json` and `docs/generated/builtins.d.ts` for the Next.js site.

## Comprehensive Example (`math/sum.rs`)
The following pseudo-final file shows how an existing builtin (`sum`) migrates to the new structure. It demonstrates variadic arguments, GPU reduction support, fusion metadata, rich documentation, and tests. Use it as a template.

```rust
//! Elementwise and reduction sum builtin for RunMat.

use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{gpu_helpers, tensor_ops};
use crate::builtins::common::test_support;
use crate::builtins::{BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy,
    FusionError, FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook,
    ResidencyPolicy, ScalarType, ShapeRequirements};

pub const DOC_MD: &str = r#"---
  title: "sum"
  category: "math/reduction"
  keywords: ["sum", "reduction", "gpu", "accumulate"]
  summary: "Sum elements of scalars, vectors, matrices, or N-D tensors."
  references: []
  gpu_support:
    elementwise: false
    reduction: true
    precisions: ["f32", "f64"]
    broadcasting: "matlab"
    notes: "Falls back to host if provider lacks reduce_sum support."
  fusion:
    elementwise: false
    reduction: true
    max_inputs: 1
    constants: "inline"
  requires_feature: null
  tested:
    unit: "builtins::math::sum::tests"
    integration: "fusion_gpu::fused_elementwise_residency_and_gather"
---

# MATLAB / RunMat Runtime `sum` Function
`y = sum(x)` adds together the elements of `x`. Scalar inputs return a scalar; matrices and
higher-dimensional tensors reduce along the first non-singleton dimension by default. You can
specify a dimension explicitly with `sum(x, dim)`.

## Behavior
- `sum(x)` on a vector returns the sum of all elements.
- `sum(X)` on a matrix returns a row vector containing column sums (dimension 1 reduction).
- `sum(X, 2)` returns a column vector of row sums.
- Logical inputs are cast to double precision (`true` → `1.0`, `false` → `0.0`).
- Empty arrays return zero with matching shape semantics (MATLAB-compatible).
- Errors for non-numeric, non-logical inputs (structs, cells, strings).

## RunMat GPU Support
RunMat observes the operations you write and builds a computation graph on the fly. Accelerate
spots GPU-friendly patterns (for example elementwise work feeding a reduction) and generates
compact WGSL kernels so less data moves between CPU and GPU. Fusion inlines constants, honors
MATLAB broadcasting rules, and delays execution until the result is needed (displayed, stored in
CPU memory, or explicitly gathered via `gather`). The net effect: you keep writing MATLAB-style code while
RunMat handles kernel generation and scheduling automatically.

## Examples
\`\`\`matlab
A = [1 2 3; 4 5 6];
colSums = sum(A);      % [5 7 9]
rowSums = sum(A, 2);   % [6; 15]
\`\`\`

\`\`\`matlab
G = gpuArray(rand(1024, 1024));
column_energy = sum(G .^ 2);
result = gather(column_energy);
\`\`\`

\`\`\`matlab
values = [1, 2, 3, 4];
partial = sum(values(1:2));      % demonstrates subranges
\`\`\`

## Common use cases for `sum`

### Sum each column of a matrix (column-wise sum)
\`\`\`matlab
A = [1 2 3; 4 5 6];
colSums = sum(A);      % [5 7 9]
\`\`\`

### Sum each row of a matrix (row-wise sum)
\`\`\`matlab
A = [1 2 3; 4 5 6];
rowSums = sum(A, 2);   % [6; 15]
\`\`\`

### Count true values in a logical array (number of matches)
\`\`\`matlab
flags = [true false true true];
numTrue = sum(flags);  % 3
\`\`\`

### Sum of squares (energy) of a vector or matrix
\`\`\`matlab
x = rand(1, 1000);
energy = sum(x .^ 2);
\`\`\`

### Sum along a specific dimension of an N-D array
\`\`\`matlab
X = rand(4, 5, 6);
s = sum(X, 3);         % sums along the 3rd dimension → size 4x5x1
\`\`\`

### Sum on the GPU (`gpuArray`)
\`\`\`matlab
G = gpuArray(rand(2048, 2048));
result = gather(sum(G));   % column-wise sum on GPU, then bring to CPU
\`\`\`

## RunMat vs MATLAB behavior
- RunMat semantics match MATLAB's `sum` for numeric and logical inputs, including edge cases with empty
  arrays and the dimension argument.
- RunMat transparently offloads eligible computation to the GPU via Accelerate and Fusion;
  MATLAB requires the Parallel Computing Toolbox and typically executes separate kernels per step.
- Users benefit from fused kernels (faster execution), automatic residency tracking, and no manual
  gather calls unless they explicitly need CPU data.
- Error wording follows MATLAB conventions; any deviations are documented here so they are easy to
  discover and report.

## Source & Feedback
- Implementation: [crates/runmat-runtime/src/builtins/math/sum.rs](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/sum.rs) (contains code, specs, and the associated tests).
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.

## See Also
[`prod`](/docs/builtins/math/prod), [`mean`](/docs/builtins/math/mean), [`cumsum`](/docs/builtins/math/cumsum), [`gpuArray`](/docs/builtins/accel/gpu_array), [`gather`](/docs/builtins/accel/gather)
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sum",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Reduction { name: "reduce_sum" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    notes: "Provider should return a fresh GPU handle sized according to the reduction dimension.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sum",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.get(0).ok_or(FusionError::MissingInput(0))?;
            Ok(format!("accumulator += {input};"))
        },
    }),
    emits_nan: false,
    notes: "Fusion planner emits standard WGSL reduction loop with inline literals for constants.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[runtime_builtin(
    name = "sum",
    category = "math/reduction",
    summary = "Sum elements of scalars, vectors, matrices, or N-D tensors.",
    keywords = "sum,reduction,gpu",
    accel = "reduction",
    doc_md = DOC_MD
)]
fn sum_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return sum_all(value);
    }
    if rest.len() == 1 {
        return sum_with_dim(value, &rest[0]);
    }
    Err("sum: unsupported arguments".to_string())
}

fn sum_all(value: Value) -> Result<Value, String> {
    if let Value::GpuTensor(handle) = value {
        return gpu_reduce_default(handle);
    }
    match value {
        Value::Tensor(t) => tensor_ops::sum_default(&t),
        Value::LogicalArray(la) => {
            let tensor = tensor_ops::logical_to_tensor(&la)?;
            tensor_ops::sum_default(&tensor)
        }
        Value::Num(n) => Ok(Value::Num(n)),
        Value::Int(i) => Ok(Value::Num(i.to_f64())),
        other => Err(format!("sum: expected numeric tensor, got {other:?}")),
    }
}

fn sum_with_dim(value: Value, dim: &Value) -> Result<Value, String> {
    let dim_f = match dim {
        Value::Num(d) => *d,
        Value::Int(i) => i.to_f64(),
        _ => return Err("sum: dim must be numeric".to_string()),
    };
    if let Value::GpuTensor(handle) = value {
        return gpu_reduce_dim(handle, dim_f);
    }
    let tensor = tensor_ops::to_tensor(value)?;
    tensor_ops::sum_dim(&tensor, dim_f)
}

fn gpu_reduce_default(handle: runmat_accelerate_api::GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(out) = provider.reduce_sum_all(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let gathered = gpu_helpers::gather_tensor(&handle)?;
    tensor_ops::sum_default(&gathered)
}

fn gpu_reduce_dim(handle: runmat_accelerate_api::GpuTensorHandle, dim: f64) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let dim_usize = if dim < 1.0 { 1usize } else { dim as usize };
        match provider.reduce_sum_dim(&handle, dim_usize) {
            Ok(out) => return Ok(Value::GpuTensor(out)),
            Err(_) => {}
        }
    }
    let gathered = gpu_helpers::gather_tensor(&handle)?;
    let tensor = Value::Tensor(gathered);
    sum_with_dim(tensor, &Value::Num(dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_scalar() {
        let result = sum_builtin(Value::Num(5.0), Vec::new()).unwrap();
        match result {
            Value::Num(v) => assert!((v - 5.0).abs() < 1e-12),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[test]
    fn sum_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), Vec::new()).unwrap();
        let out = match result {
            Value::Tensor(t) => t,
            Value::Num(v) => {
                panic!("expected tensor result, got scalar {v}");
            }
            other => panic!("expected tensor result, got {other:?}"),
        };
        assert_eq!(out.shape, vec![1, 3]);
        assert_eq!(out.data, vec![5.0, 7.0, 9.0]);
}

    #[test]
    fn sum_matrix_dim_two() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), vec![Value::Num(2.0)]).unwrap();
        let out = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.data, vec![6.0, 15.0]);
    }

    #[test]
    fn sum_gpu_fallback() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
            let handle = provider.upload_tensor(&tensor).unwrap();
            let result = sum_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
            let gathered = test_support::gather(result).unwrap();
            assert_eq!(gathered.data, vec![10.0]);
        });
    }
}
```