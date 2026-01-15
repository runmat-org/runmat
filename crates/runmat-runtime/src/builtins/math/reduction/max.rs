//! MATLAB-compatible `max` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, ReduceDimResult};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "max",
        builtin_path = "crate::builtins::math::reduction::max"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "max"
category: "math/reduction"
keywords: ["max", "maximum", "reduction", "comparisonmethod", "omitnan", "gpu"]
summary: "Return the maximum elements of scalars, vectors, matrices, or N-D tensors with MATLAB-compatible options."
references: []
gpu_support:
  elementwise: false
  reduction: true
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Uses provider reduce_max_dim / reduce_max when available. Fallback gathers data to the host for omitnan, custom comparison modes, or complex inputs."
fusion:
  elementwise: false
  reduction: true
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::reduction::max::tests"
  integration: "builtins::math::reduction::max::tests::max_gpu_dim1_matches_cpu"
---

# What does the `max` function do in MATLAB / RunMat?
`max` returns the largest values in its input while preserving MATLAB semantics for reductions, elementwise comparisons, NaN handling, complex magnitude comparisons, and linear indexing.

## How does the `max` function behave in MATLAB / RunMat?
- `max(X)` on an `m × n` array reduces along the first non-singleton dimension, returning a row vector of column-wise maxima and the corresponding indices (when requested).
- `max(X, [], dim)` reduces along the specified dimension; `max(X, [], vecdim)` reduces along each dimension listed in `vecdim`.
- `max(X, [], 'all')` collapses every element into a scalar and returns the linear index when two outputs are requested.
- `max(X, [], 'linear')` is equivalent to `'all'` but guarantees that the matching index is linear over `X(:)`.
- `max(X, [], ..., 'omitnan')` ignores `NaN` values inside each slice. If every element in a slice is `NaN`, the result for that slice is `NaN` and the index is `NaN`.
- `max(X, [], ..., 'includenan')` (default) propagates `NaN` whenever a slice contains any `NaN` element, returning the index of the first `NaN`.
- `max(A, B)` performs elementwise comparison using MATLAB's implicit expansion rules. The second output indicates whether the maximum came from `A` (index `1`) or `B` (index `2`).
- Complex inputs follow MATLAB ordering: `'ComparisonMethod','auto'` (default) compares magnitudes and breaks ties using phase angles, while `'real'` compares real components first. `'abs'` is an explicit alias for magnitude ordering on real and complex inputs.

## `max` Function GPU Execution Behaviour
When RunMat Accelerate is active, tensors that already reside on the GPU stay on the device whenever the provider exposes `reduce_max_dim` (for dimension reductions) or `reduce_max` (for whole-array reductions). Requests that require `omitnan`, custom comparison modes, `'linear'` indices, or complex arithmetic gather the data to the host, compute the MATLAB-compatible result, and return the output on the host. Elementwise `max(A, B)` currently executes on the host; the planner rematerializes tensors on the GPU when follow-on fused kernels make it profitable.

## Examples of using the `max` function in MATLAB / RunMat

### Finding column-wise maxima of a matrix
```matlab
A = [3 1 5; 4 2 6];
[m, idx] = max(A);
```
Expected output:
```matlab
m   = [4 2 6];
idx = [2 2 2];
```

### Reducing along the second dimension
```matlab
A = [3 1 5; 4 2 6];
[m, idx] = max(A, [], 2);
```
Expected output:
```matlab
m   = [5; 6];
idx = [3; 3];
```

### Collapsing all elements with linear indices
```matlab
A = reshape(1:12, [3 4]);
[m, idx] = max(A, [], 'all');
```
Expected output:
```matlab
m   = 12;
idx = 12;  % linear index into A(:)
```

### Ignoring NaN values during reduction
```matlab
values = [NaN 4 2; 3 NaN 1];
[m, idx] = max(values, [], 1, 'omitnan');
```
Expected output:
```matlab
m   = [3 4 2];
idx = [2 1 1];
```

### Elementwise maximum with broadcasting
```matlab
A = [1 4 7];
B = [2; 3; 5];
[C, origin] = max(A, B);
```
Expected output:
```matlab
C =
     2     4     7
     3     4     7
     5     5     7

origin =
     2     1     1
     2     1     1
     2     2     1
```

### Comparing complex values by magnitude
```matlab
Z = [1+2i, 2+1i, -2+2i];
M = max(Z);                         % magnitude ordering
R = max(Z, [], 'ComparisonMethod', 'real');
```
Expected output:
```matlab
M = -2.0000 + 2.0000i
R = 2.0000 + 1.0000i
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You typically do **not** need to call `gpuArray` manually. The fusion planner keeps tensors on the GPU between compatible kernels. When a reduction is supported by the active provider, the maximum values and indices stay on device. If a provider lacks the necessary hook, RunMat gathers data to the host, computes the result, and returns host tensors—subsequent fused GPU kernels can re-upload data when profitable.

## FAQ

### Can I request the linear index of the global maximum?
Yes. Use either `max(X, [], 'all')` or `max(X, [], 'linear')`. Both return a scalar maximum and the linear index into `X(:)` when you request two outputs.

### Does `max` support `'ComparisonMethod'` for real and complex arrays?
Absolutely. `'auto'` or `'abs'` compare magnitudes; `'real'` compares the real component first. The returned values always match MATLAB, including tie-breaking rules.

### What happens when all elements are `NaN` and `'omitnan'` is requested?
The value result is `NaN` and the index is `NaN`, matching MATLAB's behavior for empty slices.

### Can I mix elementwise comparisons with dimension reductions?
No. `max(A, B)` performs elementwise comparisons only. Use `max(A, [], dim)` when you want reductions along specific dimensions.

### Do GPU reductions support `'omitnan'` or custom comparison methods?
Not yet. Those requests fall back to the host implementation, which still honors MATLAB semantics. The output remains a host tensor in that case.

### Are logical and integer inputs supported?
Yes. Logical arrays are promoted to double precision, and integer inputs are converted to double before comparison, matching MATLAB's numeric tower.

## See Also
[min](./min), [sum](./sum), [mean](./mean), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `max` function is available at: [`crates/runmat-runtime/src/builtins/math/reduction/max.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/max.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;
#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::max")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "max",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_max_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_max",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: false,
    notes:
        "Providers should implement reduce_max_dim / reduce_max. Requests that require omitnan, comparisonmethod overrides, or complex inputs fall back to the host implementation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::max")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "max",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("accumulator = max(accumulator, {input});"))
        },
    }),
    emits_nan: true,
    notes: "Fusion planner emits canonical reduction kernels; providers may substitute custom WGSL via reduce_max_dim hooks.",
};

/// Evaluation artifact returned by `max` that carries both values and indices.
#[derive(Debug, Clone)]
pub struct MaxEvaluation {
    values: Value,
    indices: Value,
}

impl MaxEvaluation {
    /// Consume the evaluation and return only the maximum values (single-output call).
    pub fn into_value(self) -> Value {
        self.values
    }

    /// Consume the evaluation and return both maxima and indices.
    pub fn into_pair(self) -> (Value, Value) {
        (self.values, self.indices)
    }

    /// Peek at the indices without consuming.
    pub fn indices_value(&self) -> Value {
        self.indices.clone()
    }
}

#[runtime_builtin(
    name = "max",
    category = "math/reduction",
    summary = "Return the maximum elements of scalars, vectors, matrices, or N-D tensors.",
    keywords = "max,maximum,reduction,gpu,comparisonmethod,omitnan",
    accel = "reduction",
    builtin_path = "crate::builtins::math::reduction::max"
)]
fn max_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    evaluate(value, &rest).map(|eval| eval.into_value()).map_err(Into::into)
}

/// Evaluate the builtin once and expose both outputs (value + indices).
pub fn evaluate(value: Value, rest: &[Value]) -> Result<MaxEvaluation, String> {
    let parsed = parse_call(rest)?;
    if std::env::var("RUNMAT_DEBUG_MAX").is_ok() {
        let call_label = match &parsed {
            ParsedCall::Reduction(_) => "reduction",
            ParsedCall::Elementwise(_) => "elementwise",
        };
        let first_arg = rest.first().map(debug_value_kind).unwrap_or("None");
        tracing::debug!(
            call_type = call_label,
            rest_len = rest.len(),
            first_arg = first_arg,
            "[runmat-debug-max]"
        );
    }
    match parsed {
        ParsedCall::Elementwise(args) => elementwise_max(value, args),
        ParsedCall::Reduction(args) => reduction_max(value, args),
    }
}

#[derive(Debug, Clone)]
enum ParsedCall {
    Reduction(ReductionArgs),
    Elementwise(ElementwiseArgs),
}

#[derive(Debug, Clone)]
struct ReductionArgs {
    selection: DimSelection,
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
    linear_index: bool,
}

impl Default for ReductionArgs {
    fn default() -> Self {
        Self {
            selection: DimSelection::Auto,
            nan_mode: ReductionNaN::Include,
            comparison: ComparisonMethod::Auto,
            linear_index: false,
        }
    }
}

#[derive(Debug, Clone)]
enum DimSelection {
    Auto,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComparisonMethod {
    Auto,
    Real,
    Abs,
}

#[derive(Debug, Clone)]
struct ElementwiseArgs {
    other: Value,
    comparison: ComparisonMethod,
}

fn parse_call(rest: &[Value]) -> Result<ParsedCall, String> {
    if rest.is_empty() {
        return Ok(ParsedCall::Reduction(ReductionArgs::default()));
    }

    let first = &rest[0];
    if !is_empty_placeholder(first) {
        let comparison = parse_elementwise_options(&rest[1..])?;
        return Ok(ParsedCall::Elementwise(ElementwiseArgs {
            other: first.clone(),
            comparison,
        }));
    }

    let mut args = ReductionArgs::default();
    parse_reduction_options(&mut args, &rest[1..])?;
    Ok(ParsedCall::Reduction(args))
}

fn debug_value_kind(value: &Value) -> &'static str {
    match value {
        Value::Num(_) => "Num",
        Value::Int(_) => "Int",
        Value::Bool(_) => "Bool",
        Value::Tensor(t) => {
            if t.data.is_empty() {
                "Tensor(empty)"
            } else {
                "Tensor"
            }
        }
        Value::GpuTensor(_) => "GpuTensor",
        Value::String(_) => "String",
        Value::CharArray(_) => "CharArray",
        Value::StringArray(sa) => {
            if sa.data.is_empty() {
                "StringArray(empty)"
            } else {
                "StringArray"
            }
        }
        Value::LogicalArray(l) => {
            if l.data.is_empty() {
                "LogicalArray(empty)"
            } else {
                "LogicalArray"
            }
        }
        Value::Cell(c) => {
            if c.data.is_empty() {
                "Cell(empty)"
            } else {
                "Cell"
            }
        }
        _ => "Other",
    }
}

fn is_empty_placeholder(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => t.data.is_empty(),
        Value::LogicalArray(l) => l.data.is_empty(),
        Value::StringArray(sa) => sa.data.is_empty(),
        Value::CharArray(ca) => ca.data.is_empty(),
        Value::Cell(cell) => cell.data.is_empty(),
        Value::String(s) => s.is_empty(),
        _ => false,
    }
}

fn parse_reduction_options(args: &mut ReductionArgs, rest: &[Value]) -> Result<(), String> {
    let mut idx = 0usize;
    let mut selection_set = !matches!(args.selection, DimSelection::Auto);
    let mut comparison_set = matches!(args.comparison, ComparisonMethod::Auto);
    while idx < rest.len() {
        if let Some(keyword) = keyword_of(&rest[idx]) {
            match keyword.as_str() {
                "omitnan" => {
                    args.nan_mode = ReductionNaN::Omit;
                    idx += 1;
                    continue;
                }
                "includenan" => {
                    args.nan_mode = ReductionNaN::Include;
                    idx += 1;
                    continue;
                }
                "all" => {
                    if selection_set {
                        return Err(
                            "max: 'all' cannot be combined with an explicit dimension".to_string()
                        );
                    }
                    args.selection = DimSelection::All;
                    selection_set = true;
                    idx += 1;
                    continue;
                }
                "linear" => {
                    if selection_set {
                        return Err(
                            "max: 'linear' cannot be combined with an explicit dimension"
                                .to_string(),
                        );
                    }
                    args.selection = DimSelection::All;
                    args.linear_index = true;
                    selection_set = true;
                    idx += 1;
                    continue;
                }
                "comparisonmethod" => {
                    let Some(value) = rest.get(idx + 1) else {
                        return Err("max: expected a value after 'ComparisonMethod'".to_string());
                    };
                    args.comparison = parse_comparison_method(value)?;
                    comparison_set = true;
                    idx += 2;
                    continue;
                }
                _ => {}
            }
        }

        if !selection_set {
            if let Some(selection) = parse_dimension_value(&rest[idx])? {
                args.selection = selection;
                selection_set = true;
                idx += 1;
                continue;
            }
        }

        return Err(format!("max: unrecognised argument {:?}", rest[idx]));
    }

    if !comparison_set {
        args.comparison = ComparisonMethod::Auto;
    }

    Ok(())
}

fn parse_elementwise_options(rest: &[Value]) -> Result<ComparisonMethod, String> {
    let mut comparison = ComparisonMethod::Auto;
    let mut comparison_set = false;
    let mut idx = 0usize;
    while idx < rest.len() {
        if let Some(keyword) = keyword_of(&rest[idx]) {
            match keyword.as_str() {
                "comparisonmethod" => {
                    let Some(value) = rest.get(idx + 1) else {
                        return Err("max: expected a value after 'ComparisonMethod'".to_string());
                    };
                    comparison = parse_comparison_method(value)?;
                    comparison_set = true;
                    idx += 2;
                    continue;
                }
                "omitnan" | "includenan" | "all" | "linear" => {
                    return Err(format!(
                        "max: '{}' is only supported for reduction calls",
                        keyword
                    ));
                }
                _ => {}
            }
        }
        return Err(format!("max: unrecognised argument {:?}", rest[idx]));
    }
    if !comparison_set {
        comparison = ComparisonMethod::Auto;
    }
    Ok(comparison)
}

fn parse_comparison_method(value: &Value) -> Result<ComparisonMethod, String> {
    let Some(keyword) = keyword_of(value) else {
        return Err("max: 'ComparisonMethod' expects a string value".to_string());
    };
    match keyword.as_str() {
        "auto" => Ok(ComparisonMethod::Auto),
        "abs" | "magnitude" => Ok(ComparisonMethod::Abs),
        "real" => Ok(ComparisonMethod::Real),
        other => Err(format!("max: unsupported ComparisonMethod '{other}'")),
    }
}

fn parse_dimension_value(value: &Value) -> Result<Option<DimSelection>, String> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 1 {
                return Err("max: dimension must be >= 1".to_string());
            }
            Ok(Some(DimSelection::Dim(raw as usize)))
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err("max: dimension must be finite".to_string());
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err("max: dimension must be integral".to_string());
            }
            if rounded < 1.0 {
                return Err("max: dimension must be >= 1".to_string());
            }
            Ok(Some(DimSelection::Dim(rounded as usize)))
        }
        Value::Tensor(t) => parse_dimension_tensor(t),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(logical)?;
            parse_dimension_tensor(&tensor)
        }
        Value::GpuTensor(_) => Err(
            "max: dimension arguments must reside on the host (they cannot be gpuArray values)"
                .to_string(),
        ),
        _ => Ok(None),
    }
}

fn parse_dimension_tensor(tensor: &Tensor) -> Result<Option<DimSelection>, String> {
    if tensor.data.is_empty() {
        return Ok(Some(DimSelection::Auto));
    }
    if tensor.rows() != 1 && tensor.cols() != 1 && tensor.shape.len() != 1 {
        return Err("max: dimension vector must be a row or column vector".to_string());
    }
    let mut dims = Vec::with_capacity(tensor.data.len());
    for &value in &tensor.data {
        if !value.is_finite() {
            return Err("max: dimension entries must be finite".to_string());
        }
        let rounded = value.round();
        if (rounded - value).abs() > f64::EPSILON {
            return Err("max: dimension entries must be integers".to_string());
        }
        if rounded < 1.0 {
            return Err("max: dimension indices must be >= 1".to_string());
        }
        dims.push(rounded as usize);
    }
    if dims.is_empty() {
        Ok(Some(DimSelection::Auto))
    } else {
        // MATLAB treats duplicate entries gracefully; remove duplicates while preserving order.
        let mut seen = BTreeSet::new();
        let mut uniq = Vec::with_capacity(dims.len());
        for dim in dims {
            if seen.insert(dim) {
                uniq.push(dim);
            }
        }
        Ok(Some(DimSelection::Vec(uniq)))
    }
}

fn reduction_max(value: Value, args: ReductionArgs) -> Result<MaxEvaluation, String> {
    match value {
        Value::GpuTensor(handle) => {
            if let Some(eval) = reduction_max_gpu(handle.clone(), &args)? {
                return Ok(eval);
            }
            // Fall back to host if GPU path is unavailable.
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            reduction_max_host(Value::Tensor(tensor), &args)
        }
        other => reduction_max_host(other, &args),
    }
}

fn reduction_max_gpu(
    handle: GpuTensorHandle,
    args: &ReductionArgs,
) -> Result<Option<MaxEvaluation>, String> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if args.nan_mode == ReductionNaN::Omit {
        return Ok(None);
    }
    if args.comparison != ComparisonMethod::Auto {
        return Ok(None);
    }
    if args.linear_index {
        return Ok(None);
    }
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };
    let target_dim = match args.selection {
        DimSelection::Auto => default_dimension_from_shape(&handle.shape),
        DimSelection::Dim(dim) => dim,
        DimSelection::Vec(ref dims) if dims.len() == 1 => dims[0],
        DimSelection::All => {
            if handle.shape.len() <= 1 {
                1
            } else {
                return Ok(None);
            }
        }
        _ => return Ok(None),
    };
    if target_dim == 0 {
        return Ok(None);
    }
    // MATLAB dimensions are 1-based; `reduce_max_dim` expects zero-based.
    let zero_based = target_dim.saturating_sub(1);
    if zero_based >= handle.shape.len() {
        return Ok(None);
    }
    match provider.reduce_max_dim(&handle, zero_based) {
        Ok(ReduceDimResult { values, indices }) => Ok(Some(MaxEvaluation {
            values: Value::GpuTensor(values),
            indices: Value::GpuTensor(indices),
        })),
        Err(_) => Ok(None),
    }
}

fn reduction_max_host(value: Value, args: &ReductionArgs) -> Result<MaxEvaluation, String> {
    match materialize_for_max("max", value)? {
        InputData::Real(tensor) => reduce_real_tensor(tensor, args),
        InputData::Complex(tensor) => reduce_complex_tensor(tensor, args),
    }
}

enum InputData {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn materialize_for_max(name: &str, value: Value) -> Result<InputData, String> {
    match value {
        Value::Tensor(t) => Ok(InputData::Real(t)),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            Ok(InputData::Real(tensor))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("{name}: {e}"))?;
            Ok(InputData::Real(tensor))
        }
        Value::Int(i) => {
            let tensor =
                Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| format!("{name}: {e}"))?;
            Ok(InputData::Real(tensor))
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| format!("{name}: {e}"))?;
            Ok(InputData::Real(tensor))
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| format!("{name}: {e}"))?;
            Ok(InputData::Complex(tensor))
        }
        Value::ComplexTensor(ct) => Ok(InputData::Complex(ct)),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => Err(
            format!("{name}: expected numeric or logical input, received non-numeric value"),
        ),
        Value::GpuTensor(_) => Err(format!(
            "{name}: internal error – GPU tensors must be gathered before host execution"
        )),
        Value::Object(_) | Value::HandleObject(_) | Value::Struct(_) | Value::Listener(_) => {
            Err(format!("{name}: unsupported input type"))
        }
        Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(format!("{name}: unsupported input type")),
    }
}

fn reduce_real_tensor(tensor: Tensor, args: &ReductionArgs) -> Result<MaxEvaluation, String> {
    let shape = tensor.shape.clone();
    if tensor.data.is_empty() {
        let output_shape = resolve_output_shape(&shape, &args.selection, &[])?;
        let values =
            Tensor::new(Vec::new(), output_shape.clone()).map_err(|e| format!("max: {e}"))?;
        let indices = Tensor::new(Vec::new(), output_shape).map_err(|e| format!("max: {e}"))?;
        return Ok(MaxEvaluation {
            values: tensor::tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }
    let resolved = resolve_reduction_dims(&shape, &args.selection)?;
    let output_shape = resolved.output_shape.clone();
    let output_len = tensor::element_count(&output_shape);

    if output_len == 0 {
        let values =
            Tensor::new(Vec::new(), output_shape.clone()).map_err(|e| format!("max: {e}"))?;
        let indices = Tensor::new(Vec::new(), output_shape).map_err(|e| format!("max: {e}"))?;
        return Ok(MaxEvaluation {
            values: tensor::tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }

    let strides = compute_strides(&shape);
    let output_strides = compute_strides(&output_shape);
    let dims_mask = resolved.dims_mask.clone();
    let reduce_strides = resolved.reduce_strides.clone();

    let mut best = vec![BestReal::new(); output_len];
    let mut coords = vec![0usize; shape.len()];
    for &value in &tensor.data {
        let out_idx = map_output_index(&coords, &output_strides, &dims_mask);
        let reduce_idx = map_reduce_index(
            &coords,
            &resolved.reduced_dims,
            &reduce_strides,
            resolved.reduce_all,
        );
        let full_idx = map_linear_index(&coords, &strides);

        update_best_real(
            &mut best[out_idx],
            value,
            reduce_idx,
            full_idx,
            args.nan_mode,
            args.comparison,
        );
        increment_coords(&mut coords, &shape);
    }

    let mut values = vec![0.0f64; output_len];
    let mut indices = vec![0.0f64; output_len];

    for (i, entry) in best.iter().enumerate() {
        if entry.nan_fixed {
            values[i] = f64::NAN;
            indices[i] = if args.linear_index || resolved.reduce_all {
                (entry.full_index + 1) as f64
            } else if resolved.reduced_dims.is_empty() {
                1.0
            } else {
                (entry.reduce_index + 1) as f64
            };
            continue;
        }
        if !entry.has_value {
            values[i] = f64::NAN;
            indices[i] = f64::NAN;
            continue;
        }
        values[i] = entry.value;
        indices[i] = if args.linear_index || resolved.reduce_all {
            (entry.full_index + 1) as f64
        } else if resolved.reduced_dims.is_empty() {
            1.0
        } else {
            (entry.reduce_index + 1) as f64
        };
    }

    let value_tensor =
        Tensor::new(values, output_shape.clone()).map_err(|e| format!("max: {e}"))?;
    let index_tensor = Tensor::new(indices, output_shape).map_err(|e| format!("max: {e}"))?;

    Ok(MaxEvaluation {
        values: tensor::tensor_into_value(value_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn reduce_complex_tensor(
    tensor: ComplexTensor,
    args: &ReductionArgs,
) -> Result<MaxEvaluation, String> {
    let shape = tensor.shape.clone();
    if tensor.data.is_empty() {
        let output_shape = resolve_output_shape(&shape, &args.selection, &[])?;
        let values = ComplexTensor::new(Vec::new(), output_shape.clone())
            .map_err(|e| format!("max: {e}"))?;
        let indices = Tensor::new(Vec::new(), output_shape).map_err(|e| format!("max: {e}"))?;
        return Ok(MaxEvaluation {
            values: complex_tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }

    let resolved = resolve_reduction_dims(&shape, &args.selection)?;
    let output_shape = resolved.output_shape.clone();
    let output_len = tensor::element_count(&output_shape);

    if output_len == 0 {
        let values = ComplexTensor::new(Vec::new(), output_shape.clone())
            .map_err(|e| format!("max: {e}"))?;
        let indices = Tensor::new(Vec::new(), output_shape).map_err(|e| format!("max: {e}"))?;
        return Ok(MaxEvaluation {
            values: complex_tensor_into_value(values),
            indices: tensor::tensor_into_value(indices),
        });
    }

    let strides = compute_strides(&shape);
    let output_strides = compute_strides(&output_shape);
    let dims_mask = resolved.dims_mask.clone();
    let reduce_strides = resolved.reduce_strides.clone();

    let mut best = vec![BestComplex::new(); output_len];
    let mut coords = vec![0usize; shape.len()];

    for &(re, im) in &tensor.data {
        let out_idx = map_output_index(&coords, &output_strides, &dims_mask);
        let reduce_idx = map_reduce_index(
            &coords,
            &resolved.reduced_dims,
            &reduce_strides,
            resolved.reduce_all,
        );
        let full_idx = map_linear_index(&coords, &strides);
        update_best_complex(
            &mut best[out_idx],
            (re, im),
            reduce_idx,
            full_idx,
            args.nan_mode,
            args.comparison,
        );
        increment_coords(&mut coords, &shape);
    }

    let mut values = vec![(0.0f64, 0.0f64); output_len];
    let mut indices = vec![0.0f64; output_len];

    for (i, entry) in best.iter().enumerate() {
        if entry.nan_fixed {
            values[i] = (f64::NAN, f64::NAN);
            indices[i] = if args.linear_index || resolved.reduce_all {
                (entry.full_index + 1) as f64
            } else if resolved.reduced_dims.is_empty() {
                1.0
            } else {
                (entry.reduce_index + 1) as f64
            };
            continue;
        }
        if !entry.has_value {
            values[i] = (f64::NAN, f64::NAN);
            indices[i] = f64::NAN;
            continue;
        }
        values[i] = entry.value;
        indices[i] = if args.linear_index || resolved.reduce_all {
            (entry.full_index + 1) as f64
        } else if resolved.reduced_dims.is_empty() {
            1.0
        } else {
            (entry.reduce_index + 1) as f64
        };
    }

    let value_tensor =
        ComplexTensor::new(values, output_shape.clone()).map_err(|e| format!("max: {e}"))?;
    let index_tensor = Tensor::new(indices, output_shape).map_err(|e| format!("max: {e}"))?;
    Ok(MaxEvaluation {
        values: complex_tensor_into_value(value_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

#[derive(Debug, Clone)]
struct BestReal {
    value: f64,
    reduce_index: usize,
    full_index: usize,
    has_value: bool,
    nan_fixed: bool,
}

impl BestReal {
    fn new() -> Self {
        Self {
            value: 0.0,
            reduce_index: 0,
            full_index: 0,
            has_value: false,
            nan_fixed: false,
        }
    }
}

#[derive(Debug, Clone)]
struct BestComplex {
    value: (f64, f64),
    reduce_index: usize,
    full_index: usize,
    has_value: bool,
    nan_fixed: bool,
}

impl BestComplex {
    fn new() -> Self {
        Self {
            value: (0.0, 0.0),
            reduce_index: 0,
            full_index: 0,
            has_value: false,
            nan_fixed: false,
        }
    }
}

fn resolve_output_shape(
    shape: &[usize],
    selection: &DimSelection,
    reduced_dims: &[usize],
) -> Result<Vec<usize>, String> {
    if shape.is_empty() {
        return Ok(Vec::new());
    }
    let mut output = shape.to_vec();
    match selection {
        DimSelection::All => {
            output.fill(1);
        }
        _ => {
            for &dim in reduced_dims {
                if dim < output.len() {
                    output[dim] = 1;
                }
            }
        }
    }
    Ok(output)
}

struct ResolvedDims {
    output_shape: Vec<usize>,
    reduced_dims: Vec<usize>,
    reduce_all: bool,
    dims_mask: Vec<bool>,
    reduce_strides: Vec<usize>,
}

fn resolve_reduction_dims(
    shape: &[usize],
    selection: &DimSelection,
) -> Result<ResolvedDims, String> {
    if shape.is_empty() {
        return Ok(ResolvedDims {
            output_shape: Vec::new(),
            reduced_dims: Vec::new(),
            reduce_all: true,
            dims_mask: Vec::new(),
            reduce_strides: Vec::new(),
        });
    }

    let mut reduced_dims = match selection {
        DimSelection::Auto => {
            let mut dim = None;
            for (index, &len) in shape.iter().enumerate() {
                if len > 1 {
                    dim = Some(index);
                    break;
                }
            }
            vec![dim.unwrap_or(0)]
        }
        DimSelection::Dim(dim) => {
            if *dim == 0 {
                return Err("max: dimension must be >= 1".to_string());
            }
            let index = dim.saturating_sub(1);
            if index >= shape.len() {
                Vec::new()
            } else {
                vec![index]
            }
        }
        DimSelection::Vec(dims) => {
            if dims.is_empty() {
                Vec::new()
            } else {
                dims.iter()
                    .filter_map(|dim| {
                        if *dim == 0 {
                            None
                        } else {
                            let idx = dim - 1;
                            if idx < shape.len() {
                                Some(idx)
                            } else {
                                None
                            }
                        }
                    })
                    .collect()
            }
        }
        DimSelection::All => (0..shape.len()).collect(),
    };

    reduced_dims.sort_unstable();
    reduced_dims.dedup();

    let reduce_all = !reduced_dims.is_empty()
        && reduced_dims.len() == shape.len()
        && reduced_dims.iter().enumerate().all(|(i, &d)| i == d);

    let output_shape = resolve_output_shape(shape, selection, &reduced_dims)?;
    let mut dims_mask = vec![false; shape.len()];
    for &dim in &reduced_dims {
        if dim < dims_mask.len() {
            dims_mask[dim] = true;
        }
    }
    let reduce_strides = compute_subspace_strides(shape, &reduced_dims);

    Ok(ResolvedDims {
        output_shape,
        reduced_dims,
        reduce_all,
        dims_mask,
        reduce_strides,
    })
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &len in shape {
        strides.push(stride);
        stride = stride.saturating_mul(len.max(1));
    }
    strides
}

fn compute_subspace_strides(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }
    let mut strides = Vec::with_capacity(dims.len());
    let mut accum = 1usize;
    for &dim in dims {
        let len = shape.get(dim).copied().unwrap_or(1).max(1);
        strides.push(accum);
        accum = accum.saturating_mul(len);
    }
    strides
}

fn map_output_index(coords: &[usize], output_strides: &[usize], dims_mask: &[bool]) -> usize {
    if coords.is_empty() {
        return 0;
    }
    let mut index = 0usize;
    for (dim, stride) in output_strides.iter().enumerate() {
        let coord = if *dims_mask.get(dim).unwrap_or(&false) {
            0
        } else {
            coords[dim]
        };
        index = index.saturating_add(coord.saturating_mul(*stride));
    }
    index
}

fn map_reduce_index(
    coords: &[usize],
    reduced_dims: &[usize],
    reduce_strides: &[usize],
    reduce_all: bool,
) -> usize {
    if reduced_dims.is_empty() {
        return 0;
    }
    if reduce_all {
        // When all dimensions are reduced, the full index is used separately.
        return 0;
    }
    let mut index = 0usize;
    for (pos, &dim) in reduced_dims.iter().enumerate() {
        if let Some(coord) = coords.get(dim) {
            if let Some(stride) = reduce_strides.get(pos) {
                index = index.saturating_add(coord.saturating_mul(*stride));
            }
        }
    }
    index
}

fn map_linear_index(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides.iter())
        .fold(0usize, |acc, (&coord, &stride)| {
            acc.saturating_add(coord.saturating_mul(stride))
        })
}

fn increment_coords(coords: &mut [usize], shape: &[usize]) {
    for dim in 0..coords.len() {
        if shape[dim] == 0 {
            continue;
        }
        coords[dim] += 1;
        if coords[dim] < shape[dim] {
            break;
        }
        coords[dim] = 0;
    }
}

fn update_best_real(
    best: &mut BestReal,
    value: f64,
    reduce_index: usize,
    full_index: usize,
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
) {
    if value.is_nan() {
        match nan_mode {
            ReductionNaN::Include => {
                if !best.nan_fixed {
                    best.value = f64::NAN;
                    best.reduce_index = reduce_index;
                    best.full_index = full_index;
                    best.has_value = true;
                    best.nan_fixed = true;
                }
            }
            ReductionNaN::Omit => {}
        }
        return;
    }
    if best.nan_fixed {
        return;
    }

    if !best.has_value {
        best.value = value;
        best.reduce_index = reduce_index;
        best.full_index = full_index;
        best.has_value = true;
        return;
    }

    if should_replace_real(best.value, value, comparison) {
        best.value = value;
        best.reduce_index = reduce_index;
        best.full_index = full_index;
    }
}

fn update_best_complex(
    best: &mut BestComplex,
    value: (f64, f64),
    reduce_index: usize,
    full_index: usize,
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
) {
    if value.0.is_nan() || value.1.is_nan() {
        match nan_mode {
            ReductionNaN::Include => {
                if !best.nan_fixed {
                    best.value = (f64::NAN, f64::NAN);
                    best.reduce_index = reduce_index;
                    best.full_index = full_index;
                    best.has_value = true;
                    best.nan_fixed = true;
                }
            }
            ReductionNaN::Omit => {}
        }
        return;
    }
    if best.nan_fixed {
        return;
    }

    if !best.has_value {
        best.value = value;
        best.reduce_index = reduce_index;
        best.full_index = full_index;
        best.has_value = true;
        return;
    }

    if should_replace_complex(best.value, value, comparison) {
        best.value = value;
        best.reduce_index = reduce_index;
        best.full_index = full_index;
    }
}

fn should_replace_real(current: f64, candidate: f64, comparison: ComparisonMethod) -> bool {
    match comparison {
        ComparisonMethod::Auto | ComparisonMethod::Real => {
            if candidate > current {
                return true;
            }
            if candidate < current {
                return false;
            }
            if candidate == 0.0 && current == 0.0 {
                return candidate.is_sign_positive() && !current.is_sign_positive();
            }
            false
        }
        ComparisonMethod::Abs => {
            let curr_abs = current.abs();
            let cand_abs = candidate.abs();
            if cand_abs > curr_abs {
                return true;
            }
            if cand_abs < curr_abs {
                return false;
            }
            if candidate > current {
                return true;
            }
            if candidate < current {
                return false;
            }
            if candidate == 0.0 && current == 0.0 {
                return candidate.is_sign_positive() && !current.is_sign_positive();
            }
            false
        }
    }
}

fn should_replace_complex(
    current: (f64, f64),
    candidate: (f64, f64),
    comparison: ComparisonMethod,
) -> bool {
    match comparison {
        ComparisonMethod::Auto | ComparisonMethod::Abs => {
            compare_complex_auto(current, candidate) == Ordering::Less
        }
        ComparisonMethod::Real => compare_complex_real(current, candidate) == Ordering::Less,
    }
}

fn compare_complex_auto(a: (f64, f64), b: (f64, f64)) -> Ordering {
    let a_mag = magnitude_squared(a);
    let b_mag = magnitude_squared(b);
    if a_mag < b_mag {
        return Ordering::Less;
    }
    if a_mag > b_mag {
        return Ordering::Greater;
    }
    // Equal magnitude: tie-break using phase angle.
    let a_angle = a.1.atan2(a.0);
    let b_angle = b.1.atan2(b.0);
    if a_angle < b_angle {
        Ordering::Less
    } else if a_angle > b_angle {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn compare_complex_real(a: (f64, f64), b: (f64, f64)) -> Ordering {
    if a.0 < b.0 {
        return Ordering::Less;
    }
    if a.0 > b.0 {
        return Ordering::Greater;
    }
    // Equal real parts: use magnitude and phase tie-breakers.
    compare_complex_auto(a, b)
}

fn magnitude_squared(z: (f64, f64)) -> f64 {
    z.0.mul_add(z.0, z.1 * z.1)
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    for (i, &len) in shape.iter().enumerate() {
        if len > 1 {
            return i + 1;
        }
    }
    1
}

fn elementwise_max(value: Value, args: ElementwiseArgs) -> Result<MaxEvaluation, String> {
    let ElementwiseArgs { other, comparison } = args;
    match (value, other) {
        (Value::GpuTensor(handle_a), Value::GpuTensor(handle_b)) => {
            if gpu_tensor_is_scalar(&handle_b) {
                if let Some(num) = gpu_tensor_scalar_value(&handle_b) {
                    let scalar = Value::Num(num);
                    return elementwise_max_gpu_scalar_left(&handle_a, &scalar, comparison)
                        .or_else(|| {
                            let ta = gpu_helpers::gather_tensor(&handle_a).ok()?;
                            elementwise_real_or_complex(
                                Value::Tensor(ta),
                                scalar.clone(),
                                comparison,
                            )
                            .ok()
                        })
                        .ok_or_else(|| "max: elementwise GPU scalar path failed".to_string());
                }
            }
            if gpu_tensor_is_scalar(&handle_a) {
                if let Some(num) = gpu_tensor_scalar_value(&handle_a) {
                    let scalar = Value::Num(num);
                    return elementwise_max_gpu_scalar_right(&scalar, &handle_b, comparison)
                        .or_else(|| {
                            let tb = gpu_helpers::gather_tensor(&handle_b).ok()?;
                            elementwise_real_or_complex(
                                scalar.clone(),
                                Value::Tensor(tb),
                                comparison,
                            )
                            .ok()
                        })
                        .ok_or_else(|| "max: elementwise GPU scalar path failed".to_string());
                }
            }
            elementwise_max_gpu_pair(&handle_a, &handle_b, comparison)
                .or_else(|| {
                    // Fallback to host path if provider path unavailable or unsupported
                    let ta = gpu_helpers::gather_tensor(&handle_a).ok()?;
                    let tb = gpu_helpers::gather_tensor(&handle_b).ok()?;
                    elementwise_real_or_complex(Value::Tensor(ta), Value::Tensor(tb), comparison)
                        .ok()
                })
                .ok_or_else(|| "max: elementwise GPU path failed".to_string())
        }
        (Value::GpuTensor(handle), other) => {
            elementwise_max_gpu_scalar_left(&handle, &other, comparison)
                .or_else(|| {
                    let t = gpu_helpers::gather_tensor(&handle).ok()?;
                    elementwise_real_or_complex(Value::Tensor(t), other, comparison).ok()
                })
                .ok_or_else(|| "max: elementwise GPU scalar path failed".to_string())
        }
        (other, Value::GpuTensor(handle)) => {
            elementwise_max_gpu_scalar_right(&other, &handle, comparison)
                .or_else(|| {
                    let t = gpu_helpers::gather_tensor(&handle).ok()?;
                    elementwise_real_or_complex(other, Value::Tensor(t), comparison).ok()
                })
                .ok_or_else(|| "max: elementwise GPU scalar path failed".to_string())
        }
        (lhs, rhs) => elementwise_real_or_complex(lhs, rhs, comparison),
    }
}

fn elementwise_max_gpu_pair(
    a: &GpuTensorHandle,
    b: &GpuTensorHandle,
    comparison: ComparisonMethod,
) -> Option<MaxEvaluation> {
    if comparison != ComparisonMethod::Auto {
        return None;
    }
    let provider = runmat_accelerate_api::provider()?;
    // Equal-shape fast path
    if a.shape == b.shape {
        let values = provider.elem_max(a, b).ok()?;
        // Try device mask first; if unavailable, compute indices on host while keeping values on device
        if let Ok(mask) = provider.elem_ge(a, b) {
            let indices = gpu_mask_indices(provider, &mask)?;
            let _ = provider.free(&mask);
            return Some(MaxEvaluation {
                values: Value::GpuTensor(values),
                indices: Value::GpuTensor(indices),
            });
        } else {
            // Host path for indices only
            let ta = gpu_helpers::gather_tensor(a).ok()?;
            let tb = gpu_helpers::gather_tensor(b).ok()?;
            let mut indices = Vec::with_capacity(ta.data.len());
            for i in 0..ta.data.len() {
                indices.push(if ta.data[i] >= tb.data[i] { 1.0 } else { 2.0 });
            }
            let index_tensor = Tensor::new(indices, ta.shape.clone()).ok()?;
            return Some(MaxEvaluation {
                values: Value::GpuTensor(values),
                indices: tensor::tensor_into_value(index_tensor),
            });
        }
    }
    // Broadcast-compatible path via repmat, then device compare
    let (out_shape, reps_a, reps_b) = broadcast_reps(&a.shape, &b.shape)?;
    let a_exp = if reps_a.iter().any(|&r| r != 1) {
        provider.repmat(a, &reps_a).ok()?
    } else {
        a.clone()
    };
    let b_exp = if reps_b.iter().any(|&r| r != 1) {
        provider.repmat(b, &reps_b).ok()?
    } else {
        b.clone()
    };
    let values = provider.elem_max(&a_exp, &b_exp).ok();
    let mask = provider.elem_ge(&a_exp, &b_exp).ok();
    if !std::ptr::eq(&a_exp, a) {
        let _ = provider.free(&a_exp);
    }
    if !std::ptr::eq(&b_exp, b) {
        let _ = provider.free(&b_exp);
    }
    let values = values?;
    if values.shape != out_shape {
        let _ = provider.free(&values);
        return None;
    }
    let index_tensor = if let Some(mask) = mask {
        let mask_host = gpu_helpers::gather_tensor(&mask).ok()?;
        let _ = provider.free(&mask);
        let mut indices = Vec::with_capacity(mask_host.data.len());
        for &m in &mask_host.data {
            indices.push(if m != 0.0 { 1.0 } else { 2.0 });
        }
        Tensor::new(indices, out_shape).ok()?
    } else {
        // Host indices fallback
        let ta = gpu_helpers::gather_tensor(&a_exp).ok()?;
        let tb = gpu_helpers::gather_tensor(&b_exp).ok()?;
        let mut indices = Vec::with_capacity(ta.data.len());
        for i in 0..ta.data.len() {
            indices.push(if ta.data[i] >= tb.data[i] { 1.0 } else { 2.0 });
        }
        Tensor::new(indices, out_shape).ok()?
    };
    Some(MaxEvaluation {
        values: Value::GpuTensor(values),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn broadcast_reps(a: &[usize], b: &[usize]) -> Option<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let rank = a.len().max(b.len()).max(1);
    let mut out = vec![1usize; rank];
    let mut aa = vec![1usize; rank];
    let mut bb = vec![1usize; rank];
    for i in 0..rank {
        aa[i] = *a.get(i).unwrap_or(&1);
        bb[i] = *b.get(i).unwrap_or(&1);
    }
    for i in 0..rank {
        let (ad, bd) = (aa[i], bb[i]);
        if ad == bd {
            out[i] = ad;
        } else if ad == 1 {
            out[i] = bd;
        } else if bd == 1 {
            out[i] = ad;
        } else {
            return None;
        }
    }
    let reps_a: Vec<usize> = (0..rank)
        .map(|i| if aa[i] == out[i] { 1 } else { out[i] })
        .collect();
    let reps_b: Vec<usize> = (0..rank)
        .map(|i| if bb[i] == out[i] { 1 } else { out[i] })
        .collect();
    Some((out, reps_a, reps_b))
}

fn elementwise_max_gpu_scalar_left(
    a: &GpuTensorHandle,
    other: &Value,
    comparison: ComparisonMethod,
) -> Option<MaxEvaluation> {
    if comparison != ComparisonMethod::Auto {
        return None;
    }
    let provider = runmat_accelerate_api::provider()?;
    let scalar = extract_scalar(other)?;
    // Prefer tensorize + elem_max for broader provider compatibility
    let values = if let Ok(fill) = provider.fill_like(a, scalar) {
        let vals = provider.elem_max(a, &fill).ok();
        let _ = provider.free(&fill);
        vals?
    } else {
        provider.scalar_max(a, scalar).ok()?
    };
    // Try device mask; if unavailable, compute on host
    let index_tensor = if let Ok(fill) = provider.fill_like(a, scalar) {
        if let Ok(mask) = provider.elem_ge(a, &fill) {
            let _ = provider.free(&fill);
            let indices = gpu_mask_indices(provider, &mask)?;
            let _ = provider.free(&mask);
            return Some(MaxEvaluation {
                values: Value::GpuTensor(values),
                indices: Value::GpuTensor(indices),
            });
        } else {
            let _ = provider.free(&fill);
            let ta = gpu_helpers::gather_tensor(a).ok()?;
            let mut indices = Vec::with_capacity(ta.data.len());
            for &v in &ta.data {
                indices.push(if v >= scalar { 1.0 } else { 2.0 });
            }
            Tensor::new(indices, ta.shape.clone()).ok()?
        }
    } else {
        let ta = gpu_helpers::gather_tensor(a).ok()?;
        let mut indices = Vec::with_capacity(ta.data.len());
        for &v in &ta.data {
            indices.push(if v >= scalar { 1.0 } else { 2.0 });
        }
        Tensor::new(indices, ta.shape.clone()).ok()?
    };
    Some(MaxEvaluation {
        values: Value::GpuTensor(values),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn elementwise_max_gpu_scalar_right(
    other: &Value,
    b: &GpuTensorHandle,
    comparison: ComparisonMethod,
) -> Option<MaxEvaluation> {
    if comparison != ComparisonMethod::Auto {
        return None;
    }
    let provider = runmat_accelerate_api::provider()?;
    let scalar = extract_scalar(other)?;
    let values = if let Ok(fill) = provider.fill_like(b, scalar) {
        let vals = provider.elem_max(&fill, b).ok();
        let _ = provider.free(&fill);
        vals?
    } else {
        provider.scalar_max(b, scalar).ok()?
    };
    // Try device mask; if unavailable, compute on host (origin 1 if scalar >= b)
    let index_tensor = if let Ok(fill) = provider.fill_like(b, scalar) {
        if let Ok(mask) = provider.elem_ge(&fill, b) {
            let _ = provider.free(&fill);
            let indices = gpu_mask_indices(provider, &mask)?;
            let _ = provider.free(&mask);
            return Some(MaxEvaluation {
                values: Value::GpuTensor(values),
                indices: Value::GpuTensor(indices),
            });
        } else {
            let _ = provider.free(&fill);
            let tb = gpu_helpers::gather_tensor(b).ok()?;
            let mut indices = Vec::with_capacity(tb.data.len());
            for &v in &tb.data {
                indices.push(if scalar >= v { 1.0 } else { 2.0 });
            }
            Tensor::new(indices, tb.shape.clone()).ok()?
        }
    } else {
        let tb = gpu_helpers::gather_tensor(b).ok()?;
        let mut indices = Vec::with_capacity(tb.data.len());
        for &v in &tb.data {
            indices.push(if scalar >= v { 1.0 } else { 2.0 });
        }
        Tensor::new(indices, tb.shape.clone()).ok()?
    };
    Some(MaxEvaluation {
        values: Value::GpuTensor(values),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn extract_scalar(v: &Value) -> Option<f64> {
    match v {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => t.data.first().copied(),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn gpu_tensor_is_scalar(handle: &GpuTensorHandle) -> bool {
    handle.shape.iter().copied().product::<usize>().max(1) == 1
}

fn gpu_tensor_scalar_value(handle: &GpuTensorHandle) -> Option<f64> {
    let tensor = gpu_helpers::gather_tensor(handle).ok()?;
    tensor.data.first().copied()
}

fn gpu_mask_indices(
    provider: &dyn AccelProvider,
    mask: &GpuTensorHandle,
) -> Option<GpuTensorHandle> {
    let scaled = provider.scalar_mul(mask, -1.0).ok()?;
    let shifted = provider.scalar_add(&scaled, 2.0).ok()?;
    let _ = provider.free(&scaled);
    Some(shifted)
}

fn elementwise_real_or_complex(
    lhs: Value,
    rhs: Value,
    comparison: ComparisonMethod,
) -> Result<MaxEvaluation, String> {
    match (
        materialize_for_max("max", lhs)?,
        materialize_for_max("max", rhs)?,
    ) {
        (InputData::Complex(a), InputData::Complex(b)) => elementwise_complex_max(a, b, comparison),
        (InputData::Complex(a), InputData::Real(b)) => {
            let converted = promote_real_tensor_to_complex(b);
            elementwise_complex_max(a, converted, comparison)
        }
        (InputData::Real(a), InputData::Complex(b)) => {
            let converted = promote_real_tensor_to_complex(a);
            elementwise_complex_max(converted, b, comparison)
        }
        (InputData::Real(a), InputData::Real(b)) => elementwise_real_max(a, b, comparison),
    }
}

fn elementwise_real_max(
    lhs: Tensor,
    rhs: Tensor,
    comparison: ComparisonMethod,
) -> Result<MaxEvaluation, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| format!("max: {}", err))?;
    let mut values = vec![0.0f64; plan.len()];
    let mut indices = vec![0.0f64; plan.len()];

    for (offset, index_a, index_b) in plan.iter() {
        let a = lhs.data.get(index_a).copied().unwrap_or(f64::NAN);
        let b = rhs.data.get(index_b).copied().unwrap_or(f64::NAN);
        let (value, origin) = choose_real_elementwise(a, b, comparison);
        values[offset] = value;
        indices[offset] = origin;
    }

    let value_tensor =
        Tensor::new(values, plan.output_shape().to_vec()).map_err(|e| format!("max: {e}"))?;
    let index_tensor =
        Tensor::new(indices, plan.output_shape().to_vec()).map_err(|e| format!("max: {e}"))?;

    Ok(MaxEvaluation {
        values: tensor::tensor_into_value(value_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn elementwise_complex_max(
    lhs: ComplexTensor,
    rhs: ComplexTensor,
    comparison: ComparisonMethod,
) -> Result<MaxEvaluation, String> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape).map_err(|err| format!("max: {}", err))?;
    let mut values = vec![(0.0f64, 0.0f64); plan.len()];
    let mut indices = vec![0.0f64; plan.len()];

    for (offset, index_a, index_b) in plan.iter() {
        let a = lhs
            .data
            .get(index_a)
            .copied()
            .unwrap_or((f64::NAN, f64::NAN));
        let b = rhs
            .data
            .get(index_b)
            .copied()
            .unwrap_or((f64::NAN, f64::NAN));
        let (value, origin) = choose_complex_elementwise(a, b, comparison);
        values[offset] = value;
        indices[offset] = origin;
    }

    let value_tensor = ComplexTensor::new(values, plan.output_shape().to_vec())
        .map_err(|e| format!("max: {e}"))?;
    let index_tensor =
        Tensor::new(indices, plan.output_shape().to_vec()).map_err(|e| format!("max: {e}"))?;

    Ok(MaxEvaluation {
        values: complex_tensor_into_value(value_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn promote_real_tensor_to_complex(tensor: Tensor) -> ComplexTensor {
    let data = tensor
        .data
        .iter()
        .copied()
        .map(|re| (re, 0.0))
        .collect::<Vec<_>>();
    ComplexTensor {
        data,
        shape: tensor.shape.clone(),
        rows: tensor.rows,
        cols: tensor.cols,
    }
}

fn choose_real_elementwise(a: f64, b: f64, comparison: ComparisonMethod) -> (f64, f64) {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => (f64::NAN, 1.0),
        (true, false) => (f64::NAN, 1.0),
        (false, true) => (f64::NAN, 2.0),
        (false, false) => {
            if should_replace_real(a, b, comparison) {
                (b, 2.0)
            } else {
                (a, 1.0)
            }
        }
    }
}

fn choose_complex_elementwise(
    a: (f64, f64),
    b: (f64, f64),
    comparison: ComparisonMethod,
) -> ((f64, f64), f64) {
    let a_nan = a.0.is_nan() || a.1.is_nan();
    let b_nan = b.0.is_nan() || b.1.is_nan();
    match (a_nan, b_nan) {
        (true, true) => ((f64::NAN, f64::NAN), 1.0),
        (true, false) => ((f64::NAN, f64::NAN), 1.0),
        (false, true) => ((f64::NAN, f64::NAN), 2.0),
        (false, false) => {
            if should_replace_complex(a, b, comparison) {
                (b, 2.0)
            } else {
                (a, 1.0)
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, Tensor, Value};

    fn placeholder() -> Value {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        Value::Tensor(tensor)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_scalar_returns_input() {
        let result = max_builtin(Value::Num(5.0), Vec::new()).expect("max");
        assert_eq!(result, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_vector_with_indices() {
        let tensor = Tensor::new(vec![3.0, 1.0, 5.0], vec![3, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(5.0));
        assert_eq!(indices, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_matrix_default_dimension() {
        let tensor = Tensor::new(vec![3.0, 4.0, 1.0, 2.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![4.0, 2.0, 6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![2.0, 2.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_all_linear_index() {
        let tensor =
            Tensor::new((1..=12).map(|v| v as f64).collect::<Vec<_>>(), vec![3, 4]).unwrap();
        let args = vec![placeholder(), Value::from("all")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(12.0));
        assert_eq!(indices, Value::Num(12.0));

        let args_linear = vec![placeholder(), Value::from("linear")];
        let eval = evaluate(
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
            &args_linear,
        )
        .expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(3.0));
        assert_eq!(indices, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_with_omitnan() {
        let tensor = Tensor::new(vec![f64::NAN, 4.0, 2.0], vec![3, 1]).unwrap();
        let args = vec![placeholder(), Value::from("omitnan")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(4.0));
        assert_eq!(indices, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_omitnan_all_nan_slice() {
        let tensor = Tensor::new(vec![f64::NAN, f64::NAN], vec![2, 1]).unwrap();
        let args = vec![placeholder(), Value::from("omitnan")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
        match indices {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar NaN index, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_reduction_abs_comparison() {
        let tensor = Tensor::new(vec![1.0, -3.0, -2.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![
            placeholder(),
            Value::from("ComparisonMethod"),
            Value::from("abs"),
        ];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![-3.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![2.0, 2.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_reduction_complex_real_comparison() {
        let tensor = ComplexTensor::new(vec![(1.0, 2.0), (0.5, 5.0)], vec![2, 1]).expect("tensor");
        let args = vec![
            placeholder(),
            Value::from("ComparisonMethod"),
            Value::from("real"),
        ];
        let eval = evaluate(Value::ComplexTensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Complex(re, im) => {
                assert!((re - 1.0).abs() < 1e-12);
                assert!((im - 2.0).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
        assert_eq!(indices, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_elementwise_broadcast() {
        let lhs = Tensor::new(vec![1.0, 4.0, 7.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![2.0, 3.0, 5.0], vec![3, 1]).unwrap();
        let eval = evaluate(Value::Tensor(lhs), &[Value::Tensor(rhs)]).expect("evaluate");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                assert_eq!([t.data[0], t.data[3], t.data[6]], [2.0, 4.0, 7.0]);
                assert_eq!([t.data[1], t.data[4], t.data[7]], [3.0, 4.0, 7.0]);
                assert_eq!([t.data[2], t.data[5], t.data[8]], [5.0, 5.0, 7.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                assert_eq!([t.data[0], t.data[3], t.data[6]], [2.0, 1.0, 1.0]);
                assert_eq!([t.data[1], t.data[4], t.data[7]], [2.0, 1.0, 1.0]);
                assert_eq!([t.data[2], t.data[5], t.data[8]], [2.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_elementwise_abs_comparison() {
        let lhs = Tensor::new(vec![-2.0, 1.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![1.5, -3.0], vec![2, 1]).unwrap();
        let args = vec![
            Value::Tensor(rhs),
            Value::from("ComparisonMethod"),
            Value::from("abs"),
        ];
        let eval = evaluate(Value::Tensor(lhs), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![-2.0, -3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_elementwise_rejects_reduction_only_keywords() {
        let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let rhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
        let err = evaluate(
            Value::Tensor(lhs),
            &[Value::Tensor(rhs), Value::from("omitnan")],
        )
        .expect_err("expected error");
        assert!(err.contains("only supported for reduction"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_complex_real_comparison() {
        let lhs = ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let rhs = ComplexTensor::new(vec![(0.5, 5.0)], vec![1, 1]).unwrap();
        let args = vec![
            Value::ComplexTensor(rhs),
            Value::from("ComparisonMethod"),
            Value::from("real"),
        ];
        let eval = evaluate(Value::ComplexTensor(lhs), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Complex(1.0, 2.0));
        assert_eq!(indices, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_dimension_argument_parsing() {
        let tensor = Tensor::new(vec![3.0, 4.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = vec![placeholder(), Value::Tensor(dims)];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(4.0));
        assert_eq!(indices, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_vecdim_duplicate_entries() {
        let tensor = Tensor::new(vec![5.0, 2.0, 7.0, 1.0], vec![2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let args = vec![placeholder(), Value::Tensor(dims)];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(7.0));
        assert_eq!(indices, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_dimension_gpu_argument_errors() {
        let tensor = Tensor::new(vec![3.0, 1.0], vec![2, 1]).unwrap();
        let dim_handle = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 42,
        });
        let err = evaluate(Value::Tensor(tensor), &[placeholder(), dim_handle])
            .expect_err("expected error");
        assert!(err.contains("dimension arguments must reside on the host"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_invalid_comparison_method_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = vec![
            placeholder(),
            Value::from("ComparisonMethod"),
            Value::from("chebyshev"),
        ];
        let err = evaluate(Value::Tensor(tensor), &args).expect_err("expected error");
        assert!(err.contains("unsupported ComparisonMethod"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_doc_examples_present() {
        let blocks = test_support::doc_examples(super::DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn max_gpu_dim1_matches_cpu() {
        let tensor = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let eval_cpu = evaluate(Value::Tensor(tensor.clone()), &[]).expect("cpu");
        let (values_cpu, indices_cpu) = eval_cpu.into_pair();

        test_support::with_test_provider(|provider| {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval_gpu = evaluate(Value::GpuTensor(handle), &[]).expect("gpu");
            let (values_gpu, indices_gpu) = eval_gpu.into_pair();
            match (&values_gpu, &indices_gpu) {
                (Value::GpuTensor(_), Value::GpuTensor(_)) => {}
                other => panic!("expected GPU tensors, got {other:?}"),
            }
            let gathered_vals = test_support::gather(values_gpu).expect("gather values");
            let gathered_idx = test_support::gather(indices_gpu).expect("gather indices");
            let expected_vals = match values_cpu {
                Value::Tensor(t) => t,
                other => panic!("expected tensor values from cpu eval, got {other:?}"),
            };
            let expected_idx = match indices_cpu {
                Value::Tensor(t) => t,
                other => panic!("expected tensor indices from cpu eval, got {other:?}"),
            };
            assert_eq!(gathered_vals.shape, expected_vals.shape);
            assert_eq!(gathered_vals.data, expected_vals.data);
            assert_eq!(gathered_idx.shape, expected_idx.shape);
            assert_eq!(gathered_idx.data, expected_idx.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_dimension_numeric_argument() {
        let tensor = Tensor::new(vec![3.0, 4.0, 1.0, 2.0], vec![2, 2]).unwrap();
        let args = vec![placeholder(), Value::Num(2.0)];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.data, vec![3.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_complex_auto_comparison() {
        let lhs = ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let rhs = ComplexTensor::new(vec![(2.0, 1.0)], vec![1, 1]).unwrap();
        let eval =
            evaluate(Value::ComplexTensor(lhs), &[Value::ComplexTensor(rhs)]).expect("evaluate");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Complex(1.0, 2.0));
        assert_eq!(indices, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_scalar_pair_arguments() {
        let args = vec![Value::Num(2.0)];
        let result = max_builtin(Value::Num(3.0), args).expect("max");
        assert_eq!(result, Value::Num(3.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_rejects_invalid_dimension() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![placeholder(), Value::Int(IntValue::I32(0))];
        let err = evaluate(Value::Tensor(tensor), &args).expect_err("expected error");
        assert!(err.contains("dimension must be >= 1"));
    }
}
