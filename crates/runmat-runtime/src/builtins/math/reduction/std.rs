//! MATLAB-compatible `std` builtin with GPU-aware semantics for RunMat.
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, HostTensorView, ProviderNanMode, ProviderPrecision,
    ProviderStdNormalization,
};
use runmat_builtins::{ComplexTensor, IntValue, NumericDType, Tensor, Value};
const NAME: &str = "std";

use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::dispatcher;
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "std"
category: "math/reduction"
keywords: ["std", "standard deviation", "statistics", "gpu", "omitnan", "all", "like", "native"]
summary: "Standard deviation of scalars, vectors, matrices, or N-D tensors with MATLAB-compatible options."
references: []
gpu_support:
  elementwise: false
  reduction: true
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to host when 'omitnan' is requested or the provider lacks std reductions."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::reduction::std::tests"
  integration: "builtins::math::reduction::std::tests::std_gpu_provider_roundtrip"
  gpu: "builtins::math::reduction::std::tests::std_wgpu_dim1_sample_matches_cpu"
---

# What does the `std` function do in MATLAB / RunMat?
`std(x)` measures the spread of the elements in `x`. By default RunMat matches MATLAB’s sample definition (dividing by `n-1`) and works along the first non-singleton dimension.

## How does the `std` function behave in MATLAB / RunMat?
- `std(X)` on an `m × n` matrix returns a `1 × n` row vector with the sample standard deviation of each column.
- `std(X, 1)` switches to population normalisation (`n` in the denominator). Use `std(X, 0)` or `std(X, [])` to keep the default sample behaviour.
- `std(X, flag, dim)` lets you pick both the normalisation (`flag = 0` sample, `1` population, or `[]`) and the dimension to reduce. `std(X, flag, 'all')` collapses every dimension, while `std(X, flag, vecdim)` accepts a dimension vector such as `[1 3]` and reduces all listed axes in a single call. Multi-axis reductions execute on the host today when the active GPU provider cannot fuse them.
- Strings like `'omitnan'` and `'includenan'` decide whether `NaN` values are skipped or propagated.
- Optional out-type arguments (`'double'`, `'default'`, `'native'`, or `'like', prototype`) mirror MATLAB behaviour. `'native'` rounds scalar integer results back to their original class; `'like'` mirrors both the numeric class and device residency of `prototype` (complex prototypes yield complex outputs with zero imaginary parts).
- Logical inputs are promoted to double precision before reduction so that results follow MATLAB’s numeric rules.
- Empty slices return `NaN` with MATLAB-compatible shapes. Scalars return `0`, regardless of the normalisation mode.
- Dimensions greater than `ndims(X)` leave the input untouched.
- Weighted standard deviations (`flag` as a vector) are not implemented yet; RunMat reports a descriptive error when they are requested.

Complex tensors are not currently supported; convert them to real magnitudes manually before calling `std`.

## `std` Function GPU Execution Behaviour
When RunMat Accelerate is active, device-resident tensors remain on the GPU whenever the provider implements the relevant hooks. Providers that expose `reduce_std_dim`/`reduce_std` execute the reduction in-place on the device; the default WGPU backend currently supports two-dimensional inputs, single-axis reductions, and `'includenan'` only. Whenever `'omitnan'`, multi-axis reductions, or unsupported shapes are requested, RunMat transparently gathers the data to the host, computes the result there, and then applies the requested output template (`'native'`, `'like'`) before returning.

## Examples of using the `std` function in MATLAB / RunMat

### Sample standard deviation of a vector

```matlab
x = [1 2 3 4 5];
s = std(x);                 % uses flag = 0 (sample) by default
```

Expected output:

```matlab
s = 1.5811;
```

### Population standard deviation of each column

```matlab
A = [1 3 5; 2 4 6];
spop = std(A, 1);           % divide by n instead of n-1
```

Expected output:

```matlab
spop = [0.5 0.5 0.5];
```

### Collapsing every dimension at once

```matlab
B = reshape(1:12, [3 4]);
overall = std(B, 0, 'all');
```

Expected output:

```matlab
overall = 3.6056;
```

### Reducing across multiple dimensions

```matlab
C = cat(3, [1 2; 3 4], [5 6; 7 8]);
sliceStd = std(C, [], [1 3]);   % keep columns, reduce rows & pages
```

Expected output:

```matlab
sliceStd = [2.5820 2.5820];
```

### Ignoring NaN values

```matlab
D = [1 NaN 3; 2 4 NaN];
rowStd = std(D, 0, 2, 'omitnan');
```

Expected output:

```matlab
rowStd = [1.4142; 1.4142];
```

### Matching a prototype using `'like'`

```matlab
proto = gpuArray(single(42));
G = gpuArray(rand(1024, 512));
spread = std(G, 1, 'all', 'like', proto);
answer = gather(spread);
```

`spread` stays on the GPU as a single-precision scalar because it inherits the prototype’s class and residency; `answer` equals the scalar gathered back to the host.

### Preserving default behaviour with an empty normalisation flag

```matlab
C = [1 2; 3 4];
rowStd = std(C, [], 2);
```

Expected output:

```matlab
rowStd = [0.7071; 0.7071];
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Usually you do not need to call `gpuArray` manually. The fusion planner keeps tensors on the GPU across fused expressions and gathers them only when necessary. For explicit control or MATLAB compatibility, you can still call `gpuArray`/`gather` yourself.

## FAQ

### What values can I pass as the normalisation flag?
Use `0` (or `[]`) for the sample definition, `1` for population. RunMat rejects non-scalar weight vectors and reports that weighted standard deviations are not implemented yet.

### How can I collapse multiple dimensions?
Pass a vector of dimensions such as `std(A, [], [1 3])`. You can also use `'all'` to collapse every dimension into a single scalar.

### How do `'omitnan'` and `'includenan'` work?
`'omitnan'` skips NaN values; if every element in a slice is NaN the result is NaN. `'includenan'` (the default) propagates a single NaN to the output slice.

### What do `'native'` and `'like'` do?
`'native'` rounds scalar results back to the input’s integer class (multi-element outputs stay in double precision for now), while `'double'`/`'default'` keep double precision. `'like', prototype` mirrors both the numeric class and the device residency of `prototype`, including GPU tensors; complex prototypes produce complex outputs with zero imaginary parts.

### What happens if I request a dimension greater than `ndims(X)`?
RunMat returns the input unchanged so that MATLAB-compatible code relying on that behaviour continues to work.

### Are complex inputs supported?
Not yet. RunMat currently requires real inputs for `std`. Convert complex data to magnitude or separate real/imaginary parts before calling the builtin.

## See Also
[mean](./mean), [sum](./sum), [median](./median), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `std` function is available at: [`crates/runmat-runtime/src/builtins/math/reduction/std.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/std.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "std",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Reduction {
            name: "reduce_std_dim",
        },
        ProviderHook::Reduction {
            name: "reduce_std",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
    notes: "Providers may offer reduce_std_dim/reduce_std implementations; host fallback ensures correctness when they are unavailable.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "std",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion currently gathers to the host; future kernels can reuse the variance accumulator directly.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("std", DOC_MD);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StdNormalization {
    Sample,
    Population,
}

#[derive(Clone, Debug)]
enum StdAxes {
    Default,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

#[derive(Clone)]
struct ParsedArguments {
    axes: StdAxes,
    normalization: StdNormalization,
    nan_mode: ReductionNaN,
    output: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    Native,
    Like(Value),
}

#[derive(Clone, Copy)]
enum DevicePreference {
    Host,
    Gpu,
}

#[derive(Clone, Copy)]
enum InputClass {
    Double,
    Complex,
    Logical,
    Integer(IntClass),
    Bool,
}

#[derive(Clone, Copy)]
enum IntClass {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

struct InputMeta {
    class: InputClass,
    device: DevicePreference,
    numeric_dtype: Option<NumericDType>,
}

impl InputMeta {
    fn from_value(value: &Value) -> Self {
        let class = match value {
            Value::Complex(_, _) | Value::ComplexTensor(_) => InputClass::Complex,
            Value::LogicalArray(_) => InputClass::Logical,
            Value::Bool(_) => InputClass::Bool,
            Value::Int(i) => InputClass::Integer(IntClass::from_int_value(i)),
            _ => InputClass::Double,
        };
        let device = match value {
            Value::GpuTensor(_) => DevicePreference::Gpu,
            _ => DevicePreference::Host,
        };
        let numeric_dtype = numeric_dtype_from_value(value);
        Self {
            class,
            device,
            numeric_dtype,
        }
    }
}

fn numeric_dtype_from_value(value: &Value) -> Option<NumericDType> {
    match value {
        Value::Tensor(t) => Some(t.dtype),
        Value::GpuTensor(handle) => {
            let precision = runmat_accelerate_api::handle_precision(handle).or_else(|| {
                runmat_accelerate_api::provider_for_handle(handle)
                    .map(|provider| provider.precision())
            });
            precision.map(precision_to_dtype)
        }
        Value::Num(_) => Some(NumericDType::F64),
        Value::LogicalArray(_) => Some(NumericDType::F64),
        _ => None,
    }
}

fn precision_to_dtype(precision: ProviderPrecision) -> NumericDType {
    match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    }
}

impl IntClass {
    fn from_int_value(value: &IntValue) -> Self {
        match value {
            IntValue::I8(_) => IntClass::I8,
            IntValue::I16(_) => IntClass::I16,
            IntValue::I32(_) => IntClass::I32,
            IntValue::I64(_) => IntClass::I64,
            IntValue::U8(_) => IntClass::U8,
            IntValue::U16(_) => IntClass::U16,
            IntValue::U32(_) => IntClass::U32,
            IntValue::U64(_) => IntClass::U64,
        }
    }

    fn to_value(self, scalar: f64) -> Result<Value, String> {
        if scalar.is_nan() {
            return Err("std: cannot represent NaN as an integer output".to_string());
        }
        let rounded = scalar.round();
        if !rounded.is_finite() {
            return Err("std: integer output overflowed the target type".to_string());
        }
        Ok(match self {
            IntClass::I8 => Value::Int(IntValue::I8(rounded as i8)),
            IntClass::I16 => Value::Int(IntValue::I16(rounded as i16)),
            IntClass::I32 => Value::Int(IntValue::I32(rounded as i32)),
            IntClass::I64 => Value::Int(IntValue::I64(rounded as i64)),
            IntClass::U8 => Value::Int(IntValue::U8(rounded as u8)),
            IntClass::U16 => Value::Int(IntValue::U16(rounded as u16)),
            IntClass::U32 => Value::Int(IntValue::U32(rounded as u32)),
            IntClass::U64 => Value::Int(IntValue::U64(rounded as u64)),
        })
    }
}

enum NormParse {
    NotMatched,
    Placeholder,
    Value(StdNormalization),
}

#[runtime_builtin(
    name = "std",
    category = "math/reduction",
    summary = "Standard deviation of scalars, vectors, matrices, or N-D tensors.",
    keywords = "std,standard deviation,statistics,gpu,omitnan,all,like,native",
    accel = "reduction"
)]
fn std_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let input_meta = InputMeta::from_value(&value);
    let parsed = parse_arguments(&rest)?;
    let raw = match value {
        Value::GpuTensor(handle) => std_gpu(handle, &parsed)?,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err("std: complex inputs are not supported yet".to_string())
        }
        other => std_host(other, &parsed)?,
    };
    apply_output_template(raw, &parsed.output, &input_meta)
}

fn parse_arguments(args: &[Value]) -> Result<ParsedArguments, String> {
    let mut axes = StdAxes::Default;
    let mut axes_set = false;
    let mut normalization = StdNormalization::Sample;
    let mut normalization_consumed = false;
    let mut nan_mode = ReductionNaN::Include;
    let mut output = OutputTemplate::Double;
    let mut output_set = false;

    let mut idx = 0;
    while idx < args.len() {
        let arg = &args[idx];
        if let Some(keyword) = keyword_of(arg) {
            match keyword.as_str() {
                "omitnan" => {
                    nan_mode = ReductionNaN::Omit;
                    idx += 1;
                    continue;
                }
                "includenan" => {
                    nan_mode = ReductionNaN::Include;
                    idx += 1;
                    continue;
                }
                "double" | "default" => {
                    if output_set {
                        return Err(
                            "std: multiple output class specifications provided".to_string()
                        );
                    }
                    output = OutputTemplate::Double;
                    output_set = true;
                    idx += 1;
                    continue;
                }
                "native" => {
                    if output_set {
                        return Err(
                            "std: multiple output class specifications provided".to_string()
                        );
                    }
                    output = OutputTemplate::Native;
                    output_set = true;
                    idx += 1;
                    continue;
                }
                "like" => {
                    if output_set {
                        return Err(
                            "std: cannot combine 'like' with another output class specifier"
                                .to_string(),
                        );
                    }
                    let Some(proto) = args.get(idx + 1).cloned() else {
                        return Err("std: expected prototype after 'like'".to_string());
                    };
                    output = OutputTemplate::Like(proto);
                    idx += 2;
                    if idx < args.len() {
                        return Err("std: 'like' must be the final argument".to_string());
                    }
                    break;
                }
                "all" => {
                    if axes_set && !matches!(axes, StdAxes::Default) {
                        return Err(
                            "std: 'all' cannot be combined with an explicit dimension".to_string()
                        );
                    }
                }
                _ => {}
            }
        }

        if !normalization_consumed {
            match parse_normalization(arg)? {
                NormParse::Value(norm) => {
                    normalization = norm;
                    normalization_consumed = true;
                    idx += 1;
                    continue;
                }
                NormParse::Placeholder => {
                    normalization_consumed = true;
                    idx += 1;
                    continue;
                }
                NormParse::NotMatched => {}
            }
        }

        if !axes_set || matches!(axes, StdAxes::Default) {
            match parse_axes(arg)? {
                Some(selection) => {
                    if matches!(selection, StdAxes::All)
                        && axes_set
                        && !matches!(axes, StdAxes::Default)
                    {
                        return Err(
                            "std: 'all' cannot be combined with an explicit dimension".to_string()
                        );
                    }
                    axes = selection;
                    axes_set = true;
                    idx += 1;
                    continue;
                }
                None => {}
            }
        } else if let Some(selection) = parse_axes(arg)? {
            if matches!(selection, StdAxes::All) {
                return Err("std: 'all' cannot be combined with an explicit dimension".to_string());
            }
            return Err("std: multiple dimension specifications provided".to_string());
        }

        return Err(format!("std: unrecognised argument {arg:?}"));
    }

    Ok(ParsedArguments {
        axes,
        normalization,
        nan_mode,
        output,
    })
}

fn parse_normalization(value: &Value) -> Result<NormParse, String> {
    match value {
        Value::Tensor(tensor) => {
            if tensor.data.is_empty() {
                return Ok(NormParse::Placeholder);
            }
            if tensor.data.len() == 1 {
                let scalar = tensor.data[0];
                return parse_normalization_scalar(scalar);
            }
            Ok(NormParse::NotMatched)
        }
        Value::LogicalArray(logical) => {
            if logical.data.is_empty() {
                return Ok(NormParse::Placeholder);
            }
            if logical.data.len() == 1 {
                return parse_normalization_scalar(if logical.data[0] != 0 { 1.0 } else { 0.0 });
            }
            Ok(NormParse::NotMatched)
        }
        Value::Bool(flag) => Ok(NormParse::Value(if *flag {
            StdNormalization::Population
        } else {
            StdNormalization::Sample
        })),
        Value::Int(i) => match i.to_i64() {
            0 => Ok(NormParse::Value(StdNormalization::Sample)),
            1 => Ok(NormParse::Value(StdNormalization::Population)),
            _ => Ok(NormParse::NotMatched),
        },
        Value::Num(n) => parse_normalization_scalar(*n),
        Value::GpuTensor(_) => Err("std: normalisation flag must reside on the host".to_string()),
        _ => Ok(NormParse::NotMatched),
    }
}

fn parse_normalization_scalar(value: f64) -> Result<NormParse, String> {
    if !value.is_finite() {
        return Err("std: normalisation flag must be finite".to_string());
    }
    if (value - 0.0).abs() < f64::EPSILON {
        return Ok(NormParse::Value(StdNormalization::Sample));
    }
    if (value - 1.0).abs() < f64::EPSILON {
        return Ok(NormParse::Value(StdNormalization::Population));
    }
    Ok(NormParse::NotMatched)
}

fn parse_axes(value: &Value) -> Result<Option<StdAxes>, String> {
    if let Some(text) = value_as_str(value) {
        let lowered = text.trim().to_ascii_lowercase();
        return match lowered.as_str() {
            "all" => Ok(Some(StdAxes::All)),
            "omitnan" | "includenan" | "double" | "native" | "default" | "like" => Ok(None),
            "" => Err("std: dimension string must not be empty".to_string()),
            _ => Ok(None),
        };
    }

    match value {
        Value::Tensor(t) => {
            if t.data.is_empty() {
                return Ok(Some(StdAxes::Default));
            }
            if t.data.len() == 1 {
                let dim = tensor::parse_dimension(&Value::Num(t.data[0]), "std")?;
                return Ok(Some(StdAxes::Dim(dim)));
            }
            let dims = parse_dimension_vector(t)?;
            Ok(Some(StdAxes::Vec(dims)))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(logical)?;
            if tensor.data.is_empty() {
                return Ok(Some(StdAxes::Default));
            }
            if tensor.data.len() == 1 {
                let dim = tensor::parse_dimension(&Value::Num(tensor.data[0]), "std")?;
                return Ok(Some(StdAxes::Dim(dim)));
            }
            let dims = parse_dimension_vector(&tensor)?;
            Ok(Some(StdAxes::Vec(dims)))
        }
        Value::Int(_) | Value::Num(_) => {
            let dim = tensor::parse_dimension(value, "std")?;
            Ok(Some(StdAxes::Dim(dim)))
        }
        Value::GpuTensor(_) => Err("std: dimension arguments cannot be GPU tensors".to_string()),
        Value::Bool(_) => Err("std: dimension must be numeric".to_string()),
        _ => Ok(None),
    }
}

fn value_as_str(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn parse_dimension_vector(tensor: &Tensor) -> Result<Vec<usize>, String> {
    let mut dims = Vec::with_capacity(tensor.data.len());
    for &val in &tensor.data {
        if !val.is_finite() {
            return Err("std: dimension entries must be finite integers".to_string());
        }
        let rounded = val.round();
        if (rounded - val).abs() > f64::EPSILON {
            return Err("std: dimension entries must be integers".to_string());
        }
        if rounded < 1.0 {
            return Err("std: dimension entries must be >= 1".to_string());
        }
        dims.push(rounded as usize);
    }
    if dims.is_empty() {
        return Err("std: dimension vector must not be empty".to_string());
    }
    Ok(dims)
}

fn std_host(value: Value, args: &ParsedArguments) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor_for("std", value)?;
    let reduced = std_tensor(tensor, &args.axes, args.normalization, args.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn std_tensor(
    tensor: Tensor,
    axes: &StdAxes,
    normalization: StdNormalization,
    nan_mode: ReductionNaN,
) -> Result<Tensor, String> {
    let (dims, had_request) = resolve_axes(&tensor.shape, axes)?;
    if dims.is_empty() {
        if had_request && tensor.data.len() == 1 {
            return std_scalar_tensor(&tensor, nan_mode);
        }
        return Ok(tensor);
    }
    std_tensor_reduce(&tensor, &dims, normalization, nan_mode)
}

fn std_scalar_tensor(tensor: &Tensor, nan_mode: ReductionNaN) -> Result<Tensor, String> {
    let value = tensor.data.get(0).copied().unwrap_or(f64::NAN);
    let result = if value.is_nan() {
        f64::NAN
    } else {
        match nan_mode {
            ReductionNaN::Include | ReductionNaN::Omit => 0.0,
        }
    };
    Tensor::new(vec![result], vec![1, 1]).map_err(|e| format!("std: {e}"))
}

fn std_tensor_reduce(
    tensor: &Tensor,
    dims: &[usize],
    normalization: StdNormalization,
    nan_mode: ReductionNaN,
) -> Result<Tensor, String> {
    let mut dims_sorted = dims.to_vec();
    dims_sorted.sort_unstable();
    dims_sorted.dedup();
    if dims_sorted.is_empty() {
        return Ok(tensor.clone());
    }

    let output_shape = reduced_shape(&tensor.shape, &dims_sorted);
    let out_len = tensor::element_count(&output_shape);
    if tensor.data.is_empty() {
        let fill = vec![f64::NAN; out_len];
        return Tensor::new(fill, output_shape).map_err(|e| format!("std: {e}"));
    }

    let mut counts = vec![0usize; out_len];
    let mut means = vec![0.0f64; out_len];
    let mut m2 = vec![0.0f64; out_len];
    let mut saw_nan = vec![false; out_len];
    let mut coords = vec![0usize; tensor.shape.len()];
    let mut out_coords = vec![0usize; tensor.shape.len()];
    let mut reduce_mask = vec![false; tensor.shape.len()];
    for &dim in &dims_sorted {
        if dim < reduce_mask.len() {
            reduce_mask[dim] = true;
        }
    }

    for (linear, &value) in tensor.data.iter().enumerate() {
        linear_to_multi(linear, &tensor.shape, &mut coords);
        for (i, coord) in coords.iter().enumerate() {
            out_coords[i] = if reduce_mask[i] { 0 } else { *coord };
        }
        let out_idx = multi_to_linear(&out_coords, &output_shape);
        if value.is_nan() {
            if matches!(nan_mode, ReductionNaN::Include) {
                saw_nan[out_idx] = true;
            }
            continue;
        }

        let mean = &mut means[out_idx];
        let m2_slot = &mut m2[out_idx];
        counts[out_idx] += 1;
        let count = counts[out_idx];
        let delta = value - *mean;
        *mean += delta / (count as f64);
        let delta2 = value - *mean;
        *m2_slot += delta * delta2;
    }

    let mut output = vec![0.0f64; out_len];
    for idx in 0..out_len {
        output[idx] = if saw_nan[idx] && matches!(nan_mode, ReductionNaN::Include) {
            f64::NAN
        } else if counts[idx] == 0 {
            f64::NAN
        } else {
            let count = counts[idx];
            let variance = match normalization {
                StdNormalization::Sample => {
                    if count > 1 {
                        (m2[idx] / (count - 1) as f64).max(0.0)
                    } else {
                        0.0
                    }
                }
                StdNormalization::Population => (m2[idx] / (count as f64)).max(0.0),
            };
            variance.sqrt()
        };
    }

    Tensor::new(output, output_shape).map_err(|e| format!("std: {e}"))
}

fn resolve_axes(shape: &[usize], axes: &StdAxes) -> Result<(Vec<usize>, bool), String> {
    match axes {
        StdAxes::Default => {
            if shape.is_empty() {
                Ok((Vec::new(), true))
            } else {
                let dim = default_dimension_from_shape(shape);
                let zero = dim.saturating_sub(1);
                if zero < shape.len() {
                    Ok((vec![zero], true))
                } else {
                    Ok((Vec::new(), true))
                }
            }
        }
        StdAxes::Dim(dim) => {
            if *dim == 0 {
                return Err("std: dimension must be >= 1".to_string());
            }
            let zero = dim - 1;
            if zero < shape.len() {
                Ok((vec![zero], true))
            } else {
                Ok((Vec::new(), true))
            }
        }
        StdAxes::Vec(dims) => {
            if dims.is_empty() {
                return resolve_axes(shape, &StdAxes::Default);
            }
            let mut out = Vec::with_capacity(dims.len());
            for &dim in dims {
                if dim == 0 {
                    return Err("std: dimension must be >= 1".to_string());
                }
                let zero = dim - 1;
                if zero < shape.len() {
                    out.push(zero);
                }
            }
            out.sort_unstable();
            out.dedup();
            Ok((out, true))
        }
        StdAxes::All => {
            if shape.is_empty() {
                Ok((Vec::new(), true))
            } else {
                Ok(((0..shape.len()).collect(), true))
            }
        }
    }
}

fn reduced_shape(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut out = shape.to_vec();
    for &dim in dims {
        if dim < out.len() {
            out[dim] = 1;
        }
    }
    out
}

fn linear_to_multi(index: usize, shape: &[usize], out: &mut [usize]) {
    let mut remainder = index;
    for (dim, &size) in shape.iter().enumerate() {
        if size == 0 {
            out[dim] = 0;
        } else {
            out[dim] = remainder % size;
            remainder /= size;
        }
    }
}

fn multi_to_linear(coords: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut index = 0usize;
    for (&coord, &size) in coords.iter().zip(shape.iter()) {
        if size == 0 {
            continue;
        }
        index += coord * stride;
        stride *= size;
    }
    index
}

fn std_gpu(handle: GpuTensorHandle, args: &ParsedArguments) -> Result<Value, String> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(device_value) = std_gpu_reduce(provider, &handle, args) {
            return Ok(Value::GpuTensor(device_value));
        }
    }
    std_gpu_fallback(&handle, args)
}

fn std_gpu_reduce(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    args: &ParsedArguments,
) -> Option<GpuTensorHandle> {
    let (dims, _) = resolve_axes(&handle.shape, &args.axes).ok()?;
    let normalization = match args.normalization {
        StdNormalization::Sample => ProviderStdNormalization::Sample,
        StdNormalization::Population => ProviderStdNormalization::Population,
    };
    let nan_mode = match args.nan_mode {
        ReductionNaN::Include => ProviderNanMode::Include,
        ReductionNaN::Omit => ProviderNanMode::Omit,
    };

    if dims.is_empty() {
        return Some(handle.clone());
    }

    if dims.len() == handle.shape.len() {
        if handle.shape.is_empty() {
            return Some(handle.clone());
        }
        return provider
            .reduce_std(handle, normalization, nan_mode)
            .map_err(|err| {
                log::trace!("std: provider reduce_std fallback triggered: {err}");
                err
            })
            .ok();
    }

    if dims.len() == 1 {
        let dim = dims[0] + 1;
        return reduce_std_dim_gpu(provider, handle.clone(), dim, normalization, nan_mode);
    }

    None
}

fn reduce_std_dim_gpu(
    provider: &dyn AccelProvider,
    handle: GpuTensorHandle,
    dim: usize,
    normalization: ProviderStdNormalization,
    nan_mode: ProviderNanMode,
) -> Option<GpuTensorHandle> {
    if dim == 0 {
        return None;
    }
    if handle.shape.len() < dim {
        return Some(handle);
    }
    provider
        .reduce_std_dim(&handle, dim - 1, normalization, nan_mode)
        .map_err(|err| {
            log::trace!("std: provider reduce_std_dim fallback triggered: {err}");
            err
        })
        .ok()
}

fn std_gpu_fallback(handle: &GpuTensorHandle, args: &ParsedArguments) -> Result<Value, String> {
    let tensor = gpu_helpers::gather_tensor(handle)?;
    let reduced = std_tensor(tensor, &args.axes, args.normalization, args.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn apply_output_template(
    value: Value,
    template: &OutputTemplate,
    meta: &InputMeta,
) -> Result<Value, String> {
    match template {
        OutputTemplate::Double => Ok(value),
        OutputTemplate::Native => {
            let value = apply_native_template(value, meta)?;
            ensure_device(value, meta.device)
        }
        OutputTemplate::Like(proto) => apply_like_template(value, proto),
    }
}

fn apply_native_template(value: Value, meta: &InputMeta) -> Result<Value, String> {
    match meta.class {
        InputClass::Integer(class) => match value {
            Value::Num(n) => class.to_value(n),
            Value::Tensor(t) if t.data.len() == 1 => class.to_value(t.data[0]),
            other => Ok(other),
        },
        _ => {
            if let Some(dtype) = meta.numeric_dtype {
                coerce_value_to_dtype(value, dtype)
            } else {
                Ok(value)
            }
        }
    }
}

fn coerce_value_to_dtype(value: Value, dtype: NumericDType) -> Result<Value, String> {
    match dtype {
        NumericDType::F64 => Ok(value),
        NumericDType::F32 => match value {
            Value::Tensor(tensor) => {
                let tensor = coerce_tensor_dtype(tensor, NumericDType::F32);
                Ok(Value::Tensor(tensor))
            }
            Value::Num(n) => {
                let tensor = Tensor::new_with_dtype(vec![n], vec![1, 1], NumericDType::F32)
                    .map_err(|e| format!("{NAME}: {e}"))?;
                Ok(Value::Tensor(tensor))
            }
            Value::LogicalArray(logical) => {
                let tensor =
                    tensor::logical_to_tensor(&logical).map_err(|e| format!("{NAME}: {e}"))?;
                let tensor = coerce_tensor_dtype(tensor, NumericDType::F32);
                Ok(Value::Tensor(tensor))
            }
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor(&handle)?;
                let tensor = coerce_tensor_dtype(tensor, NumericDType::F32);
                Ok(Value::Tensor(tensor))
            }
            other => Ok(other),
        },
    }
}

fn coerce_tensor_dtype(mut tensor: Tensor, dtype: NumericDType) -> Tensor {
    match dtype {
        NumericDType::F64 => {
            tensor.dtype = NumericDType::F64;
        }
        NumericDType::F32 => {
            for value in &mut tensor.data {
                *value = (*value as f32) as f64;
            }
            tensor.dtype = NumericDType::F32;
        }
    }
    tensor
}

fn ensure_device(value: Value, device: DevicePreference) -> Result<Value, String> {
    match device {
        DevicePreference::Host => match value {
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor(&handle)?;
                Ok(tensor::tensor_into_value(tensor))
            }
            _ => Ok(value),
        },
        DevicePreference::Gpu => match value {
            Value::GpuTensor(_) => Ok(value),
            Value::Tensor(tensor) => upload_tensor(tensor),
            Value::Num(n) => {
                let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("std: {e}"))?;
                upload_tensor(tensor)
            }
            Value::LogicalArray(logical) => {
                let tensor = tensor::logical_to_tensor(&logical)?;
                upload_tensor(tensor)
            }
            other => Err(format!("std: cannot place value {other:?} on the GPU")),
        },
    }
}

fn upload_tensor(tensor: Tensor) -> Result<Value, String> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err("std: no acceleration provider available to honour GPU output".to_string());
    };
    let view = HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    let handle = provider
        .upload(&view)
        .map_err(|e| format!("std: failed to upload GPU result: {e}"))?;
    Ok(Value::GpuTensor(handle))
}

fn apply_like_template(value: Value, prototype: &Value) -> Result<Value, String> {
    let analysed = analyse_like_prototype(prototype)?;
    match analysed.class {
        PrototypeClass::Real => match analysed.device {
            DevicePreference::Host => ensure_device(value, DevicePreference::Host),
            DevicePreference::Gpu => ensure_device(value, DevicePreference::Gpu),
        },
        PrototypeClass::Complex => {
            let host_value = ensure_device(value, DevicePreference::Host)?;
            real_to_complex(host_value)
        }
    }
}

fn real_to_complex(value: Value) -> Result<Value, String> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(tensor) => {
            let data: Vec<(f64, f64)> = tensor.data.iter().map(|&v| (v, 0.0)).collect();
            let tensor =
                ComplexTensor::new(data, tensor.shape.clone()).map_err(|e| format!("std: {e}"))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            real_to_complex(Value::Tensor(tensor))
        }
        other => Err(format!(
            "std: cannot convert value {other:?} to a complex result"
        )),
    }
}

struct LikeAnalysis {
    device: DevicePreference,
    class: PrototypeClass,
}

enum PrototypeClass {
    Real,
    Complex,
}

fn analyse_like_prototype(proto: &Value) -> Result<LikeAnalysis, String> {
    match proto {
        Value::GpuTensor(_) => Ok(LikeAnalysis {
            device: DevicePreference::Gpu,
            class: PrototypeClass::Real,
        }),
        Value::Tensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::LogicalArray(_)
        | Value::Bool(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
            class: PrototypeClass::Real,
        }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
            class: PrototypeClass::Complex,
        }),
        other => {
            let gathered = dispatcher::gather_if_needed(other).map_err(|e| format!("std: {e}"))?;
            analyse_like_prototype(&gathered)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::IntValue;

    #[test]
    fn std_vector_sample_default() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let result = std_builtin(Value::Tensor(tensor), Vec::new()).expect("std");
        match result {
            Value::Num(v) => {
                let diff = (v - 1.58113883008).abs();
                assert!(diff < 1e-10, "value={v}, diff={diff}");
            }
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn std_population_columns() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let result =
            std_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(1))]).expect("std");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                for value in out.data {
                    let diff = (value - 0.5).abs();
                    assert!(diff < 1e-12, "value={value}, diff={diff}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn std_all_dimensions() {
        let tensor = Tensor::new((1..=12).map(|v| v as f64).collect(), vec![3, 4]).unwrap();
        let result = std_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("std");
        match result {
            Value::Num(v) => {
                let diff = (v - 3.60555127546).abs();
                assert!(diff < 1e-10, "value={v}, diff={diff}");
            }
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn std_vecdim_multiple_axes() {
        let tensor = Tensor::new(
            (1..=24).map(|v| v as f64).collect::<Vec<_>>(),
            vec![3, 4, 2],
        )
        .unwrap();
        let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let result = std_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("std");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 4, 1]);
                for value in out.data {
                    let diff = (value - 6.63324958071).abs();
                    assert!(diff < 1e-8, "value={value}, diff={diff}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn std_with_omit_nan_dimension_two() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0, 2.0, 4.0, f64::NAN], vec![3, 2]).unwrap();
        let result = std_builtin(
            Value::Tensor(tensor),
            vec![
                Value::Int(IntValue::I32(0)),
                Value::Int(IntValue::I32(2)),
                Value::from("omitnan"),
            ],
        )
        .expect("std");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [0.70710678119, 0.0, 0.0];
                for (value, exp) in out.data.iter().zip(expected.iter()) {
                    let diff = (value - exp).abs();
                    assert!(diff < 1e-9, "value={value}, expected={exp}, diff={diff}");
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn std_with_omit_nan_all_nan_slice() {
        let tensor = Tensor::new(vec![f64::NAN, f64::NAN], vec![2, 1]).unwrap();
        let result = std_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("std");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected NaN scalar, got {other:?}"),
        }
    }

    #[test]
    fn std_native_integer_scalar() {
        let value = Value::Int(IntValue::I16(42));
        let result = std_builtin(value, vec![Value::from("native")]).expect("std");
        assert_eq!(result, Value::Int(IntValue::I16(0)));
    }

    #[test]
    fn std_like_complex_prototype() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let proto = Value::Complex(0.0, 1.0);
        let result = std_builtin(
            Value::Tensor(tensor),
            vec![Value::from("like"), proto.clone()],
        )
        .expect("std");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 1.29099444874).abs() < 1e-10);
                assert_eq!(im, 0.0);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn std_dimension_greater_than_ndims_returns_input() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let original = tensor.clone();
        let result = std_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(0)), Value::Int(IntValue::I32(5))],
        )
        .expect("std");
        match result {
            Value::Tensor(out) => assert_eq!(out, original),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn std_invalid_weight_vector_reports_error() {
        let weights = Tensor::new(vec![0.2, 0.8], vec![1, 2]).unwrap();
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = std_builtin(Value::Tensor(tensor), vec![Value::Tensor(weights)]).unwrap_err();
        assert!(
            err.contains("std: dimension entries must be integers")
                || err.contains("std: dimension vector must not be empty"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn std_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = std_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("std");
            let gathered = test_support::gather(result).expect("gather");
            let host = std_builtin(Value::Tensor(tensor), Vec::new()).expect("std");
            match (host, gathered) {
                (Value::Num(expected), actual) => {
                    assert_eq!(actual.shape, vec![1, 1]);
                    assert!((actual.data[0] - expected).abs() < 1e-10);
                }
                _ => panic!("unexpected shapes"),
            }
            provider.free(&handle).ok();
        });
    }

    #[test]
    fn std_gpu_all_population_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new((1..=12).map(|v| v as f64).collect::<Vec<_>>(), vec![3, 4]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = std_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::Int(IntValue::I32(1)), Value::from("all")],
            )
            .expect("std");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            let value = gathered.data[0];
            let diff = (value - 3.45205252953).abs();
            assert!(diff < 1e-6, "value={value}, diff={diff}");
            provider.free(&handle).ok();
        });
    }

    #[test]
    fn std_gpu_omit_nan_falls_back() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = std_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::Int(IntValue::I32(0)), Value::from("omitnan")],
            )
            .expect("std");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            assert!((gathered.data[0] - 1.41421356237).abs() < 1e-8);
            provider.free(&handle).ok();
        });
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn std_wgpu_dim1_sample_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let args = ParsedArguments {
            axes: StdAxes::Dim(1),
            normalization: StdNormalization::Sample,
            nan_mode: ReductionNaN::Include,
            output: OutputTemplate::Double,
        };
        let cpu = std_host(Value::Tensor(tensor.clone()), &args).expect("cpu std");
        let cpu_tensor = test_support::gather(cpu).expect("gather cpu");

        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = std_gpu(handle.clone(), &args).expect("gpu std");
        let gpu_tensor = test_support::gather(gpu_value).expect("gather gpu");
        provider.free(&handle).ok();

        assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (gpu, cpu) in gpu_tensor.data.iter().zip(cpu_tensor.data.iter()) {
            assert!(
                (gpu - cpu).abs() < tol,
                "std mismatch: gpu={gpu} cpu={cpu} tol={tol}"
            );
        }
    }
}
