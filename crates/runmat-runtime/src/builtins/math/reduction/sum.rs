//! MATLAB-compatible `sum` builtin with GPU-aware semantics for RunMat.

use std::collections::HashSet;

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, IntValue, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "sum"
category: "math/reduction"
keywords: ["sum", "reduction", "omitnan", "gpu"]
summary: "Sum elements of scalars, vectors, matrices, or N-D tensors with MATLAB-compatible options."
references: []
gpu_support:
  elementwise: false
  reduction: true
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Prefers provider reduce_sum_dim / reduce_sum hooks; falls back to host for omitnan, multi-axis, or class-prototype requests."
fusion:
  elementwise: false
  reduction: true
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::reduction::sum::tests"
  integration: "builtins::math::reduction::sum::tests::sum_gpu_provider_roundtrip"
---

# What does the `sum` function do in MATLAB / RunMat?
`sum(X)` adds together elements of scalars, vectors, matrices, and higher-dimensional tensors.
When no dimension is supplied, the reduction runs along the first non-singleton dimension.

## How does the `sum` function behave in MATLAB / RunMat?
- `sum(X)` on an `m × n` matrix returns a row vector (`1 × n`) with column sums.
- `sum(X, 2)` returns a column vector (`m × 1`) containing row sums.
- `sum(X, dims)` accepts a vector of dimensions (e.g., `[1 3]`) and collapses each listed axis while leaving the others untouched.
- `sum(X, 'all')` flattens every dimension into a single scalar sum.
- Logical inputs are promoted to double precision (`true → 1.0`, `false → 0.0`).
- `sum(___, 'omitnan')` ignores `NaN` values; if every element in the slice is `NaN`, the result becomes `0`.
- `sum(___, 'includenan')` (default) propagates `NaN` when any element in the slice is `NaN`.
- `sum(___, outtype)` accepts `'double'`, `'default'`, or `'native'` to control the output class.
- `sum(___, 'like', prototype)` matches the numeric class and residency of `prototype` when supported by the active provider.
- Empty inputs or reductions along dimensions with size `0` return zeros that follow MATLAB shape semantics.

## `sum` Function GPU Execution Behaviour
RunMat Accelerate keeps tensors on the GPU whenever a provider is active:

1. If a tensor already resides on the device, the runtime calls the provider’s `reduce_sum_dim` (or `reduce_sum` for whole-array reductions). Successful hooks return a new GPU handle so downstream consumers stay on device.
2. Cases that require extra bookkeeping—such as `'omitnan'`, multi-axis reductions, or `'like'`/`'native'` class coercions—fall back to the host implementation. The runtime gathers the data, computes the correct MATLAB result, and re-uploads it only when a `'like'` prototype demands GPU residency.
3. The fusion planner keeps surrounding elementwise producers and consumers on the GPU, so manual `gpuArray` / `gather` calls are optional unless you want to force residency for interoperability with legacy MATLAB workflows.

## Examples of using the `sum` function in MATLAB / RunMat

### Summing the elements of a matrix

```matlab
A = [1 2 3; 4 5 6];
colSums = sum(A);
rowSums = sum(A, 2);
```

Expected output:

```matlab
colSums = [5 7 9];
rowSums = [6; 15];
```

### Summing across multiple dimensions

```matlab
B = reshape(1:24, [3 4 2]);
collapse = sum(B, [1 3]);
```

Expected output:

```matlab
collapse = [48 66 84 102];
```

### Summing with NaN values ignored

```matlab
values = [1 NaN 3];
total = sum(values, 'omitnan');
```

Expected output:

```matlab
total = 4;
```

### Summing on the GPU and matching an existing prototype

```matlab
proto = gpuArray.zeros(1, 1, 'single');
result = sum(gpuArray([1 2 3]), 'all', 'like', proto);
```

Expected output:

```matlab
result =
  1x1 gpuArray  single
     6
```

### Summing all elements of an array into a scalar

```matlab
C = [1 2 3; 4 5 6];
grandTotal = sum(C, 'all');
```

Expected output:

```matlab
grandTotal = 21;
```

### Summing with native output type

```matlab
ints = int32([100 200 300]);
total = sum(ints, 'native');
```

Expected output:

```matlab
total = int32(600);
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` yourself. The fusion planner keeps tensors on the GPU whenever the provider exposes the required kernels. To mirror MATLAB, RunMat still accepts and respects explicit `gpuArray` / `gather` usage and the `'like'` option to control residency explicitly.

## FAQ

### When should I use the `sum` function?
Use `sum` whenever you need to add together slices of a tensor, whether across a single dimension, multiple dimensions, or the entire dataset.

### Does `sum` produce double arrays by default?
Yes. Unless you request `'native'` or provide a `'like'` prototype, the result is a dense double-precision array on the host.

### What does `sum(A)` return?
For arrays, `sum(A)` reduces along the first non-singleton dimension, returning a new array whose reduced axis has size `1`. Scalars are returned unchanged.

### How do I compute the sum of a specific dimension?
Provide the dimension index: `sum(A, 2)` sums rows, `sum(A, 3)` sums along the third dimension, and so on. You can also pass a vector to collapse multiple dimensions.

### What happens if all elements are `NaN` and I request `'omitnan'`?
`sum(..., 'omitnan')` treats `NaN` values as missing data. If every element in the slice is `NaN`, the result becomes `0`, matching MATLAB semantics.

### Does `sum` preserve integer classes?
Only when you explicitly request `'native'` or `'like'`. Otherwise integers are promoted to double precision so you do not have to manage overflow manually.

## See Also
[prod](./prod), [mean](./mean), [cumsum](./cumsum), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `sum` function is available at: [`crates/runmat-runtime/src/builtins/math/reduction/sum.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/sum.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sum",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Reduction {
        name: "reduce_sum_dim",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
    notes:
        "Providers may specialise reduce_sum_dim / reduce_sum; omitnan and multi-axis reductions fall back to the CPU path when unsupported.",
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
    notes: "Planner emits a standard column-major reduction template; providers can substitute custom kernels.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("sum", DOC_MD);

#[runtime_builtin(
    name = "sum",
    category = "math/reduction",
    summary = "Sum elements of scalars, vectors, matrices, or N-D tensors.",
    keywords = "sum,reduction,gpu,omitnan,all,like",
    accel = "reduction"
)]
fn sum_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let input_meta = InputMeta::from_value(&value);
    let parsed = parse_arguments(&rest)?;
    let raw_result = match value {
        Value::GpuTensor(handle) => sum_gpu(handle, &parsed)?,
        Value::ComplexTensor(ct) => sum_host_complex_tensor(ct, &parsed)?,
        Value::Complex(re, im) => sum_host_complex_scalar(re, im, &parsed)?,
        other => sum_host(other, &parsed)?,
    };
    apply_output_template(raw_result, &parsed.output, &input_meta)
}

#[derive(Clone, PartialEq, Eq)]
enum DimSelection {
    Auto,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}

#[derive(Clone)]
struct ResolvedDims {
    dims_in_bounds: Vec<usize>,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    Native,
    Like(Value),
}

struct ParsedArguments {
    selection: DimSelection,
    nan_mode: ReductionNaN,
    output: OutputTemplate,
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
        Self { class, device }
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
            return Err("sum: cannot represent NaN as an integer output".to_string());
        }
        let rounded = scalar.round();
        if !rounded.is_finite() {
            return Err("sum: integer output overflowed the target type".to_string());
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

fn parse_arguments(args: &[Value]) -> Result<ParsedArguments, String> {
    let mut selection = DimSelection::Auto;
    let mut selection_set = false;
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
                "all" => {
                    if selection_set && !matches!(selection, DimSelection::Auto) {
                        return Err(
                            "sum: 'all' cannot be combined with an explicit dimension".to_string()
                        );
                    }
                    selection = DimSelection::All;
                    selection_set = true;
                    idx += 1;
                    continue;
                }
                "double" | "default" => {
                    if output_set {
                        return Err(
                            "sum: multiple output class specifications provided".to_string()
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
                            "sum: multiple output class specifications provided".to_string()
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
                            "sum: cannot combine 'like' with another output class specifier"
                                .to_string(),
                        );
                    }
                    let Some(proto) = args.get(idx + 1).cloned() else {
                        return Err("sum: expected prototype after 'like'".to_string());
                    };
                    output = OutputTemplate::Like(proto);
                    idx += 2;
                    if idx < args.len() {
                        return Err("sum: 'like' must be the final argument".to_string());
                    }
                    break;
                }
                _ => {}
            }
        }

        if !selection_set || matches!(selection, DimSelection::Auto) {
            match parse_dimension_spec(arg)? {
                Some(sel) => {
                    selection = sel;
                    selection_set = true;
                    idx += 1;
                    continue;
                }
                None => {}
            }
        }

        return Err(format!("sum: unrecognised argument {arg:?}"));
    }

    Ok(ParsedArguments {
        selection,
        nan_mode,
        output,
    })
}

fn parse_dimension_spec(value: &Value) -> Result<Option<DimSelection>, String> {
    match value {
        Value::Int(i) => {
            let dim = i.to_i64();
            if dim < 1 {
                return Err("sum: dimension must be >= 1".to_string());
            }
            Ok(Some(DimSelection::Dim(dim as usize)))
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err("sum: dimension must be finite".to_string());
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err("sum: dimension must be an integer".to_string());
            }
            if rounded < 1.0 {
                return Err("sum: dimension must be >= 1".to_string());
            }
            Ok(Some(DimSelection::Dim(rounded as usize)))
        }
        Value::Tensor(t) => parse_dimension_tensor(t),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(logical)?;
            parse_dimension_tensor(&tensor)
        }
        Value::GpuTensor(_) => Err("sum: dimension arguments must reside on the host".to_string()),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Ok(None),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(None),
        _ => Ok(None),
    }
}

fn parse_dimension_tensor(tensor: &Tensor) -> Result<Option<DimSelection>, String> {
    if tensor.data.is_empty() {
        return Ok(Some(DimSelection::Auto));
    }
    if !is_vector_shape(&tensor.shape) {
        return Err("sum: dimension vector must be a row or column vector".to_string());
    }
    let mut dims = Vec::with_capacity(tensor.data.len());
    for &v in &tensor.data {
        if !v.is_finite() {
            return Err("sum: dimensions must be finite".to_string());
        }
        let rounded = v.round();
        if (rounded - v).abs() > f64::EPSILON {
            return Err("sum: dimensions must contain integers".to_string());
        }
        if rounded < 1.0 {
            return Err("sum: dimension indices must be >= 1".to_string());
        }
        dims.push(rounded as usize);
    }
    if dims.is_empty() {
        Ok(Some(DimSelection::Auto))
    } else {
        Ok(Some(DimSelection::Vec(dims)))
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape.len() {
        0 => true,
        1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => shape.iter().filter(|&&d| d > 1).count() <= 1,
    }
}

fn sum_host(value: Value, parsed: &ParsedArguments) -> Result<Value, String> {
    let tensor = tensor::value_into_tensor(value)?;
    let resolved = resolve_dims(&tensor.shape, &parsed.selection)?;
    let reduced = sum_tensor(&tensor, &resolved, parsed.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn sum_host_complex_tensor(ct: ComplexTensor, parsed: &ParsedArguments) -> Result<Value, String> {
    let resolved = resolve_dims(&ct.shape, &parsed.selection)?;
    let reduced = sum_complex_tensor(&ct, &resolved, parsed.nan_mode)?;
    Ok(complex_tensor_into_value(reduced))
}

fn sum_host_complex_scalar(re: f64, im: f64, parsed: &ParsedArguments) -> Result<Value, String> {
    let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| format!("sum: {e}"))?;
    sum_host_complex_tensor(tensor, parsed)
}

fn sum_gpu(handle: GpuTensorHandle, parsed: &ParsedArguments) -> Result<Value, String> {
    if matches!(parsed.nan_mode, ReductionNaN::Omit) {
        return sum_gpu_with_omitnan(handle, parsed);
    }

    let Some(provider) = runmat_accelerate_api::provider() else {
        return sum_gpu_fallback(&handle, parsed);
    };

    let resolved = resolve_dims(&handle.shape, &parsed.selection)?;
    if resolved.dims_in_bounds.is_empty() {
        return Ok(Value::GpuTensor(handle));
    }

    if resolved.dims_in_bounds.len() == handle.shape.len() && !handle.shape.is_empty() {
        if let Ok(reduced) = provider.reduce_sum(&handle) {
            return Ok(Value::GpuTensor(reduced));
        }
    }

    let mut current = handle.clone();
    for &dim in &resolved.dims_in_bounds {
        match provider.reduce_sum_dim(&current, dim) {
            Ok(next) => {
                current = next;
            }
            Err(_) => return sum_gpu_fallback(&handle, parsed),
        }
    }
    Ok(Value::GpuTensor(current))
}

fn sum_gpu_with_omitnan(
    handle: GpuTensorHandle,
    parsed: &ParsedArguments,
) -> Result<Value, String> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return sum_gpu_fallback(&handle, parsed);
    };
    let resolved = resolve_dims(&handle.shape, &parsed.selection)?;
    if resolved.dims_in_bounds.len() != 1 || handle.shape.len() != 2 {
        return sum_gpu_fallback(&handle, parsed);
    }
    let axis = resolved.dims_in_bounds[0];
    let expected_shape = reduced_shape(&handle.shape, &[axis]);
    if expected_shape.is_empty() {
        return sum_gpu_fallback(&handle, parsed);
    }
    if let Some(result) = reduce_omitnan_2d(provider, &handle, axis, &expected_shape) {
        Ok(Value::GpuTensor(result))
    } else {
        sum_gpu_fallback(&handle, parsed)
    }
}

fn reduce_omitnan_2d(
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    axis: usize,
    expected_shape: &[usize],
) -> Option<GpuTensorHandle> {
    let rows = *handle.shape.get(0)?;
    let cols = *handle.shape.get(1)?;
    let (reduce_len, num_slices, axis_is_row) = if axis == 0 {
        (rows, cols, false)
    } else if axis == 1 {
        (cols, rows, true)
    } else {
        return None;
    };
    let scalar_ty = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => "f32",
        runmat_accelerate_api::ProviderPrecision::F64 => "f64",
    };
    let shader = build_omitnan_shader(scalar_ty, axis_is_row);
    provider
        .fused_reduction(
            &shader,
            std::slice::from_ref(handle),
            expected_shape,
            reduce_len,
            num_slices,
            256,
        )
        .ok()
}

fn build_omitnan_shader(scalar_ty: &str, axis_is_row: bool) -> String {
    let mut shader = String::new();
    shader.push_str(&format!("struct Tensor {{ data: array<{scalar_ty}>; }}\n"));
    shader.push_str("struct MParams { nrows: u32, ncols: u32, ld: u32, flags: u32 }\n\n");
    shader.push_str("@group(0) @binding(0) var<storage, read> input0: Tensor;\n");
    shader.push_str("@group(0) @binding(1) var<storage, read_write> output: Tensor;\n");
    shader.push_str("@group(0) @binding(2) var<uniform> params: MParams;\n\n");
    shader.push_str("var<workgroup> tile: array<f32, 256u>;\n");
    shader.push_str("fn toBoolNan(x: f32) -> bool { return x != x; }\n\n");
    shader.push_str("@compute @workgroup_size(256)\n");
    if axis_is_row {
        shader.push_str(
            "fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {\n",
        );
        shader.push_str("  let row = wid.x; if (row >= params.nrows) { return; }\n");
        shader.push_str(&format!(
            "  var acc: {scalar_ty} = {}0.0;\n",
            if scalar_ty == "f64" { "f64(" } else { "" }
        ));
        if scalar_ty == "f64" {
            shader.push_str("  // close literal\n");
        }
        shader.push_str(
            "  var c = lid.x;\n  while (c < params.ncols) {\n    let v = input0.data[row + (c * params.ld)];\n    if (!toBoolNan(v)) { acc = acc + v; }\n    c += 256u;\n  }\n",
        );
        shader.push_str("  tile[lid.x] = acc;\n  workgroupBarrier();\n");
    } else {
        shader.push_str(
            "fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {\n",
        );
        shader.push_str("  let col = wid.x; if (col >= params.ncols) { return; }\n");
        shader.push_str(&format!(
            "  var acc: {scalar_ty} = {}0.0;\n",
            if scalar_ty == "f64" { "f64(" } else { "" }
        ));
        if scalar_ty == "f64" {
            shader.push_str("  // close literal\n");
        }
        shader.push_str(
            "  var r = lid.x;\n  while (r < params.nrows) {\n    let v = input0.data[(col * params.ld) + r];\n    if (!toBoolNan(v)) { acc = acc + v; }\n    r += 256u;\n  }\n",
        );
        shader.push_str("  tile[lid.x] = acc;\n  workgroupBarrier();\n");
    }
    shader.push_str(
        "  var off = 128u;\n  loop {\n    if (off == 0u) { break; }\n    if (lid.x < off) {\n      let a = tile[lid.x];\n      let b = tile[lid.x + off];\n      tile[lid.x] = a + b;\n    }\n    workgroupBarrier();\n    off = off / 2u;\n  }\n",
    );
    if axis_is_row {
        shader.push_str("  if (lid.x == 0u) { output.data[row] = tile[0u]; }\n}\n");
    } else {
        shader.push_str("  if (lid.x == 0u) { output.data[col] = tile[0u]; }\n}\n");
    }
    shader
}

fn sum_gpu_fallback(handle: &GpuTensorHandle, parsed: &ParsedArguments) -> Result<Value, String> {
    let tensor = gpu_helpers::gather_tensor(handle)?;
    let resolved = resolve_dims(&tensor.shape, &parsed.selection)?;
    let reduced = sum_tensor(&tensor, &resolved, parsed.nan_mode)?;
    Ok(tensor::tensor_into_value(reduced))
}

fn resolve_dims(shape: &[usize], selection: &DimSelection) -> Result<ResolvedDims, String> {
    let dims_1_based: Vec<usize> = match selection {
        DimSelection::Auto => vec![default_dimension_from_shape(shape)],
        DimSelection::Dim(d) => vec![*d],
        DimSelection::Vec(v) => {
            if v.is_empty() {
                vec![default_dimension_from_shape(shape)]
            } else {
                v.clone()
            }
        }
        DimSelection::All => {
            let ndims = if shape.is_empty() { 1 } else { shape.len() };
            (1..=ndims).collect()
        }
    };

    let mut seen = HashSet::new();
    let mut dims_in_bounds = Vec::new();
    let ndims = shape.len();

    for dim1 in dims_1_based {
        if dim1 == 0 {
            return Err("sum: dimension indices must be >= 1".to_string());
        }
        if !seen.insert(dim1) {
            continue;
        }
        let zero = dim1 - 1;
        if zero < ndims {
            dims_in_bounds.push(zero);
        }
    }

    dims_in_bounds.sort_unstable();

    Ok(ResolvedDims { dims_in_bounds })
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

fn sum_tensor(
    tensor: &Tensor,
    dims: &ResolvedDims,
    nan_mode: ReductionNaN,
) -> Result<Tensor, String> {
    let mut shape = tensor.shape.clone();
    if shape.is_empty() {
        shape = vec![tensor.rows, tensor.cols];
    }

    if dims.dims_in_bounds.is_empty() {
        return Ok(tensor.clone());
    }

    let mut output_shape = shape.clone();
    for &dim in &dims.dims_in_bounds {
        if dim < output_shape.len() {
            output_shape[dim] = 1;
        }
    }

    let out_len = tensor::element_count(&output_shape);
    let mut sums = vec![0.0f64; out_len];
    let mut saw_value = vec![false; out_len];
    let mut saw_nan = vec![false; out_len];
    let mut coords = vec![0usize; shape.len()];
    let mut out_coords = vec![0usize; shape.len()];
    let mut reduce_mask = vec![false; shape.len()];
    for &dim in &dims.dims_in_bounds {
        if dim < reduce_mask.len() {
            reduce_mask[dim] = true;
        }
    }

    for (linear, &value) in tensor.data.iter().enumerate() {
        linear_to_multi(linear, &shape, &mut coords);
        for (i, coord) in coords.iter().enumerate() {
            out_coords[i] = if reduce_mask[i] { 0 } else { *coord };
        }
        let out_idx = multi_to_linear(&out_coords, &output_shape);
        match nan_mode {
            ReductionNaN::Include => {
                if value.is_nan() {
                    saw_nan[out_idx] = true;
                } else {
                    sums[out_idx] += value;
                    saw_value[out_idx] = true;
                }
            }
            ReductionNaN::Omit => {
                if !value.is_nan() {
                    sums[out_idx] += value;
                    saw_value[out_idx] = true;
                }
            }
        }
    }

    let mut output = Vec::with_capacity(out_len);
    for idx in 0..out_len {
        let result = match nan_mode {
            ReductionNaN::Include => {
                if saw_nan[idx] {
                    f64::NAN
                } else if saw_value[idx] {
                    sums[idx]
                } else {
                    0.0
                }
            }
            ReductionNaN::Omit => {
                if saw_value[idx] {
                    sums[idx]
                } else {
                    0.0
                }
            }
        };
        output.push(result);
    }

    Tensor::new(output, output_shape).map_err(|e| format!("sum: {e}"))
}

fn sum_complex_tensor(
    tensor: &ComplexTensor,
    dims: &ResolvedDims,
    nan_mode: ReductionNaN,
) -> Result<ComplexTensor, String> {
    let mut shape = tensor.shape.clone();
    if shape.is_empty() {
        shape = vec![tensor.rows, tensor.cols];
    }

    if dims.dims_in_bounds.is_empty() {
        return Ok(tensor.clone());
    }

    let mut output_shape = shape.clone();
    for &dim in &dims.dims_in_bounds {
        if dim < output_shape.len() {
            output_shape[dim] = 1;
        }
    }

    let out_len = tensor::element_count(&output_shape);
    let mut sums = vec![(0.0f64, 0.0f64); out_len];
    let mut saw_value = vec![false; out_len];
    let mut saw_nan = vec![false; out_len];
    let mut coords = vec![0usize; shape.len()];
    let mut out_coords = vec![0usize; shape.len()];
    let mut reduce_mask = vec![false; shape.len()];
    for &dim in &dims.dims_in_bounds {
        if dim < reduce_mask.len() {
            reduce_mask[dim] = true;
        }
    }

    for (linear, &(re, im)) in tensor.data.iter().enumerate() {
        linear_to_multi(linear, &shape, &mut coords);
        for (i, coord) in coords.iter().enumerate() {
            out_coords[i] = if reduce_mask[i] { 0 } else { *coord };
        }
        let out_idx = multi_to_linear(&out_coords, &output_shape);
        let is_nan = re.is_nan() || im.is_nan();
        match nan_mode {
            ReductionNaN::Include => {
                if is_nan {
                    saw_nan[out_idx] = true;
                } else {
                    sums[out_idx].0 += re;
                    sums[out_idx].1 += im;
                    saw_value[out_idx] = true;
                }
            }
            ReductionNaN::Omit => {
                if !is_nan {
                    sums[out_idx].0 += re;
                    sums[out_idx].1 += im;
                    saw_value[out_idx] = true;
                }
            }
        }
    }

    let mut output = Vec::with_capacity(out_len);
    for idx in 0..out_len {
        let (re, im) = sums[idx];
        let result = match nan_mode {
            ReductionNaN::Include => {
                if saw_nan[idx] {
                    (f64::NAN, f64::NAN)
                } else if saw_value[idx] {
                    (re, im)
                } else {
                    (0.0, 0.0)
                }
            }
            ReductionNaN::Omit => {
                if saw_value[idx] {
                    (re, im)
                } else {
                    (0.0, 0.0)
                }
            }
        };
        output.push(result);
    }

    ComplexTensor::new(output, output_shape).map_err(|e| format!("sum: {e}"))
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
    for (dim, &size) in shape.iter().enumerate() {
        if size == 0 {
            continue;
        }
        index += coords[dim] * stride;
        stride *= size;
    }
    index
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
        _ => Ok(value),
    }
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
            Value::Tensor(t) => upload_tensor(t),
            Value::Num(n) => {
                let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("sum: {e}"))?;
                upload_tensor(tensor)
            }
            Value::LogicalArray(logical) => {
                let tensor = tensor::logical_to_tensor(&logical)?;
                upload_tensor(tensor)
            }
            other => Err(format!("sum: cannot place value {other:?} on the GPU")),
        },
    }
}

fn upload_tensor(tensor: Tensor) -> Result<Value, String> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Err("sum: no acceleration provider available to honour GPU output".to_string());
    };
    let view = HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    let handle = provider
        .upload(&view)
        .map_err(|e| format!("sum: failed to upload GPU result: {e}"))?;
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
        Value::Tensor(t) => {
            let data: Vec<(f64, f64)> = t.data.iter().map(|&v| (v, 0.0)).collect();
            let tensor =
                ComplexTensor::new(data, t.shape.clone()).map_err(|e| format!("sum: {e}"))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            real_to_complex(Value::Tensor(tensor))
        }
        other => Err(format!(
            "sum: cannot convert value {other:?} to a complex result"
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
            let gathered =
                crate::dispatcher::gather_if_needed(other).map_err(|e| format!("sum: {e}"))?;
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
    fn sum_scalar_num() {
        let result = sum_builtin(Value::Num(5.0), Vec::new()).expect("sum");
        assert_eq!(result, Value::Num(5.0));
    }

    #[test]
    fn sum_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), Vec::new()).expect("sum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.data, vec![5.0, 7.0, 9.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sum_matrix_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result =
            sum_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))]).expect("sum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![6.0, 15.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sum_all_dimension() {
        let tensor =
            Tensor::new((1..=6).map(|v| v as f64).collect::<Vec<_>>(), vec![2, 3]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("sum");
        assert_eq!(result, Value::Num(21.0));
    }

    #[test]
    fn sum_vecdim_multiple_axes() {
        let tensor = Tensor::new(
            (1..=24).map(|v| v as f64).collect::<Vec<_>>(),
            vec![3, 4, 2],
        )
        .unwrap();
        let dims = Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("sum");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 4, 1]);
                assert_eq!(out.data, vec![48.0, 66.0, 84.0, 102.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sum_with_omit_nan_default_dimension() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("sum");
        assert_eq!(result, Value::Num(4.0));
    }

    #[test]
    fn sum_with_include_nan_propagates() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let result = sum_builtin(Value::Tensor(tensor), Vec::new()).expect("sum");
        match result {
            Value::Num(n) => assert!(n.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[test]
    fn sum_dimension_greater_than_ndims_returns_input() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let original = tensor.clone();
        let value = Value::Tensor(tensor);
        let result = sum_builtin(value, vec![Value::Int(IntValue::I32(5))]).expect("sum");
        match result {
            Value::Tensor(out) => assert_eq!(out, original),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sum_native_integer_scalar() {
        let value = Value::Int(IntValue::I16(42));
        let result = sum_builtin(value, vec![Value::from("native")]).expect("sum");
        assert_eq!(result, Value::Int(IntValue::I16(42)));
    }

    #[test]
    fn sum_like_complex_prototype() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let proto = Value::Complex(0.0, 1.0);
        let result = sum_builtin(
            Value::Tensor(tensor),
            vec![Value::from("all"), Value::from("like"), proto.clone()],
        )
        .expect("sum");
        match result {
            Value::Complex(re, im) => {
                assert_eq!(re, 6.0);
                assert_eq!(im, 0.0);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn sum_like_without_prototype_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = sum_builtin(Value::Tensor(tensor), vec![Value::from("like")])
            .expect_err("expected error");
        assert!(err.contains("prototype"));
    }

    #[test]
    fn sum_all_omitnan() {
        let tensor = Tensor::new(vec![2.0, f64::NAN, 10.0, 4.0], vec![2, 2]).unwrap();
        let result = sum_builtin(
            Value::Tensor(tensor),
            vec![Value::from("all"), Value::from("omitnan")],
        )
        .expect("sum");
        assert_eq!(result, Value::Num(16.0));
    }

    #[test]
    fn sum_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = sum_builtin(Value::GpuTensor(handle), Vec::new()).expect("sum");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(gathered.data, vec![5.0, 7.0, 9.0]);
        });
    }

    #[test]
    fn sum_gpu_all_reduction() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new((1..=8).map(|v| v as f64).collect::<Vec<_>>(), vec![2, 4]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                sum_builtin(Value::GpuTensor(handle), vec![Value::from("all")]).expect("sum");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            assert_eq!(gathered.data, vec![36.0]);
        });
    }

    #[test]
    fn sum_gpu_like_prototype() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto_handle = provider.upload(&proto_view).expect("upload");
            let result = sum_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto_handle.clone())],
            )
            .expect("sum");
            match result {
                Value::GpuTensor(h) => {
                    let gathered = test_support::gather(Value::GpuTensor(h)).expect("gather");
                    assert_eq!(gathered.shape, vec![1, 2]);
                    assert_eq!(gathered.data, vec![3.0, 7.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
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
    fn sum_wgpu_dim1_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let t = Tensor::new(vec![1.0, 4.0, 2.0, 6.0], vec![2, 2]).unwrap();
        let cpu = sum_host(
            Value::Tensor(t.clone()),
            &ParsedArguments {
                selection: DimSelection::Dim(1),
                nan_mode: ReductionNaN::Include,
                output: OutputTemplate::Double,
            },
        )
        .unwrap();
        let view = HostTensorView {
            data: &t.data,
            shape: &t.shape,
        };
        let h = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = sum_gpu(
            h,
            &ParsedArguments {
                selection: DimSelection::Dim(1),
                nan_mode: ReductionNaN::Include,
                output: OutputTemplate::Double,
            },
        )
        .unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match cpu {
            Value::Tensor(ct) => {
                assert_eq!(gathered.shape, ct.shape);
                assert_eq!(gathered.data, ct.data);
            }
            other => panic!("unexpected shapes {other:?}"),
        }
    }
}
