//! MATLAB-compatible `circshift` builtin with GPU-aware semantics for RunMat.
//!
//! This module implements the `circshift` function, matching MathWorks MATLAB
//! behaviour for rotating tensors, logical masks, complex arrays, string
//! arrays, and character matrices. When an acceleration provider exposes a
//! native `circshift` hook the runtime keeps data on the GPU; otherwise it
//! gathers the tensor once, performs the rotation on the host, and re-uploads
//! the result so downstream operations continue to benefit from gpu residency.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;
use std::collections::HashSet;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "circshift",
        wasm_path = "crate::builtins::array::shape::circshift"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "circshift"
category: "array/shape"
keywords: ["circshift", "circular shift", "rotate array", "gpu", "cyclic shift"]
summary: "Rotate arrays circularly along one or more dimensions."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64", "i32", "bool"]
  broadcasting: "none"
  notes: "Uses the provider circshift hook when available; otherwise gathers once, rotates on the host, and re-uploads."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::shape::circshift::tests"
  integration: "builtins::array::shape::circshift::tests::circshift_gpu_roundtrip"
---

# What does the `circshift` function do in MATLAB / RunMat?
`circshift(A, K)` rotates the elements of `A` circularly. Positive shifts move
elements toward higher indices, wrapping values that fall off the end back to
the beginning. Shifts can be specified per-dimension using vectors or paired
with explicit dimension lists.

## How does the `circshift` function behave in MATLAB / RunMat?
- `circshift(A, k)` shifts by `k` positions along the first non-singleton
  dimension. Negative values shift in the opposite direction.
- `circshift(A, shiftVec)` accepts a numeric vector that supplies a shift per
  dimension. Missing dimensions default to zero. Extra dimensions are treated
  as size-one axes, so they have no effect unless an explicit reshape added
  those dimensions previously.
- `circshift(A, K, dims)` lets you target specific dimensions. `K` and `dims`
  must have the same length; each entry in `dims` is 1-based.
- Works for numeric tensors, logical masks, complex arrays, string arrays, and
  character matrices. Character arrays only support dimensions one and two,
  mirroring MATLAB limitations.
- Empty dimensions remain empty; shifting them is a no-op. Scalars are returned
  unchanged.
- Shifts that are integer multiples of a dimension’s extent leave that axis
  unchanged. RunMat reduces each shift modulo the axis length, matching MATLAB.

## `circshift` Function GPU Execution Behaviour
When RunMat Accelerate is active, the runtime first asks the selected provider
for a native `circshift` implementation. Providers that implement the hook
perform the rotation entirely on the device, preserving residency metadata and
enabling fusion with subsequent GPU kernels. If the hook is missing, RunMat
downloads the tensor once, rotates it on the host using the same semantics, and
uploads the result back to the GPU so downstream work continues without manual
intervention.

## Examples of using the `circshift` function in MATLAB / RunMat

### Shifting matrix rows downward by one location
```matlab
A = [1 2 3; 4 5 6; 7 8 9];
B = circshift(A, 1);
```
Expected output:
```matlab
B =
     7     8     9
     1     2     3
     4     5     6
```

### Shifting columns left by two positions
```matlab
A = reshape(1:12, [3 4]);
C = circshift(A, [0 -2]);
```
Expected output:
```matlab
C =
     7    10     1     4
     8    11     2     5
     9    12     3     6
```

### Rotating a 3-D tensor along multiple dimensions
```matlab
T = reshape(1:8, [2 2 2]);
U = circshift(T, [1 0 -1]);
U(:, :, 1)
U(:, :, 2)
```
Expected output:
```matlab
ans(:,:,1) =
     6     8
     5     7

ans(:,:,2) =
     2     4
     1     3
```

### Shifting along specific dimensions using the `dims` argument
```matlab
A = reshape(1:12, [3 4]);
V = circshift(A, [2 -1], [1 2]);
```
Expected output:
```matlab
V =
     6     7     8     5
     9    10    11    12
     3     4     1     2
```

### Rotating a character matrix to cycle through rows and columns
```matlab
C = ['r','u','n'; 'm','a','t'];
R = circshift(C, [0 1]);
```
Expected output:
```matlab
R =
    'nru'
    'tma'
```

### Applying `circshift` to a gpuArray and keeping the result on the device
```matlab
G = gpuArray(reshape(1:12, [3 4]));
H = circshift(G, [1 -2]);
isgpuarray(H)
```
Expected output:
```matlab
ans = logical 1
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No additional steps are required. RunMat’s auto-offload planner keeps tensors on
the GPU whenever the provider exposes native support. If a provider falls back
to the host implementation, the runtime gathers and re-uploads the data
transparently so subsequent operations can continue on the GPU. Explicit calls
to `gpuArray` remain available for MATLAB compatibility but are not mandatory.

## FAQ

### Do `K` and `dims` need the same length?
Yes. When you supply the `dims` argument, the shift vector must have the same number of entries. Each shift is paired with the corresponding dimension.

### Can I shift along dimensions that are currently size one?
You can, and the result matches MATLAB: the axis length is one so the shift has no visible effect.

### How are negative shifts handled?
They rotate values toward lower indices. Internally RunMat reduces each shift modulo the axis length, so `-1` on a length-4 dimension is equivalent to `3`.

### What happens with empty tensors?
The result remains empty with identical shape metadata. `circshift` never introduces new elements.

### Are logical and string arrays supported?
Yes. Logical arrays stay logical, and string arrays preserve their individual strings while rotating their positions.

### Can character arrays be shifted along the third dimension?
No. MATLAB character arrays are strictly two-dimensional; RunMat matches this restriction and raises an error if you request dimension three or higher.

### Does `circshift` fuse with other GPU kernels?
When a provider supplies the hook, yes. Otherwise the rotation acts as a residency boundary in fused graphs.

### How does `circshift` interact with complex numbers?
Real and imaginary parts move together. Scalars are unaffected, and arrays retain their complex entries without modification.

### What if I pass non-integer shifts?
RunMat enforces MATLAB semantics: shifts must be finite integers. Fractional values raise an error.

### Does the builtin allocate new GPU buffers?
Providers may reuse buffers internally, but from the user’s perspective the result is a fresh gpuArray handle that preserves residency information.

## See Also
[`permute`](./permute), [`rot90`](./rot90), [`flip`](./flip),
[`gpuArray`](../../acceleration/gpu/gpuArray), [`gather`](../../acceleration/gpu/gather)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/array/shape/circshift.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/shape/circshift.rs)
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::array::shape::circshift")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "circshift",
    op_kind: GpuOpKind::Custom("circshift"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("circshift")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may implement a dedicated circshift hook; otherwise the runtime gathers, rotates, and re-uploads once.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::array::shape::circshift")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "circshift",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Circshift reorders data; fusion planners treat it as a residency boundary between kernels.",
};

#[runtime_builtin(
    name = "circshift",
    category = "array/shape",
    summary = "Rotate arrays circularly along one or more dimensions.",
    keywords = "circshift,circular shift,rotate array,gpu,cyclic shift",
    accel = "custom",
    wasm_path = "crate::builtins::array::shape::circshift"
)]
fn circshift_builtin(value: Value, shift: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.len() > 1 {
        return Err("circshift: too many input arguments".to_string());
    }
    let spec = parse_circshift_spec(&shift, &rest)?;
    let dims = &spec.dims;
    let shifts = &spec.shifts;

    match value {
        Value::Tensor(tensor) => {
            circshift_tensor(tensor, dims, shifts).map(tensor::tensor_into_value)
        }
        Value::LogicalArray(array) => {
            circshift_logical_array(array, dims, shifts).map(Value::LogicalArray)
        }
        Value::ComplexTensor(ct) => {
            circshift_complex_tensor(ct, dims, shifts).map(Value::ComplexTensor)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| format!("circshift: {e}"))?;
            circshift_complex_tensor(tensor, dims, shifts).map(complex_tensor_into_value)
        }
        Value::StringArray(strings) => {
            circshift_string_array(strings, dims, shifts).map(Value::StringArray)
        }
        Value::CharArray(chars) => circshift_char_array(chars, dims, shifts),
        Value::String(scalar) => Ok(Value::String(scalar)),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor = tensor::value_into_tensor_for("circshift", value)?;
            circshift_tensor(tensor, dims, shifts).map(tensor::tensor_into_value)
        }
        Value::GpuTensor(handle) => circshift_gpu(handle, dims, shifts),
        Value::Cell(_) => Err("circshift: cell arrays are not yet supported".to_string()),
        Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err("circshift: unsupported input type".to_string()),
    }
}

#[derive(Debug, Clone)]
struct CircshiftSpec {
    dims: Vec<usize>,
    shifts: Vec<isize>,
}

fn parse_circshift_spec(shift: &Value, rest: &[Value]) -> Result<CircshiftSpec, String> {
    let shifts = value_to_shift_vector(shift)?;
    let dims: Vec<usize> = if rest.is_empty() {
        (0..shifts.len()).collect()
    } else {
        let dims_vec = value_to_dims_vector(&rest[0])?;
        if dims_vec.len() != shifts.len() {
            return Err(
                "circshift: shift and dimension vectors must have the same length".to_string(),
            );
        }
        dims_vec.into_iter().map(|dim| dim - 1).collect()
    };

    if dims.len() != shifts.len() {
        return Err("circshift: shift vector must match the number of dimensions".to_string());
    }

    let mut seen = HashSet::new();
    for &dim in &dims {
        if !seen.insert(dim) {
            return Err("circshift: dimension indices must be unique".to_string());
        }
    }

    Ok(CircshiftSpec { dims, shifts })
}

fn value_to_shift_vector(value: &Value) -> Result<Vec<isize>, String> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < isize::MIN as i64 || raw > isize::MAX as i64 {
                return Err("circshift: shift magnitude is too large".to_string());
            }
            Ok(vec![raw as isize])
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err("circshift: shift values must be finite numbers".to_string());
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err("circshift: shifts must be integers".to_string());
            }
            if rounded < isize::MIN as f64 || rounded > isize::MAX as f64 {
                return Err("circshift: shift magnitude is too large".to_string());
            }
            Ok(vec![rounded as isize])
        }
        Value::Tensor(tensor) => {
            if !is_vector_shape(&tensor.shape) && !tensor.data.is_empty() {
                return Err("circshift: shifts must be specified as a scalar or vector".to_string());
            }
            tensor
                .data
                .iter()
                .map(|val| numeric_to_isize(*val))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("circshift: {e}"))
        }
        Value::LogicalArray(array) => {
            if !is_vector_shape(&array.shape) && !array.data.is_empty() {
                return Err("circshift: shifts must be specified as a scalar or vector".to_string());
            }
            Ok(array
                .data
                .iter()
                .map(|&b| if b != 0 { 1 } else { 0 })
                .collect())
        }
        Value::Bool(flag) => Ok(vec![if *flag { 1 } else { 0 }]),
        Value::GpuTensor(_) => Err("circshift: shift vector must reside on the host".to_string()),
        Value::StringArray(_) | Value::CharArray(_) | Value::String(_) => {
            Err("circshift: shift values must be numeric".to_string())
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("circshift: shifts must be real integers".to_string())
        }
        Value::Cell(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err("circshift: unsupported shift argument type".to_string()),
    }
}

fn value_to_dims_vector(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 1 {
                return Err("circshift: dimensions must be >= 1".to_string());
            }
            Ok(vec![raw as usize])
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err("circshift: dimensions must be finite integers".to_string());
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err("circshift: dimensions must be integers".to_string());
            }
            if rounded < 1.0 {
                return Err("circshift: dimensions must be >= 1".to_string());
            }
            Ok(vec![rounded as usize])
        }
        Value::Tensor(tensor) => {
            if !is_vector_shape(&tensor.shape) && !tensor.data.is_empty() {
                return Err(
                    "circshift: dimension vectors must be row or column vectors".to_string()
                );
            }
            let mut dims = Vec::with_capacity(tensor.data.len());
            for &val in &tensor.data {
                if !val.is_finite() {
                    return Err("circshift: dimensions must be finite integers".to_string());
                }
                let rounded = val.round();
                if (rounded - val).abs() > f64::EPSILON {
                    return Err("circshift: dimensions must be integers".to_string());
                }
                if rounded < 1.0 {
                    return Err("circshift: dimensions must be >= 1".to_string());
                }
                dims.push(rounded as usize);
            }
            Ok(dims)
        }
        Value::LogicalArray(array) => {
            if !is_vector_shape(&array.shape) && !array.data.is_empty() {
                return Err(
                    "circshift: dimension vectors must be row or column vectors".to_string()
                );
            }
            let mut dims = Vec::new();
            for (idx, &flag) in array.data.iter().enumerate() {
                if flag != 0 {
                    dims.push(idx + 1);
                }
            }
            Ok(dims)
        }
        Value::Bool(flag) => {
            if *flag {
                Ok(vec![1])
            } else {
                Err("circshift: dimension indices must be >= 1".to_string())
            }
        }
        Value::GpuTensor(_) => {
            Err("circshift: dimension vector must reside on the host".to_string())
        }
        Value::StringArray(_) | Value::CharArray(_) | Value::String(_) => {
            Err("circshift: dimension indices must be numeric".to_string())
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err("circshift: dimensions must be real integers".to_string())
        }
        Value::Cell(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err("circshift: unsupported dimension argument type".to_string()),
    }
}

fn numeric_to_isize(value: f64) -> Result<isize, String> {
    if !value.is_finite() {
        return Err("shift values must be finite".to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err("shift values must be integers".to_string());
    }
    if rounded < isize::MIN as f64 || rounded > isize::MAX as f64 {
        return Err("shift magnitude is too large".to_string());
    }
    Ok(rounded as isize)
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().copied().filter(|&dim| dim > 1).count() <= 1
}

#[derive(Debug, Clone)]
struct ShiftPlan {
    ext_shape: Vec<usize>,
    positive: Vec<usize>,
    provider: Vec<isize>,
}

impl ShiftPlan {
    fn is_noop(&self) -> bool {
        self.positive.iter().all(|&shift| shift == 0)
    }
}

fn build_shift_plan(
    shape: &[usize],
    dims: &[usize],
    shifts: &[isize],
) -> Result<ShiftPlan, String> {
    if dims.len() != shifts.len() {
        return Err("circshift: shift vector must match the number of dimensions".to_string());
    }
    let mut target_len = shape.len();
    if let Some(max_axis) = dims.iter().copied().max() {
        if max_axis + 1 > target_len {
            target_len = max_axis + 1;
        }
    }
    let mut ext_shape = shape.to_vec();
    if target_len > ext_shape.len() {
        ext_shape.resize(target_len, 1);
    }

    let mut positive = vec![0usize; ext_shape.len()];
    let mut provider = vec![0isize; ext_shape.len()];

    for (&axis, &shift) in dims.iter().zip(shifts.iter()) {
        if axis >= ext_shape.len() {
            return Err("circshift: dimension index out of range".to_string());
        }
        provider[axis] = shift;
        let size = ext_shape[axis];
        if size == 0 || size == 1 {
            continue;
        }
        let size_isize = size as isize;
        let mut normalized = shift % size_isize;
        if normalized < 0 {
            normalized += size_isize;
        }
        positive[axis] = normalized as usize;
    }

    Ok(ShiftPlan {
        ext_shape,
        positive,
        provider,
    })
}

fn normalize_shift_amount(shift: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let len_isize = len as isize;
    let mut normalized = shift % len_isize;
    if normalized < 0 {
        normalized += len_isize;
    }
    normalized as usize
}

fn circshift_tensor(tensor: Tensor, dims: &[usize], shifts: &[isize]) -> Result<Tensor, String> {
    let Tensor { data, shape, .. } = tensor;
    let plan = build_shift_plan(&shape, dims, shifts)?;
    if data.is_empty() || plan.is_noop() {
        return Tensor::new(data, shape).map_err(|e| format!("circshift: {e}"));
    }
    let ShiftPlan {
        ext_shape,
        positive,
        ..
    } = plan;
    let rotated = circshift_generic(&data, &ext_shape, &positive)?;
    Tensor::new(rotated, ext_shape).map_err(|e| format!("circshift: {e}"))
}

fn circshift_complex_tensor(
    tensor: ComplexTensor,
    dims: &[usize],
    shifts: &[isize],
) -> Result<ComplexTensor, String> {
    let ComplexTensor { data, shape, .. } = tensor;
    let plan = build_shift_plan(&shape, dims, shifts)?;
    if data.is_empty() || plan.is_noop() {
        return ComplexTensor::new(data, shape).map_err(|e| format!("circshift: {e}"));
    }
    let ShiftPlan {
        ext_shape,
        positive,
        ..
    } = plan;
    let rotated = circshift_generic(&data, &ext_shape, &positive)?;
    ComplexTensor::new(rotated, ext_shape).map_err(|e| format!("circshift: {e}"))
}

fn circshift_logical_array(
    array: LogicalArray,
    dims: &[usize],
    shifts: &[isize],
) -> Result<LogicalArray, String> {
    let LogicalArray { data, shape } = array;
    let plan = build_shift_plan(&shape, dims, shifts)?;
    if data.is_empty() || plan.is_noop() {
        return LogicalArray::new(data, shape).map_err(|e| format!("circshift: {e}"));
    }
    let ShiftPlan {
        ext_shape,
        positive,
        ..
    } = plan;
    let rotated = circshift_generic(&data, &ext_shape, &positive)?;
    LogicalArray::new(rotated, ext_shape).map_err(|e| format!("circshift: {e}"))
}

fn circshift_string_array(
    array: StringArray,
    dims: &[usize],
    shifts: &[isize],
) -> Result<StringArray, String> {
    let StringArray { data, shape, .. } = array;
    let plan = build_shift_plan(&shape, dims, shifts)?;
    if data.is_empty() || plan.is_noop() {
        return StringArray::new(data, shape).map_err(|e| format!("circshift: {e}"));
    }
    let ShiftPlan {
        ext_shape,
        positive,
        ..
    } = plan;
    let rotated = circshift_generic(&data, &ext_shape, &positive)?;
    StringArray::new(rotated, ext_shape).map_err(|e| format!("circshift: {e}"))
}

fn circshift_char_array(
    array: CharArray,
    dims: &[usize],
    shifts: &[isize],
) -> Result<Value, String> {
    let mut row_shift = 0isize;
    let mut col_shift = 0isize;
    for (&axis, &shift) in dims.iter().zip(shifts.iter()) {
        match axis {
            0 => row_shift = shift,
            1 => col_shift = shift,
            _ => {
                if shift != 0 {
                    return Err(
                        "circshift: character arrays only support dimensions 1 and 2".to_string(),
                    );
                }
            }
        }
    }
    let CharArray { data, rows, cols } = array;
    if data.is_empty() {
        return CharArray::new(data, rows, cols)
            .map(Value::CharArray)
            .map_err(|e| format!("circshift: {e}"));
    }
    let row_shift = normalize_shift_amount(row_shift, rows);
    let col_shift = normalize_shift_amount(col_shift, cols);
    if row_shift == 0 && col_shift == 0 {
        return CharArray::new(data, rows, cols)
            .map(Value::CharArray)
            .map_err(|e| format!("circshift: {e}"));
    }
    let mut out = vec!['\0'; data.len()];
    for row in 0..rows {
        for col in 0..cols {
            let src_row = if rows == 0 {
                0
            } else {
                (row + rows - row_shift) % rows
            };
            let src_col = if cols == 0 {
                0
            } else {
                (col + cols - col_shift) % cols
            };
            let dst_idx = row * cols + col;
            let src_idx = src_row * cols + src_col;
            out[dst_idx] = data[src_idx];
        }
    }
    CharArray::new(out, rows, cols)
        .map(Value::CharArray)
        .map_err(|e| format!("circshift: {e}"))
}

fn circshift_gpu(
    handle: GpuTensorHandle,
    dims: &[usize],
    shifts: &[isize],
) -> Result<Value, String> {
    let plan = build_shift_plan(&handle.shape, dims, shifts)?;
    if plan.is_noop() {
        return Ok(Value::GpuTensor(handle));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let mut working = handle.clone();
        if plan.ext_shape != working.shape {
            match provider.reshape(&working, &plan.ext_shape) {
                Ok(reshaped) => working = reshaped,
                Err(_) => return circshift_gpu_fallback(handle, dims, shifts),
            }
        }
        if let Ok(out) = provider.circshift(&working, &plan.provider) {
            return Ok(Value::GpuTensor(out));
        }
    }

    circshift_gpu_fallback(handle, dims, shifts)
}

fn circshift_gpu_fallback(
    handle: GpuTensorHandle,
    dims: &[usize],
    shifts: &[isize],
) -> Result<Value, String> {
    let host_tensor = gpu_helpers::gather_tensor(&handle)?;
    let rotated = circshift_tensor(host_tensor, dims, shifts)?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &rotated.data,
            shape: &rotated.shape,
        };
        return provider
            .upload(&view)
            .map(Value::GpuTensor)
            .map_err(|e| format!("circshift: {e}"));
    }
    Ok(tensor::tensor_into_value(rotated))
}

fn circshift_generic<T: Clone>(
    data: &[T],
    shape: &[usize],
    shifts: &[usize],
) -> Result<Vec<T>, String> {
    if shape.len() != shifts.len() {
        return Err("circshift: internal shape mismatch".to_string());
    }
    let total: usize = shape.iter().product();
    if total != data.len() {
        return Err("circshift: shape does not match data length".to_string());
    }
    if total == 0 {
        return Ok(Vec::new());
    }
    if shifts.iter().all(|&s| s == 0) {
        return Ok(data.to_vec());
    }

    let mut strides = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        strides[axis] = strides[axis - 1] * shape[axis - 1];
    }

    let mut result = vec![data[0].clone(); total];
    let mut coords = vec![0usize; shape.len()];

    for (dest_idx, slot) in result.iter_mut().enumerate() {
        let mut remainder = dest_idx;
        for axis in 0..shape.len() {
            let size = shape[axis];
            coords[axis] = if size == 0 { 0 } else { remainder % size };
            if size > 0 {
                remainder /= size;
            }
        }
        let mut src_idx = 0usize;
        for axis in 0..shape.len() {
            let size = shape[axis];
            if size == 0 {
                continue;
            }
            let stride = strides[axis];
            if size <= 1 || shifts[axis] == 0 {
                src_idx += coords[axis] * stride;
            } else {
                let shift = shifts[axis] % size;
                let src_coord = (coords[axis] + size - shift) % size;
                src_idx += src_coord * stride;
            }
        }
        *slot = data[src_idx].clone();
    }

    Ok(result)
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor.data.len() == 1 {
        let (re, im) = tensor.data[0];
        Value::Complex(re, im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{CharArray, IntValue, LogicalArray, StringArray, Tensor};

    #[test]
    fn circshift_vector_positive_shift() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5, 1]).unwrap();
        let result = circshift_builtin(
            Value::Tensor(tensor),
            Value::Int(IntValue::I32(2)),
            Vec::new(),
        )
        .expect("circshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![5, 1]);
                assert_eq!(out.data, vec![4.0, 5.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn circshift_matrix_negative_column_shift() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let shift_vec = Tensor::new(vec![0.0, -1.0], vec![1, 2]).unwrap();
        let result =
            circshift_builtin(Value::Tensor(tensor), Value::Tensor(shift_vec), Vec::new()).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![2.0, 5.0, 3.0, 6.0, 1.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn circshift_column_vector_shift() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let shift_vec = Tensor::new(vec![1.0, -2.0], vec![2, 1]).unwrap();
        let expected = circshift_tensor(tensor.clone(), &[0, 1], &[1, -2]).expect("expected shift");
        let result =
            circshift_builtin(Value::Tensor(tensor), Value::Tensor(shift_vec), Vec::new()).unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn circshift_with_dimension_argument() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = circshift_builtin(
            Value::Tensor(tensor),
            Value::Int(IntValue::I32(-1)),
            vec![Value::Int(IntValue::I32(2))],
        )
        .unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![2.0, 4.0, 1.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn circshift_dims_tensor_argument() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let shift = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let expected =
            circshift_tensor(tensor.clone(), &[0, 1], &[1, -1]).expect("expected host shift");
        let result = circshift_builtin(
            Value::Tensor(tensor),
            Value::Tensor(shift),
            vec![Value::Tensor(dims)],
        )
        .expect("circshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn circshift_logical_array_supported() {
        let array = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let shift_vec = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let result = circshift_builtin(
            Value::LogicalArray(array),
            Value::Tensor(shift_vec),
            Vec::new(),
        )
        .unwrap();
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![0, 1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn circshift_dims_logical_mask() {
        let tensor =
            Tensor::new((1..=8).map(|v| v as f64).collect::<Vec<_>>(), vec![2, 2, 2]).unwrap();
        let mask = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let shift = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let expected =
            circshift_tensor(tensor.clone(), &[0, 2], &[1, -1]).expect("expected host shift");
        let result = circshift_builtin(
            Value::Tensor(tensor),
            Value::Tensor(shift),
            vec![Value::LogicalArray(mask)],
        )
        .expect("circshift");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn circshift_string_array_rotation() {
        let array = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).unwrap();
        let result = circshift_builtin(
            Value::StringArray(array),
            Value::Int(IntValue::I32(-1)),
            Vec::new(),
        )
        .unwrap();
        match result {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                assert_eq!(
                    out.data,
                    vec!["b".to_string(), "c".to_string(), "a".to_string()]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn circshift_char_array_rows() {
        let chars = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        let result = circshift_builtin(
            Value::CharArray(chars),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            Vec::new(),
        )
        .unwrap();
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 2);
                assert_eq!(out.data, vec!['c', 'd', 'a', 'b']);
            }
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    #[test]
    fn circshift_noop_preserves_shape() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let shift_vec = Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let result = circshift_builtin(
            Value::Tensor(tensor.clone()),
            Value::Tensor(shift_vec),
            Vec::new(),
        )
        .unwrap();
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, tensor.shape);
                assert_eq!(out.data, tensor.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn circshift_rejects_duplicate_dims() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let err = circshift_builtin(
            Value::Tensor(tensor),
            Value::Tensor(Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap()),
            vec![Value::Tensor(dims)],
        )
        .unwrap_err();
        assert!(err.contains("dimension indices must be unique"));
    }

    #[test]
    fn circshift_rejects_non_integer_shift() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err =
            circshift_builtin(Value::Tensor(tensor), Value::Num(1.5), Vec::new()).unwrap_err();
        assert!(err.contains("shifts must be integers"));
    }

    #[test]
    fn circshift_dimension_length_mismatch() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let shift = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let dims = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = circshift_builtin(
            Value::Tensor(tensor),
            Value::Tensor(shift),
            vec![Value::Tensor(dims)],
        )
        .unwrap_err();
        assert!(err.contains("must have the same length"));
    }

    #[test]
    fn circshift_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = circshift_builtin(
                Value::GpuTensor(handle),
                Value::Tensor(Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap()),
                Vec::new(),
            )
            .expect("circshift");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![4.0, 2.0, 3.0, 1.0]);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn circshift_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let shift = Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap();
        let cpu = circshift_builtin(
            Value::Tensor(tensor.clone()),
            Value::Tensor(shift.clone()),
            Vec::new(),
        )
        .expect("cpu circshift");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload tensor");
        let gpu_value =
            circshift_builtin(Value::GpuTensor(handle), Value::Tensor(shift), Vec::new())
                .expect("gpu circshift");
        let gathered = test_support::gather(gpu_value).expect("gather gpu result");
        match cpu {
            Value::Tensor(expected) => {
                assert_eq!(expected.shape, gathered.shape);
                assert_eq!(expected.data, gathered.data);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
