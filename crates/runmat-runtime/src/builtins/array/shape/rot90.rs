//! MATLAB-compatible `rot90` builtin with GPU-aware semantics for RunMat.
//!
//! This module implements the `rot90` function, matching MathWorks MATLAB behaviour
//! for rotating matrices and higher-dimensional arrays by multiples of 90 degrees
//! around the first two axes. The implementation handles numeric tensors, logical
//! masks, complex data, string arrays, character matrices, and `gpuArray` inputs.
//! When a GPU provider supplies `permute` and `flip` hooks the runtime performs the
//! rotation entirely on the device; otherwise it gathers the data, rotates on the
//! host, and re-uploads the result so downstream operations keep GPU residency.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "rot90",
        builtin_path = "crate::builtins::array::shape::rot90"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "rot90"
category: "array/shape"
keywords: ["rot90", "rotate", "90 degrees", "matrix", "gpu", "clockwise", "counterclockwise"]
summary: "Rotate matrices and N-D arrays by multiples of 90 degrees."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses provider permute + flip hooks when available; otherwise gathers once, rotates on the host, and re-uploads."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::shape::rot90::tests"
  integration: "builtins::array::shape::rot90::tests::rot90_gpu_roundtrip"
---

# What does the `rot90` function do in MATLAB / RunMat?
`rot90(A)` rotates the matrix `A` by 90 degrees counterclockwise. A second argument
specifies additional 90-degree turns (positive values rotate counterclockwise, negative
values clockwise). For N-D arrays, only the first two dimensions participate in the
rotation; trailing dimensions remain unchanged.

## How does the `rot90` function behave in MATLAB / RunMat?
- Default behaviour rotates 90 degrees counterclockwise (`k = 1`).
- `rot90(A, K)` rotates by `K * 90°`; the rotation count can be positive, negative,
  or zero. Any integer multiple of four leaves the input unchanged.
- The direction keywords `'clockwise'` and `'counterclockwise'` are accepted as an
  alternative to the numeric argument (case-insensitive).
- Works for numeric tensors, logical masks, complex arrays, string arrays, and
  character matrices. Scalars are unchanged.
- For empty dimensions the function still swaps the first two extents, so
  `rot90(zeros(0, 3))` returns a `3×0` array.

## `rot90` Function GPU Execution Behaviour
When RunMat Accelerate is enabled, the runtime first looks for GPU provider
hooks to keep the rotation on the device:
- Providers that implement both `permute` and `flip` can realise the rotation
  without leaving the GPU, preserving residency for downstream operations.
- If a dedicated `rot90` kernel is exposed the provider may call it instead of
  composing lower-level hooks.
- When no compatible hooks are available, RunMat gathers the tensor once,
  rotates it on the host, and re-uploads the rotated tensor so subsequent GPU
  work continues without surprises.

## Examples of using the `rot90` function in MATLAB / RunMat

### Rotating a matrix 90 degrees counterclockwise
```matlab
A = [1 2 3; 4 5 6];
B = rot90(A);
```
Expected output:
```matlab
B =
     3     6
     2     5
     1     4
```

### Rotating a matrix clockwise using a direction keyword
```matlab
A = magic(3);
C = rot90(A, 'clockwise');
```
Expected output:
```matlab
C =
     6     1     8
     7     5     3
     2     9     4
```

### Applying multiple 90-degree turns with a numeric count
```matlab
A = reshape(1:9, [3 3]);
B = rot90(A, 2);    % 180-degree rotation
```
Expected output:
```matlab
B =
     9     8     7
     6     5     4
     3     2     1
```

### Rotating 3-D data while preserving trailing dimensions
```matlab
T = reshape(1:12, [2 3 2]);
R = rot90(T);
size(R)
```
Expected output:
```matlab
ans =
     3     2     2
```

### Rotating character data to reorganise text
```matlab
C = ['r','u','n'; 'm','a','t'; ' ','A','I'];
R = rot90(C);
```
Expected output:
```matlab
R =
    'ntI'
    'uaA'
    'rm '
```

### Rotating gpuArray data and keeping it on the device
```matlab
G = gpuArray(reshape(1:9, [3 3]));
H = rot90(G, -1);     % rotate clockwise
isgpuarray(H)
```
Expected output:
```matlab
ans = logical 1
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Not usually. The auto-offload planner keeps tensors on the GPU whenever it is
profitable. Explicitly creating a `gpuArray` matches MATLAB syntax, but RunMat
will also auto-promote host tensors when the planner determines that rotating
them on the device avoids unnecessary transfers. When the active provider lacks
native support the runtime downloads the tensor once, rotates on the host, and
uploads the rotated tensor back to the GPU so downstream operations continue to
benefit from residency information.

## FAQ
### What directions does `rot90` support?
Numeric rotation counts (integers) and the strings `'clockwise'` / `'counterclockwise'`
are recognised. Any other strings raise an error.

### Does `rot90` modify dimensions beyond the first two?
No. Only the first two axes participate in the rotation. Other dimensions keep
their order and extents unchanged.

### What happens for empty matrices?
Empty inputs still swap the first two dimension sizes. For example, a `0×5`
matrix becomes `5×0` after a single counterclockwise rotation.

### Can I rotate logical, string, or character arrays?
Yes. Logical results remain logical, string arrays preserve their elements, and
character matrices rotate their characters exactly like numeric data.

### How do large rotation counts behave?
The rotation count is reduced modulo 4. Values such as `rot90(A, 37)` behave the
same as `rot90(A, 1)`.

### Is `rot90` compatible with complex numbers?
Absolutely. Complex tensors rotate without altering the real or imaginary parts;
only the element positions change.

### Can providers implement a custom kernel?
Yes. Providers may implement a specialised `rot90` kernel. When unavailable, the
runtime composes the operation using the `permute` and `flip` hooks or falls back
to the host implementation.

## See Also
[`flip`](./flip), [`permute`](./permute), [`reshape`](./reshape), [`kron`](./kron),
[`gpuArray`](./gpuarray), [`gather`](./gather)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/array/shape/rot90.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/shape/rot90.rs)
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::rot90")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rot90",
    op_kind: GpuOpKind::Custom("rot90"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("permute"),
        ProviderHook::Custom("flip"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Preferred implementation composes provider permute + flip hooks; providers may offer a dedicated rot90 fast path.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::rot90")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rot90",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Rotations only reorder data; fusion planner treats rot90 as a residency-preserving boundary.",
};

fn rot90_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("rot90").build()
}

#[runtime_builtin(
    name = "rot90",
    category = "array/shape",
    summary = "Rotate matrices and N-D arrays by multiples of 90 degrees.",
    keywords = "rot90,rotate,90 degrees,matrix,gpu,clockwise,counterclockwise",
    accel = "custom",
    builtin_path = "crate::builtins::array::shape::rot90"
)]
fn rot90_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(rot90_error("rot90: too many input arguments"));
    }
    let steps = parse_rotation_steps(rest.first())?;
    match value {
        Value::Tensor(tensor) => Ok(rot90_tensor(tensor, steps).map(tensor::tensor_into_value)?),
        Value::LogicalArray(logical) => Ok(rot90_logical(logical, steps).map(Value::LogicalArray)?),
        Value::ComplexTensor(ct) => Ok(rot90_complex_tensor(ct, steps).map(Value::ComplexTensor)?),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| rot90_error(format!("rot90: {e}")))?;
            Ok(rot90_complex_tensor(tensor, steps).map(complex_tensor_into_value)?)
        }
        Value::StringArray(strings) => {
            Ok(rot90_string_array(strings, steps).map(Value::StringArray)?)
        }
        Value::CharArray(chars) => Ok(rot90_char_array(chars, steps).map(Value::CharArray)?),
        Value::String(s) => Ok(Value::String(s)),
        v @ (Value::Num(_) | Value::Int(_) | Value::Bool(_)) => {
            let tensor = tensor::value_into_tensor_for("rot90", v).map_err(|e| rot90_error(e))?;
            Ok(rot90_tensor(tensor, steps).map(tensor::tensor_into_value)?)
        }
        Value::GpuTensor(handle) => Ok(rot90_gpu(handle, steps)?),
        Value::Cell(_) => Err(rot90_error("rot90: cell arrays are not yet supported")),
        Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(rot90_error("rot90: unsupported input type")),
    }
}

fn parse_rotation_steps(arg: Option<&Value>) -> crate::BuiltinResult<usize> {
    let raw = match arg {
        None => 1,
        Some(value) => parse_rotation_value(value)?,
    };
    let modulo = ((raw % 4 + 4) % 4) as usize;
    Ok(modulo)
}

fn parse_rotation_value(value: &Value) -> crate::BuiltinResult<i64> {
    if let Some(direction) = parse_direction(value)? {
        return Ok(direction);
    }
    match value {
        Value::Int(i) => Ok(i.to_i64()),
        Value::Num(n) => parse_numeric_rotation(*n),
        Value::Bool(flag) => Ok(if *flag { 1 } else { 0 }),
        Value::Tensor(t) => parse_tensor_rotation(t),
        Value::LogicalArray(array) => parse_logical_rotation(array),
        Value::StringArray(sa) if sa.data.len() != 1 => Err(rot90_error(
            "rot90: rotation direction must be a scalar string",
        )),
        Value::StringArray(_) | Value::String(_) | Value::CharArray(_) => {
            Err(rot90_error("rot90: unknown rotation direction string"))
        }
        Value::GpuTensor(_) => Err(rot90_error(
            "rot90: rotation count must be specified on the host (numeric or direction string)",
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(rot90_error("rot90: K must be an integer"))
        }
        _ => Err(rot90_error(
            "rot90: rotation count must be numeric or a direction string",
        )),
    }
}

fn parse_direction(value: &Value) -> crate::BuiltinResult<Option<i64>> {
    let text = match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    };
    if let Some(text) = text {
        let lowered = text.trim().to_ascii_lowercase();
        let turns = match lowered.as_str() {
            "clockwise" | "cw" => -1,
            "counterclockwise" | "anticlockwise" | "ccw" | "acw" => 1,
            other => {
                return Err(rot90_error(format!(
                    "rot90: unknown rotation direction '{other}'"
                )));
            }
        };
        return Ok(Some(turns));
    }
    Ok(None)
}

fn parse_numeric_rotation(value: f64) -> crate::BuiltinResult<i64> {
    if !value.is_finite() {
        return Err(rot90_error("rot90: K must be finite"));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(rot90_error("rot90: K must be an integer"));
    }
    Ok(rounded as i64)
}

fn parse_tensor_rotation(tensor: &Tensor) -> crate::BuiltinResult<i64> {
    if tensor.data.len() != 1 {
        return Err(rot90_error("rot90: K must be a scalar integer"));
    }
    parse_numeric_rotation(tensor.data[0])
}

fn parse_logical_rotation(array: &LogicalArray) -> crate::BuiltinResult<i64> {
    if array.data.len() != 1 {
        return Err(rot90_error("rot90: K must be a scalar integer"));
    }
    Ok(if array.data[0] != 0 { 1 } else { 0 })
}

fn rot90_tensor(tensor: Tensor, steps: usize) -> crate::BuiltinResult<Tensor> {
    if steps == 0 {
        return Ok(tensor);
    }
    let (data, shape) = rot90_generic(&tensor.data, &tensor.shape, steps)?;
    Tensor::new(data, shape).map_err(|e| rot90_error(format!("rot90: {e}")))
}

fn rot90_complex_tensor(
    tensor: ComplexTensor,
    steps: usize,
) -> crate::BuiltinResult<ComplexTensor> {
    if steps == 0 {
        return Ok(tensor);
    }
    let (data, shape) = rot90_generic(&tensor.data, &tensor.shape, steps)?;
    ComplexTensor::new(data, shape).map_err(|e| rot90_error(format!("rot90: {e}")))
}

fn rot90_logical(array: LogicalArray, steps: usize) -> crate::BuiltinResult<LogicalArray> {
    if steps == 0 {
        return Ok(array);
    }
    let (data, shape) = rot90_generic(&array.data, &array.shape, steps)?;
    LogicalArray::new(data, shape).map_err(|e| rot90_error(format!("rot90: {e}")))
}

fn rot90_string_array(array: StringArray, steps: usize) -> crate::BuiltinResult<StringArray> {
    if steps == 0 {
        return Ok(array);
    }
    let (data, shape) = rot90_generic(&array.data, &array.shape, steps)?;
    StringArray::new(data, shape).map_err(|e| rot90_error(format!("rot90: {e}")))
}

fn rot90_char_array(array: CharArray, steps: usize) -> crate::BuiltinResult<CharArray> {
    if steps == 0 {
        return Ok(array);
    }
    let rows = array.rows;
    let cols = array.cols;
    let mut out_rows = rows;
    let mut out_cols = cols;
    if steps % 2 == 1 {
        std::mem::swap(&mut out_rows, &mut out_cols);
    }
    let mut out = vec!['\0'; out_rows * out_cols];
    if rows == 0 || cols == 0 {
        return CharArray::new(out, out_rows, out_cols)
            .map_err(|e| rot90_error(format!("rot90: {e}")));
    }
    for row in 0..rows {
        for col in 0..cols {
            let src_idx = row * cols + col;
            let (dest_row, dest_col) = match steps {
                1 => (cols - 1 - col, row),
                2 => (rows - 1 - row, cols - 1 - col),
                3 => (col, rows - 1 - row),
                _ => (row, col),
            };
            let dst_idx = dest_row * out_cols + dest_col;
            out[dst_idx] = array.data[src_idx];
        }
    }
    CharArray::new(out, out_rows, out_cols).map_err(|e| rot90_error(format!("rot90: {e}")))
}

fn rot90_gpu(handle: GpuTensorHandle, steps: usize) -> crate::BuiltinResult<Value> {
    if steps == 0 {
        return Ok(Value::GpuTensor(handle));
    }
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(out) = rot90_gpu_via_provider(provider, &handle, steps) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let host_tensor = gpu_helpers::gather_tensor(&handle)?;
    let rotated = rot90_tensor(host_tensor, steps)?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &rotated.data,
            shape: &rotated.shape,
        };
        provider
            .upload(&view)
            .map(Value::GpuTensor)
            .map_err(|e| rot90_error(format!("rot90: {e}")))
    } else {
        Ok(tensor::tensor_into_value(rotated))
    }
}

fn rot90_gpu_via_provider(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
    steps: usize,
) -> Option<GpuTensorHandle> {
    let rank = handle.shape.len();
    if rank < 2 {
        return None;
    }
    match steps {
        1 => {
            let mut order: Vec<usize> = (0..rank).collect();
            order.swap(0, 1);
            match provider.permute(handle, &order) {
                Ok(perm_handle) => {
                    let result = provider.flip(&perm_handle, &[0]);
                    let _ = provider.free(&perm_handle);
                    result.ok()
                }
                Err(_) => None,
            }
        }
        2 => provider.flip(handle, &[0, 1]).ok(),
        3 => {
            let mut order: Vec<usize> = (0..rank).collect();
            order.swap(0, 1);
            match provider.permute(handle, &order) {
                Ok(perm_handle) => {
                    let result = provider.flip(&perm_handle, &[1]);
                    let _ = provider.free(&perm_handle);
                    result.ok()
                }
                Err(_) => None,
            }
        }
        _ => None,
    }
}

fn rot90_generic<T: Clone>(
    data: &[T],
    shape: &[usize],
    steps: usize,
) -> crate::BuiltinResult<(Vec<T>, Vec<usize>)> {
    let ext_shape = if shape.is_empty() {
        vec![1, 1]
    } else if shape.len() == 1 {
        vec![shape[0], 1]
    } else {
        shape.to_vec()
    };
    let total: usize = ext_shape.iter().product();
    if total != data.len() {
        return Err(rot90_error(
            "rot90: data length does not match shape product",
        ));
    }
    let rows = ext_shape[0];
    let cols = ext_shape[1];
    let mut out_shape = ext_shape.clone();
    if steps % 2 == 1 {
        out_shape[0] = cols;
        out_shape[1] = rows;
    }
    let total_out: usize = out_shape.iter().product();
    let mut out = if total_out == 0 {
        Vec::new()
    } else {
        vec![data[0].clone(); total_out]
    };
    let rest_dims = &ext_shape[2..];
    let rest_total: usize = if rest_dims.is_empty() {
        1
    } else {
        rest_dims.iter().product()
    };
    for rest_index in 0..rest_total {
        let rest_coords = if rest_dims.is_empty() {
            Vec::new()
        } else {
            unravel_index(rest_index, rest_dims)
        };
        for col in 0..cols {
            for row in 0..rows {
                let mut src_coords = Vec::with_capacity(2 + rest_coords.len());
                src_coords.push(row);
                src_coords.push(col);
                src_coords.extend(rest_coords.iter().copied());
                let src_index = ravel_index(&src_coords, &ext_shape);
                let (dest_row, dest_col) = match steps {
                    1 => (cols.saturating_sub(1).saturating_sub(col), row),
                    2 => (
                        rows.saturating_sub(1).saturating_sub(row),
                        cols.saturating_sub(1).saturating_sub(col),
                    ),
                    3 => (col, rows.saturating_sub(1).saturating_sub(row)),
                    _ => (row, col),
                };
                let mut dst_coords = Vec::with_capacity(2 + rest_coords.len());
                dst_coords.push(dest_row);
                dst_coords.push(dest_col);
                dst_coords.extend(rest_coords.iter().copied());
                let dst_index = ravel_index(&dst_coords, &out_shape);
                if dst_index < out.len() {
                    out[dst_index] = data[src_index].clone();
                }
            }
        }
    }
    Ok((out, out_shape))
}

fn unravel_index(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(shape.len());
    for &extent in shape {
        if extent == 0 {
            coords.push(0);
        } else {
            coords.push(index % extent);
            index /= extent;
        }
    }
    coords
}

fn ravel_index(coords: &[usize], shape: &[usize]) -> usize {
    let mut index = 0usize;
    let mut stride = 1usize;
    for (coord, extent) in coords.iter().zip(shape.iter()) {
        if *extent > 0 {
            index += coord * stride;
            stride *= extent;
        }
    }
    index
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
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_default_counterclockwise() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = rot90_builtin(Value::Tensor(tensor), Vec::new()).expect("rot90");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 2]);
                assert_eq!(out.data, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_two_rotations() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = rot90_builtin(Value::Tensor(tensor), args).expect("rot90 k=2");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![6.0, 3.0, 5.0, 2.0, 4.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_clockwise_direction() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = rot90_builtin(Value::Tensor(tensor), vec![Value::from("clockwise")])
            .expect("rot90 clockwise");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 2]);
                assert_eq!(out.data, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_counterclockwise_direction_keyword() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let result = rot90_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("counterclockwise")],
        )
        .expect("rot90 counterclockwise");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                // 90° counterclockwise should match the default call.
                let default =
                    rot90_builtin(Value::Tensor(tensor), Vec::new()).expect("rot90 default");
                match default {
                    Value::Tensor(default_tensor) => {
                        assert_eq!(out.shape, default_tensor.shape);
                        assert_eq!(out.data, default_tensor.data);
                    }
                    other => panic!("expected tensor, got {other:?}"),
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_negative_rotation_count() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let args = vec![Value::Int(IntValue::I32(-1))];
        let result = rot90_builtin(Value::Tensor(tensor), args).expect("rot90 k=-1");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![4.0, 5.0, 1.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_rotation_count_from_tensor_scalar() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let k_tensor = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let args = vec![Value::Tensor(k_tensor)];
        let result = rot90_builtin(Value::Tensor(tensor), args).expect("rot90 tensor k");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![4.0, 5.0, 1.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_rotation_count_from_logical_scalar() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let logical = LogicalArray::new(vec![1], vec![1]).unwrap();
        let args = vec![Value::LogicalArray(logical)];
        let result = rot90_builtin(Value::Tensor(tensor), args).expect("rot90 logical k");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![2.0, 1.0, 5.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_logical_array_input() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let result =
            rot90_builtin(Value::LogicalArray(logical), Vec::new()).expect("rot90 logical input");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![0, 1, 1, 0]);
            }
            other => panic!("expected LogicalArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_empty_matrix_swaps_extents() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
        let result = rot90_builtin(Value::Tensor(tensor), Vec::new()).expect("rot90 empty");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 0]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_preserves_trailing_dimensions() {
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let tensor = Tensor::new(data, vec![2, 3, 2]).unwrap();
        let result = rot90_builtin(Value::Tensor(tensor), Vec::new()).expect("rot90 3d");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 2, 2]);
                assert_eq!(
                    out.data,
                    vec![5.0, 3.0, 1.0, 6.0, 4.0, 2.0, 11.0, 9.0, 7.0, 12.0, 10.0, 8.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_four_turns_returns_original() {
        let data: Vec<f64> = (1..=24).map(|n| n as f64).collect();
        let tensor = Tensor::new(data.clone(), vec![2, 3, 4]).unwrap();
        let mut value = Value::Tensor(tensor);
        for _ in 0..4 {
            value = rot90_builtin(value, Vec::new()).expect("rot90");
        }
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3, 4]);
                assert_eq!(out.data, data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_char_array_roundtrip() {
        let chars = CharArray::new("runmat".chars().collect(), 2, 3).unwrap();
        let result = rot90_builtin(Value::CharArray(chars), Vec::new()).expect("rot90 char");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 3);
                assert_eq!(out.cols, 2);
                let collected: String = out.data.iter().collect();
                assert_eq!(collected, "ntuarm");
            }
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_string_array() {
        let strings = StringArray::new(
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let result = rot90_builtin(Value::StringArray(strings), Vec::new()).expect("rot90 str");
        match result {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec!["c", "a", "d", "b"]);
            }
            other => panic!("expected StringArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_non_integer_error() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = rot90_builtin(Value::Tensor(tensor), vec![Value::Num(1.5)]).unwrap_err();
        assert!(err.to_string().contains("K must be an integer"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rot90_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let rotated = rot90_builtin(Value::GpuTensor(handle), Vec::new()).expect("rot90");
            let gathered = test_support::gather(rotated).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![2.0, 1.0, 5.0, 4.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rot90_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let cpu_value =
            rot90_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("cpu rot90");
        let cpu_tensor = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = rot90_builtin(Value::GpuTensor(handle), Vec::new()).expect("gpu rot90");
        let gpu_tensor = test_support::gather(gpu_value).expect("gather");

        assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
        assert_eq!(gpu_tensor.data, cpu_tensor.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
