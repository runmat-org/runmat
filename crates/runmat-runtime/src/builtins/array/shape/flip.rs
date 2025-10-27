//! MATLAB-compatible `flip` builtin with GPU-aware semantics for RunMat.
//!
//! This module implements the `flip` function, mirroring MathWorks MATLAB
//! behaviour for numeric tensors, logical masks, string arrays, complex data,
//! character arrays, and gpuArray handles. It honours dimension vectors,
//! direction keywords such as `'horizontal'`, and gracefully falls back to the
//! host when a registered acceleration provider does not expose a native flip
//! kernel.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "flip"
category: "array/shape"
keywords: ["flip", "reverse", "dimension", "gpu", "horizontal", "vertical"]
summary: "Reverse the order of elements along specific dimensions."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64", "i32", "bool"]
  broadcasting: "none"
  notes: "Attempts to flip tensors directly on the device when providers expose a flip hook; otherwise gathers once, flips on the host, and re-uploads to keep results gpu-resident."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::shape::flip::tests"
  integration: "builtins::array::shape::flip::tests::flip_gpu_roundtrip"
---

# What does the `flip` function do in MATLAB / RunMat?
`flip(A)` reverses the order of elements in `A` along one or more dimensions.
With no dimension argument, RunMat (like MATLAB) flips along the first dimension
whose extent is greater than one. Positive integer dimensions or dimension
vectors allow you to mirror data along any axis, and direction keywords such as
`'horizontal'` or `'vertical'` provide a readable alternative to numeric indices.

## How does the `flip` function behave in MATLAB / RunMat?
- Works for numeric arrays, logical masks, complex arrays, and string or
  character arrays.
- `flip(A)` locates the first non-singleton dimension and reverses that axis.
- `flip(A, dim)` flips along the dimension `dim` (1-based, MATLAB-compatible).
- `flip(A, dims)` accepts a vector of dimensions, flipping each axis in order.
- Dimension vectors must be numeric row or column vectors of positive integers.
- `flip(A, 'horizontal')` and `flip(A, 'vertical')` map to dimensions 2 and 1
  respectively. `'both'` flips along the first two dimensions.
- Dimensions larger than `ndims(A)` are allowed. RunMat pads trailing singleton
  axes so the request becomes a no-op, matching MATLAB semantics.
- Reversing an empty dimension (extent zero) is a no-op.
- gpuArray inputs stay on the device: the runtime calls provider flip hooks
  when available and otherwise gathers, flips, and re-uploads the data.

## `flip` Function GPU Execution Behaviour
When RunMat Accelerate is active, the runtime first checks whether the selected
provider implements the `flip` hook (`ProviderHook::Custom("flip")`). Providers
may specialise the operation using device-side kernels. If the hook is missing
(common for the simple
provider or builds without GPU support), RunMat gathers the tensor once,
performs the flip on the host, and uploads the result back to the device. The
returned value is still a gpuArray, ensuring residency is preserved for
downstream fused expressions.

## Examples of using the `flip` function in MATLAB / RunMat

### Reversing a Row Vector
```matlab
row = 1:5;
rev = flip(row);
```
Expected output:
```matlab
rev = [5 4 3 2 1];
```

### Reversing Matrix Rows (Vertical Flip)
```matlab
A = [1 2 3; 4 5 6; 7 8 9];
B = flip(A);
```
Expected output:
```matlab
B =
     7     8     9
     4     5     6
     1     2     3
```

### Flipping Columns with the Horizontal Direction Keyword
```matlab
A = magic(3);
H = flip(A, 'horizontal');
```
Expected output:
```matlab
H =
     2     7     6
     9     5     1
     4     3     8
```

### Flipping Along Multiple Dimensions
```matlab
T = reshape(1:8, [2 2 2]);
F = flip(T, [1 3]);
```
Expected output:
```matlab
F(:,:,1) =
     3     4
     1     2

F(:,:,2) =
     7     8
     5     6
```

### Flipping Character Data
```matlab
C = ['r','u','n'; 'm','a','t'];
Ct = flip(C, 'horizontal');
```
Expected output:
```matlab
Ct =
    'nur'
    'tam'
```

### Flipping gpuArray Data While Staying on the Device
```matlab
G = gpuArray(rand(4, 4));
V = flip(G, 'vertical');
isgpuarray(V)
```
Expected output:
```matlab
ans = logical 1
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No additional steps are required. RunMat keeps tensors on the GPU across fused
expressions. You can opt-in explicitly with `gpuArray` for MATLAB compatibility,
but the auto-offload planner already tracks residency and minimises host/device
copies even when a provider falls back to the gather → flip → upload pathway.

## FAQ
### Which dimension does `flip(A)` use by default?
The first dimension whose size is greater than one. Column vectors flip along
dimension 1, row vectors flip along dimension 2.

### What happens if I pass a dimension larger than `ndims(A)`?
Nothing changes; RunMat pads missing axes with singleton dimensions, so the
flip becomes a no-op, just like MATLAB.

### Do direction keywords work with any array type?
Yes. `'horizontal'` maps to dimension 2, `'vertical'` to dimension 1, and
`'both'` applies both flips in sequence.

### Can I flip logical or string arrays?
Absolutely. Logical results stay logical, and string arrays preserve string
elements while reordering their positions.

### Does flipping modify gpuArray data in place?
No. The builtin returns a new gpuArray handle. Providers may recycle buffers,
but the original value remains unchanged.

### What about complex inputs?
Complex scalars and arrays are reversed just like real data. Scalars remain
complex scalars after the operation.

## See Also
- [`flipud`](./flipud)
- `flip(A,2)` (MATLAB `fliplr`)
- [`permute`](./permute)
- [`ipermute`](./ipermute)
- [`repmat`](./repmat)
- [`gpuArray`](../../acceleration/gpu/gpuArray)
- [`gather`](../../acceleration/gpu/gather)

## Source & Feedback
- The implementation lives at [`crates/runmat-runtime/src/builtins/array/shape/flip.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/shape/flip.rs)
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "flip",
    op_kind: GpuOpKind::Custom("flip"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("flip")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement flip directly; the runtime falls back to gather→flip→upload when the hook is missing.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "flip",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Flip is a data-reordering boundary; fusion planner treats it as a residency-preserving barrier.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("flip", DOC_MD);

#[runtime_builtin(
    name = "flip",
    category = "array/shape",
    summary = "Reverse the order of elements along specific dimensions.",
    keywords = "flip,reverse,dimension,gpu,horizontal,vertical",
    accel = "custom"
)]
fn flip_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.len() > 1 {
        return Err("flip: too many input arguments".to_string());
    }
    let spec = parse_flip_spec(&rest)?;
    match value {
        Value::Tensor(tensor) => {
            let dims = resolve_dims(&spec, &tensor.shape);
            flip_tensor(tensor, &dims).map(tensor::tensor_into_value)
        }
        Value::LogicalArray(array) => {
            let dims = resolve_dims(&spec, &array.shape);
            flip_logical_array(array, &dims).map(Value::LogicalArray)
        }
        Value::ComplexTensor(ct) => {
            let dims = resolve_dims(&spec, &ct.shape);
            flip_complex_tensor(ct, &dims).map(Value::ComplexTensor)
        }
        Value::Complex(re, im) => {
            let tensor =
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| format!("flip: {e}"))?;
            let dims = resolve_dims(&spec, &tensor.shape);
            flip_complex_tensor(tensor, &dims).map(complex_tensor_into_value)
        }
        Value::StringArray(strings) => {
            let dims = resolve_dims(&spec, &strings.shape);
            flip_string_array(strings, &dims).map(Value::StringArray)
        }
        Value::CharArray(chars) => {
            let dims = resolve_dims(&spec, &[chars.rows, chars.cols]);
            flip_char_array(chars, &dims).map(Value::CharArray)
        }
        Value::String(scalar) => Ok(Value::String(scalar)),
        Value::Num(n) => {
            let tensor = tensor::value_into_tensor_for("flip", Value::Num(n))?;
            let dims = resolve_dims(&spec, &tensor.shape);
            flip_tensor(tensor, &dims).map(tensor::tensor_into_value)
        }
        Value::Int(i) => {
            let tensor = tensor::value_into_tensor_for("flip", Value::Int(i))?;
            let dims = resolve_dims(&spec, &tensor.shape);
            flip_tensor(tensor, &dims).map(tensor::tensor_into_value)
        }
        Value::Bool(flag) => {
            let tensor = tensor::value_into_tensor_for("flip", Value::Bool(flag))?;
            let dims = resolve_dims(&spec, &tensor.shape);
            flip_tensor(tensor, &dims).map(tensor::tensor_into_value)
        }
        Value::GpuTensor(handle) => {
            let dims = resolve_dims(&spec, &handle.shape);
            flip_gpu(handle, &dims)
        }
        Value::Cell(_) => Err("flip: cell arrays are not yet supported".to_string()),
        Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err("flip: unsupported input type".to_string()),
    }
}

#[derive(Clone, Debug)]
enum FlipSpec {
    Default,
    Dims(Vec<usize>),
}

fn parse_flip_spec(args: &[Value]) -> Result<FlipSpec, String> {
    match args.len() {
        0 => Ok(FlipSpec::Default),
        1 => {
            if let Some(direction_dims) = parse_direction(&args[0])? {
                return Ok(FlipSpec::Dims(direction_dims));
            }
            let dims = parse_dims_value(&args[0])?;
            Ok(FlipSpec::Dims(dims))
        }
        _ => unreachable!(),
    }
}

fn parse_direction(value: &Value) -> Result<Option<Vec<usize>>, String> {
    let text_opt = match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            tensor::value_to_string(value)
        }
        _ => None,
    };
    if let Some(text) = text_opt {
        let lowered = text.trim().to_ascii_lowercase();
        let dims = match lowered.as_str() {
            "horizontal" | "left-right" | "leftright" | "lr" | "right-left" | "righthoriz" => {
                vec![2]
            }
            "vertical" | "up-down" | "updown" | "ud" | "down-up" => vec![1],
            "both" => vec![1, 2],
            other => {
                return Err(format!("flip: unknown direction '{other}'"));
            }
        };
        return Ok(Some(dims));
    }
    Ok(None)
}

fn parse_dims_value(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::Tensor(t) => parse_dims_tensor(t),
        Value::LogicalArray(la) => {
            let tensor = tensor::logical_to_tensor(la)
                .map_err(|e| format!("flip: unable to parse dimension vector: {e}"))?;
            parse_dims_tensor(&tensor)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let dim = tensor::parse_dimension(value, "flip")?;
            Ok(vec![dim])
        }
        Value::GpuTensor(_) => Err(
            "flip: dimension argument must be specified on the host (numeric or string)"
                .to_string(),
        ),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                let tmp = Value::StringArray(sa.clone());
                parse_direction(&tmp)?
                    .ok_or_else(|| "flip: dimension vector must be numeric".to_string())
            } else {
                Err("flip: dimension vector must be numeric".to_string())
            }
        }
        Value::String(_) | Value::CharArray(_) => {
            parse_direction(value)?.ok_or_else(|| "flip: unknown direction string".to_string())
        }
        _ => Err("flip: dimension vector must be numeric or a direction string".to_string()),
    }
}

fn parse_dims_tensor(tensor: &Tensor) -> Result<Vec<usize>, String> {
    if !is_vector(&tensor.shape) {
        return Err("flip: dimension vector must be a row or column vector".to_string());
    }
    let mut dims = Vec::with_capacity(tensor.data.len());
    for entry in &tensor.data {
        if !entry.is_finite() {
            return Err("flip: dimension indices must be finite".to_string());
        }
        let rounded = entry.round();
        if (rounded - entry).abs() > f64::EPSILON {
            return Err("flip: dimension indices must be integers".to_string());
        }
        if rounded < 1.0 {
            return Err("flip: dimension indices must be >= 1".to_string());
        }
        dims.push(rounded as usize);
    }
    Ok(dims)
}

fn is_vector(shape: &[usize]) -> bool {
    let mut non_singleton = 0usize;
    for &dim in shape {
        if dim > 1 {
            non_singleton += 1;
        }
        if non_singleton > 1 {
            return false;
        }
    }
    true
}

fn resolve_dims(spec: &FlipSpec, shape: &[usize]) -> Vec<usize> {
    match spec {
        FlipSpec::Default => vec![default_flip_dim(shape)],
        FlipSpec::Dims(dims) => dims.clone(),
    }
}

fn default_flip_dim(shape: &[usize]) -> usize {
    for (idx, &extent) in shape.iter().enumerate() {
        if extent > 1 {
            return idx + 1;
        }
    }
    1
}

pub(crate) fn flip_tensor(tensor: Tensor, dims: &[usize]) -> Result<Tensor, String> {
    if tensor.data.is_empty() || dims.is_empty() {
        return Ok(tensor);
    }
    let data = flip_generic(&tensor.data, &tensor.shape, dims)?;
    Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("flip: {e}"))
}

pub(crate) fn flip_complex_tensor(
    tensor: ComplexTensor,
    dims: &[usize],
) -> Result<ComplexTensor, String> {
    if tensor.data.is_empty() || dims.is_empty() {
        return Ok(tensor);
    }
    let data = flip_generic(&tensor.data, &tensor.shape, dims)?;
    ComplexTensor::new(data, tensor.shape.clone()).map_err(|e| format!("flip: {e}"))
}

pub(crate) fn flip_logical_array(
    array: LogicalArray,
    dims: &[usize],
) -> Result<LogicalArray, String> {
    if array.data.is_empty() || dims.is_empty() {
        return Ok(array);
    }
    let data = flip_generic(&array.data, &array.shape, dims)?;
    LogicalArray::new(data, array.shape.clone()).map_err(|e| format!("flip: {e}"))
}

pub(crate) fn flip_string_array(array: StringArray, dims: &[usize]) -> Result<StringArray, String> {
    if array.data.is_empty() || dims.is_empty() {
        return Ok(array);
    }
    let data = flip_generic(&array.data, &array.shape, dims)?;
    StringArray::new(data, array.shape.clone()).map_err(|e| format!("flip: {e}"))
}

pub(crate) fn flip_char_array(array: CharArray, dims: &[usize]) -> Result<CharArray, String> {
    if array.data.is_empty() || dims.is_empty() {
        return Ok(array);
    }
    let rows = array.rows;
    let cols = array.cols;
    let mut flip_rows = false;
    let mut flip_cols = false;
    for &dim in dims {
        if dim == 0 {
            return Err("flip: dimension must be >= 1".to_string());
        }
        match dim {
            1 => flip_rows = !flip_rows,
            2 => flip_cols = !flip_cols,
            _ => {}
        }
    }
    if !flip_rows && !flip_cols {
        return Ok(array);
    }
    let mut out = vec!['\0'; array.data.len()];
    for row in 0..rows {
        for col in 0..cols {
            let dest_idx = row * cols + col;
            let src_row = if flip_rows { rows - 1 - row } else { row };
            let src_col = if flip_cols { cols - 1 - col } else { col };
            let src_idx = src_row * cols + src_col;
            out[dest_idx] = array.data[src_idx];
        }
    }
    CharArray::new(out, rows, cols).map_err(|e| format!("flip: {e}"))
}

pub(crate) fn flip_gpu(handle: GpuTensorHandle, dims: &[usize]) -> Result<Value, String> {
    if dims.is_empty() {
        return Ok(Value::GpuTensor(handle));
    }
    if dims.iter().any(|&dim| dim == 0) {
        return Err("flip: dimension indices must be >= 1".to_string());
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        let zero_based: Vec<usize> = dims.iter().map(|&d| d - 1).collect();
        if let Ok(out) = provider.flip(&handle, &zero_based) {
            return Ok(Value::GpuTensor(out));
        }
        let host_tensor = gpu_helpers::gather_tensor(&handle)?;
        let flipped = flip_tensor(host_tensor, dims)?;
        let view = HostTensorView {
            data: &flipped.data,
            shape: &flipped.shape,
        };
        provider
            .upload(&view)
            .map(Value::GpuTensor)
            .map_err(|e| format!("flip: {e}"))
    } else {
        let host_tensor = gpu_helpers::gather_tensor(&handle)?;
        flip_tensor(host_tensor, dims).map(tensor::tensor_into_value)
    }
}

fn flip_generic<T: Clone>(data: &[T], shape: &[usize], dims: &[usize]) -> Result<Vec<T>, String> {
    if dims.iter().any(|&dim| dim == 0) {
        return Err("flip: dimension indices must be >= 1".to_string());
    }
    if data.is_empty() {
        return Ok(Vec::new());
    }
    let max_dim = dims.iter().copied().max().unwrap_or(0);
    let mut ext_shape = shape.to_vec();
    if max_dim > ext_shape.len() {
        ext_shape.extend(std::iter::repeat(1).take(max_dim - ext_shape.len()));
    }
    let total: usize = ext_shape.iter().product();
    if total != data.len() {
        return Err("flip: shape does not match data length".to_string());
    }
    let mut flip_flags = vec![false; ext_shape.len()];
    for &dim in dims {
        let axis = dim - 1;
        if axis >= flip_flags.len() {
            continue;
        }
        flip_flags[axis] = !flip_flags[axis];
    }
    if !flip_flags.iter().any(|&flag| flag) {
        return Ok(data.to_vec());
    }
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let mut coords = unravel_index(idx, &ext_shape);
        for (axis, flag) in flip_flags.iter().enumerate() {
            if *flag && ext_shape[axis] > 1 {
                coords[axis] = ext_shape[axis] - 1 - coords[axis];
            }
        }
        let src_idx = ravel_index(&coords, &ext_shape);
        out.push(data[src_idx].clone());
    }
    Ok(out)
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

pub(crate) fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
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
    use runmat_builtins::{CharArray, ComplexTensor, IntValue, LogicalArray, StringArray, Tensor};

    #[test]
    fn flip_vector_defaults_to_first_non_singleton_dim() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let value =
            flip_builtin(Value::Tensor(tensor), Vec::new()).expect("flip row vector default");
        match value {
            Value::Tensor(t) => assert_eq!(t.data, vec![4.0, 3.0, 2.0, 1.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn flip_matrix_vertical_default() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).expect("tensor");
        let value = flip_builtin(Value::Tensor(tensor), Vec::new()).expect("flip matrix");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 2]);
                assert_eq!(t.data, vec![2.0, 4.0, 1.0, 6.0, 3.0, 5.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn flip_horizontal_keyword() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).expect("tensor");
        let value = flip_builtin(Value::Tensor(tensor), vec![Value::from("horizontal")])
            .expect("flip horizontal");
        match value {
            Value::Tensor(t) => assert_eq!(t.data, vec![5.0, 3.0, 6.0, 1.0, 4.0, 2.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn flip_multiple_dimensions() {
        let tensor = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let value = flip_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap(),
            )],
        )
        .expect("flip dims");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2, 2]);
                assert_eq!(t.data, vec![6.0, 5.0, 8.0, 7.0, 2.0, 1.0, 4.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn flip_both_direction_keyword() {
        let tensor = Tensor::new((1..=6).map(|v| v as f64).collect(), vec![3, 2]).unwrap();
        let expected = flip_tensor(tensor.clone(), &[1, 2]).expect("cpu flip");
        let value =
            flip_builtin(Value::Tensor(tensor), vec![Value::from("both")]).expect("flip both");
        match value {
            Value::Tensor(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn flip_char_array_horizontal() {
        let chars = CharArray::new("runmat".chars().collect(), 2, 3).unwrap();
        let value =
            flip_builtin(Value::CharArray(chars), vec![Value::from("horizontal")]).expect("flip");
        match value {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 3);
                let collected: String = out.data.iter().collect();
                assert_eq!(collected, "nurtam");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn flip_direction_accepts_char_array_keyword() {
        let keyword = CharArray::new_row("vertical");
        let tensor = Tensor::new((1..=4).map(|v| v as f64).collect(), vec![2, 2]).unwrap();
        let expected = flip_tensor(tensor.clone(), &[1]).expect("cpu flip");
        let value = flip_builtin(Value::Tensor(tensor), vec![Value::CharArray(keyword)])
            .expect("flip via char");
        match value {
            Value::Tensor(t) => assert_eq!(t.data, expected.data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn flip_logical_array_preserves_type() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let expected = flip_logical_array(logical.clone(), &[2]).expect("cpu logical flip");
        let value = flip_builtin(
            Value::LogicalArray(logical),
            vec![Value::from("horizontal")],
        )
        .expect("flip logical");
        match value {
            Value::LogicalArray(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn flip_complex_tensor_defaults_to_first_dim() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (2.0, -1.0), (3.0, 0.5), (4.0, -0.25)],
            vec![2, 2],
        )
        .unwrap();
        let expected = flip_complex_tensor(tensor.clone(), &[1]).expect("cpu complex flip");
        let value = flip_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("flip complex");
        match value {
            Value::ComplexTensor(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn flip_string_array_vertical() {
        let strings =
            StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).expect("string array");
        let value =
            flip_builtin(Value::StringArray(strings), vec![Value::from("vertical")]).expect("flip");
        match value {
            Value::StringArray(out) => {
                assert_eq!(out.data, vec!["b".to_string(), "a".to_string()])
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn flip_accepts_dimension_vector_tensor() {
        let tensor = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let dims = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let value =
            flip_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("flip dims");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn flip_dimension_tensor_must_be_vector() {
        let tensor = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let dims = Tensor::new((1..=4).map(|v| v as f64).collect(), vec![2, 2]).unwrap();
        let err =
            flip_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect_err("flip fail");
        assert!(err.contains("row or column vector"));
    }

    #[test]
    fn flip_dimensions_beyond_rank_are_noops() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let original = tensor.clone();
        let dims = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let value = flip_builtin(Value::Tensor(tensor), vec![Value::Tensor(dims)]).expect("flip");
        match value {
            Value::Tensor(t) => assert_eq!(t.data, original.data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn flip_rejects_zero_dimension() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = flip_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(0))])
            .expect_err("flip should fail");
        assert!(err.contains("dimension must be >= 1"));
    }

    #[test]
    fn flip_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let cpu =
                flip_tensor(tensor.clone(), &[default_flip_dim(&tensor.shape)]).expect("cpu flip");
            let value = flip_builtin(Value::GpuTensor(handle), Vec::new()).expect("flip gpu");
            let gathered = test_support::gather(value).expect("gather gpu result");
            assert_eq!(gathered.data, cpu.data);
        });
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn flip_doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn flip_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor =
            Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).expect("tensor");
        let cpu = flip_tensor(tensor.clone(), &[1, 3]).expect("cpu flip");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu_value = flip_builtin(
            Value::GpuTensor(handle),
            vec![Value::Tensor(
                Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap(),
            )],
        )
        .expect("flip gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.data, cpu.data);
    }
}
