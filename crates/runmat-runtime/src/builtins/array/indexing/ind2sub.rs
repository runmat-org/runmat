//! MATLAB-compatible `ind2sub` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use super::common::{build_strides, materialize_value, parse_dims, total_elements};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, make_cell, RuntimeError};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "ind2sub",
        builtin_path = "crate::builtins::array::indexing::ind2sub"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "ind2sub"
category: "array/indexing"
keywords: ["ind2sub", "linear index", "subscripts", "column major", "gpu", "nd indexing"]
summary: "Convert MATLAB column-major linear indices into per-dimension subscript arrays."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "When the active provider implements the `ind2sub` hook (WGPU today), conversions run entirely on the GPU. Other providers fall back to the host and re-upload the results."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::indexing::ind2sub::tests"
  integration: "builtins::array::indexing::ind2sub::tests::ind2sub_gpu_roundtrip"
---

# What does the `ind2sub` function do in MATLAB / RunMat?
`ind2sub(siz, idx)` converts MATLAB's 1-based column-major linear indices back into the individual subscripts for each dimension specified in `siz`.

## How does the `ind2sub` function behave in MATLAB / RunMat?
- The size vector `siz` supplies one extent per dimension. Each entry must be a positive integer.
- `idx` can be a scalar or an array of any shape; every element must be a positive integer within `prod(siz)`.
- The result is always a 1×N cell array (N is `numel(siz)`). Each cell contains a double array that matches the shape of `idx`.
- Non-integer, complex, NaN, Inf, or out-of-range indices raise MATLAB-compatible errors.
- Empty inputs produce empty outputs with matching shapes.
- When called with multiple outputs (`[i,j,k] = ind2sub(...)`) RunMat unpacks the cell array automatically, mirroring MATLAB semantics.

## `ind2sub` Function GPU Execution Behaviour
When a WGPU-backed provider is active, `ind2sub` executes entirely on the GPU. The shader mirrors MATLAB's validation rules, rejecting non-integer, non-positive, or out-of-range indices with the same diagnostic messages as the CPU path. Providers that do not yet implement the hook fall back to the host implementation; after computing the subscripts RunMat uploads the results back to the active provider so downstream fused kernels continue operating on device-resident data.

## Examples of using the `ind2sub` function in MATLAB / RunMat

### Recovering row and column subscripts from a matrix index

```matlab
[row, col] = ind2sub([3 4], 8);
```

Expected output:

```matlab
row = 2;
col = 3;
```

### Extracting multiple matrix indices at once

```matlab
idx = [7 8 9];
[rows, cols] = ind2sub([3 5], idx);
```

Expected output:

```matlab
rows = [1 2 3];
cols = [3 3 3];
```

### Converting indices for a 3-D volume

```matlab
idx = [3 11];
[r, c, p] = ind2sub([2 3 4], idx);
```

Expected output:
```matlab
r = [1 1];
c = [2 3];
p = [1 2];
```

### Keeping indices on the GPU

```matlab
rows = gpuArray([1; 2; 3]);
cols = gpuArray([4; 4; 4]);
lin = sub2ind([3 4], rows, cols);
subs = ind2sub([3 4], lin);  % subs{1} and subs{2} remain gpuArray values
class(subs{1})
class(subs{2})
```

Expected output:

```matlab
ans =
    'gpuArray'

ans =
    'gpuArray'
```

### Using a single output cell for flexible unpacking

```matlab
subs = ind2sub(size(magic(4)), 6:9);
% Access with subs{1}, subs{2}, ...
```

Expected output:

```matlab
subs =
  1×2 cell array
    {1×4 double}    {1×4 double}
```

### Validating index ranges before reshaping

```matlab
idx = 1:numel(A);
[i, j] = ind2sub(size(A), idx);
B = accumarray([i(:) j(:)], A(:));
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You typically do **not** need to convert tensors manually. When the active provider implements the `ind2sub` hook (WGPU today), the entire conversion stays on the GPU. Otherwise RunMat gathers the inputs, performs validation on the host, and uploads the resulting subscript arrays back to the device so downstream kernels or fusion plans can continue using them without additional `gather` calls.

## FAQ

### What types does `ind2sub` accept for `idx`?
Numeric and logical inputs are accepted. Logical arrays are treated as double precision (with `true → 1`, `false → 0`); complex values are rejected.

### Can `siz` contain zeros?
No. Each element of `siz` must be a positive integer, matching MATLAB behaviour.

### How are errors reported for invalid indices?
Indices that are non-integer, non-positive, or exceed `prod(siz)` raise errors matching MATLAB's wording (e.g., "Index exceeds number of array elements. Index must not exceed …").

### What shape do the output arrays have?
Each output array matches the shape of `idx`. Scalars produce scalar doubles; vectors remain vectors; higher-dimensional shapes are preserved.

### Does `ind2sub` support GPU arrays?
Yes. With the WGPU provider the conversion happens entirely on the GPU, including validation. Other providers gather the data to the host, compute the subscripts, and upload them back to the device automatically.

### Can I request fewer outputs than dimensions?
Yes. In a multi-output context RunMat provides as many outputs as requested in order, just like MATLAB. Any additional dimensions are still available inside the single-output cell if you need them.

### How does `ind2sub` relate to `sub2ind`?
They are inverse operations: `sub2ind` turns subscripts into linear indices, while `ind2sub` recovers subscripts from those linear indices.

### Does `ind2sub` allocate full copies of the outputs?
Yes. Each subscript array is materialised as a dense double array matching the shape of `idx`, mirroring MATLAB semantics.

### What happens with empty inputs?
Empty index arrays produce empty subscript arrays with the same shape.

### Can I use `ind2sub` with more dimensions than two?
Definitely—`ind2sub` works for any number of dimensions represented in `siz`.

## See Also
[sub2ind](./sub2ind), [size](./size), [find](./find), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `ind2sub` function is available at: [`crates/runmat-runtime/src/builtins/array/indexing/ind2sub.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/indexing/ind2sub.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::indexing::ind2sub")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ind2sub",
    op_kind: GpuOpKind::Custom("indexing"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("ind2sub")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "WGPU provider executes `ind2sub` entirely on-device; other providers fall back to the host implementation and re-upload results to preserve residency.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::indexing::ind2sub")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ind2sub",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Index conversion is eager and does not participate in fusion today.",
};

#[runtime_builtin(
    name = "ind2sub",
    category = "array/indexing",
    summary = "Convert MATLAB column-major linear indices into per-dimension subscript arrays.",
    keywords = "ind2sub,linear index,subscripts,column major,gpu indexing",
    accel = "custom",
    builtin_path = "crate::builtins::array::indexing::ind2sub"
)]
async fn ind2sub_builtin(dims_val: Value, indices_val: Value) -> crate::BuiltinResult<Value> {
    let (dims_value, dims_was_gpu) = materialize_value(dims_val, "ind2sub").await?;
    let dims = parse_dims(&dims_value, "ind2sub").await?;
    if dims.is_empty() {
        return Err(ind2sub_error("Size vector must have at least one element."));
    }

    let total = total_elements(&dims, "ind2sub")?;
    let strides = build_strides(&dims, "ind2sub")?;

    if let Some(result) = try_gpu_ind2sub(&dims, &strides, total, &indices_val)? {
        return Ok(result);
    }

    let (indices_value, indices_was_gpu) = materialize_value(indices_val, "ind2sub").await?;
    let indices_tensor = tensor::value_into_tensor_for("ind2sub", indices_value)
        .map_err(|message| ind2sub_error(message))?;

    let subscripts = compute_subscripts(&dims, total, &strides, &indices_tensor)?;

    let want_gpu = (dims_was_gpu || indices_was_gpu) && runmat_accelerate_api::provider().is_some();

    let mut outputs: Vec<Value> = Vec::with_capacity(dims.len());
    for tensor in subscripts {
        if want_gpu {
            #[cfg(all(test, feature = "wgpu"))]
            {
                if runmat_accelerate_api::provider().is_none() {
                    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                    );
                }
            }
            if let Some(provider) = runmat_accelerate_api::provider() {
                let view = HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                };
                if let Ok(handle) = provider.upload(&view) {
                    outputs.push(Value::GpuTensor(handle));
                    continue;
                }
            }
        }
        outputs.push(tensor::tensor_into_value(tensor));
    }

    make_cell(outputs, 1, dims.len()).map_err(|message| ind2sub_error(message))
}

fn try_gpu_ind2sub(
    dims: &[usize],
    strides: &[usize],
    total: usize,
    indices: &Value,
) -> crate::BuiltinResult<Option<Value>> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if let Value::GpuTensor(h) = indices {
            if h.device_id != 0 {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
    }
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };
    if !provider.supports_ind2sub() {
        return Ok(None);
    }
    let handle = match indices {
        Value::GpuTensor(handle) => handle,
        _ => return Ok(None),
    };
    if dims.len() != strides.len() {
        return Err(ind2sub_error("Size vector must have at least one element."));
    }
    if dims.iter().any(|&d| d > u32::MAX as usize)
        || strides.iter().any(|&s| s > u32::MAX as usize)
        || total > u32::MAX as usize
    {
        return Ok(None);
    }
    let len = if handle.shape.is_empty() {
        1usize
    } else {
        handle.shape.iter().copied().product()
    };
    if total == 0 && len > 0 {
        return Err(ind2sub_error(
            "Index exceeds number of array elements. Index must not exceed 0.",
        ));
    }
    if len > u32::MAX as usize {
        return Ok(None);
    }
    let output_shape = if handle.shape.is_empty() {
        vec![len, 1]
    } else {
        handle.shape.clone()
    };
    match provider.ind2sub(dims, strides, handle, total, len, &output_shape) {
        Ok(handles) => {
            if handles.len() != dims.len() {
                return Err(ind2sub_error(
                    "ind2sub: provider returned an unexpected number of outputs.",
                ));
            }
            let values: Vec<Value> = handles.into_iter().map(Value::GpuTensor).collect();
            make_cell(values, 1, dims.len())
                .map(Some)
                .map_err(|message| ind2sub_error(message))
        }
        Err(err) => Err(ind2sub_error(err.to_string())),
    }
}

fn compute_subscripts(
    dims: &[usize],
    total: usize,
    strides: &[usize],
    indices: &Tensor,
) -> crate::BuiltinResult<Vec<Tensor>> {
    if strides.len() != dims.len() {
        return Err(ind2sub_error("Size vector must have at least one element."));
    }

    let len = indices.data.len();
    let mut outputs: Vec<Vec<f64>> = dims.iter().map(|_| Vec::with_capacity(len)).collect();

    for &value in &indices.data {
        let idx = coerce_linear_index(value, total)?;
        let zero_based = idx - 1;
        for (dim_index, (&dim, &stride)) in dims.iter().zip(strides.iter()).enumerate() {
            let coord = ((zero_based / stride) % dim) + 1;
            outputs[dim_index].push(coord as f64);
        }
    }

    let output_shape = if indices.shape.is_empty() {
        vec![len, 1]
    } else {
        indices.shape.clone()
    };

    let mut tensors = Vec::with_capacity(dims.len());
    for data in outputs {
        let tensor = Tensor::new(data, output_shape.clone())
            .map_err(|e| ind2sub_error(format!("ind2sub: {e}")))?;
        tensors.push(tensor);
    }
    Ok(tensors)
}

fn coerce_linear_index(value: f64, max_index: usize) -> crate::BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(ind2sub_error("Linear indices must be positive integers."));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(ind2sub_error("Linear indices must be positive integers."));
    }
    if rounded < 1.0 {
        return Err(ind2sub_error("Linear indices must be positive integers."));
    }
    if rounded > usize::MAX as f64 {
        return Err(ind2sub_error(
            "Index exceeds maximum supported size for this platform.",
        ));
    }
    let coerced = rounded as usize;
    if coerced > max_index {
        return Err(ind2sub_error(format!(
            "Index exceeds number of array elements. Index must not exceed {}.",
            max_index
        )));
    }
    Ok(coerced)
}

fn ind2sub_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("ind2sub").build()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{Tensor, Value};

    fn ind2sub_builtin(dims_val: Value, indices_val: Value) -> crate::BuiltinResult<Value> {
        block_on(super::ind2sub_builtin(dims_val, indices_val))
    }

    fn cell_to_vec(cell: &runmat_builtins::CellArray) -> Vec<Value> {
        cell.data.iter().map(|ptr| (**ptr).clone()).collect()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn recovers_tensor_indices() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result = ind2sub_builtin(Value::Tensor(dims), Value::Num(8.0)).unwrap();
        match result {
            Value::Cell(cell) => {
                let values = cell_to_vec(&cell);
                assert_eq!(values.len(), 2);
                assert_eq!(values[0], Value::Num(2.0));
                assert_eq!(values[1], Value::Num(3.0));
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn handles_vector_indices() {
        let dims = Tensor::new(vec![3.0, 5.0], vec![1, 2]).unwrap();
        let idx = Tensor::new(vec![7.0, 8.0, 9.0], vec![1, 3]).unwrap();
        let result =
            ind2sub_builtin(Value::Tensor(dims), Value::Tensor(idx)).expect("ind2sub result");
        match result {
            Value::Cell(cell) => {
                let values = cell_to_vec(&cell);
                assert_eq!(values.len(), 2);
                match &values[0] {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![1, 3]);
                        assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
                    }
                    other => panic!("expected tensor rows, got {other:?}"),
                }
                match &values[1] {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![1, 3]);
                        assert_eq!(t.data, vec![3.0, 3.0, 3.0]);
                    }
                    other => panic!("expected tensor cols, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn recovers_three_dimensional_indices() {
        let dims = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let idx = Tensor::new(vec![3.0, 11.0], vec![1, 2]).unwrap();
        let result =
            ind2sub_builtin(Value::Tensor(dims), Value::Tensor(idx)).expect("ind2sub result");
        if let Value::Cell(cell) = result {
            let values = cell_to_vec(&cell);
            assert_eq!(values.len(), 3);
            assert_eq!(
                values[0],
                Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap())
            );
            assert_eq!(
                values[1],
                Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap())
            );
            assert_eq!(
                values[2],
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap())
            );
        } else {
            panic!("expected cell output");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn errors_on_out_of_range_index() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err =
            ind2sub_builtin(Value::Tensor(dims), Value::Num(13.0)).expect_err("expected failure");
        assert!(
            err.message()
                .contains("Index exceeds number of array elements"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn errors_on_zero_index() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err =
            ind2sub_builtin(Value::Tensor(dims), Value::Num(0.0)).expect_err("expected failure");
        assert!(
            err.contains("Linear indices must be positive integers"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn errors_on_fractional_index() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err =
            ind2sub_builtin(Value::Tensor(dims), Value::Num(2.5)).expect_err("expected failure");
        assert!(
            err.contains("Linear indices must be positive integers"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn errors_on_invalid_size_elements() {
        let dims = Tensor::new(vec![3.5, 4.0], vec![1, 2]).unwrap();
        let err = ind2sub_builtin(Value::Tensor(dims), Value::Num(5.0)).expect_err("expected fail");
        assert!(
            err.to_string()
                .contains("Size arguments must be positive integers"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ind2sub_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
            let idx_tensor = Tensor::new(vec![10.0, 11.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &idx_tensor.data,
                shape: &idx_tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload indices");
            let result = ind2sub_builtin(Value::Tensor(dims), Value::GpuTensor(handle)).unwrap();
            match result {
                Value::Cell(cell) => {
                    let values = cell_to_vec(&cell);
                    assert_eq!(values.len(), 2);
                    match &values[0] {
                        Value::GpuTensor(_) => {}
                        other => panic!("expected gpu tensor output, got {other:?}"),
                    }
                    match &values[1] {
                        Value::GpuTensor(_) => {}
                        other => panic!("expected gpu tensor output, got {other:?}"),
                    }
                    let rows = test_support::gather(values[0].clone()).expect("gather rows");
                    assert_eq!(rows.shape, vec![2, 1]);
                    assert_eq!(rows.data, vec![1.0, 2.0]);
                    let cols = test_support::gather(values[1].clone()).expect("gather cols");
                    assert_eq!(cols.shape, vec![2, 1]);
                    assert_eq!(cols.data, vec![4.0, 4.0]);
                }
                other => panic!("expected cell output, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ind2sub_wgpu_matches_cpu() {
        let provider_init = std::panic::catch_unwind(|| {
            runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            )
        });
        if let Ok(Ok(_)) = provider_init {
            // provider successfully registered
        } else {
            return;
        }

        let dims_tensor = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let idx_tensor = Tensor::new(vec![7.0, 8.0, 9.0], vec![1, 3]).unwrap();

        let cpu = ind2sub_builtin(
            Value::Tensor(dims_tensor.clone()),
            Value::Tensor(idx_tensor.clone()),
        )
        .expect("cpu ind2sub");

        let provider = runmat_accelerate_api::provider().unwrap();
        let view = HostTensorView {
            data: &idx_tensor.data,
            shape: &idx_tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload indices");

        let gpu = ind2sub_builtin(Value::Tensor(dims_tensor), Value::GpuTensor(handle))
            .expect("gpu ind2sub");

        let cpu_values = match cpu {
            Value::Cell(cell) => cell_to_vec(&cell),
            other => panic!("expected cell output, got {other:?}"),
        };
        let gpu_values = match gpu {
            Value::Cell(cell) => cell_to_vec(&cell),
            other => panic!("expected cell output, got {other:?}"),
        };

        assert_eq!(cpu_values.len(), gpu_values.len());

        for (cpu_val, gpu_val) in cpu_values.iter().zip(gpu_values.iter()) {
            let host_cpu = test_support::gather(cpu_val.clone()).expect("gather cpu");
            let host_gpu = test_support::gather(gpu_val.clone()).expect("gather gpu");
            assert_eq!(host_cpu.shape, host_gpu.shape);
            assert_eq!(host_cpu.data, host_gpu.data);
        }
    }
}
