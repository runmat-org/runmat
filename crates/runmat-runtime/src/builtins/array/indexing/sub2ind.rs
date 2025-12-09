//! MATLAB-compatible `sub2ind` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use super::common::{build_strides, materialize_value, parse_dims};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "sub2ind")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "sub2ind"
category: "array/indexing"
keywords: ["sub2ind", "linear index", "column major", "gpu indexing", "nd indexing"]
summary: "Convert N-D subscripts into MATLAB-style column-major linear indices."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "When a GPU provider exposes the `sub2ind` hook (WGPU today), the conversion runs on the device with bounds and integrality checks; other providers fall back to the host implementation and reupload the result."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::indexing::sub2ind::tests"
  integration: "builtins::array::indexing::sub2ind::tests::sub2ind_gpu_roundtrip"
---

# What does the `sub2ind` function do in MATLAB / RunMat?
`sub2ind(sz, s1, s2, ...)` converts row/column (or higher-dimensional) subscripts into MATLAB's column-major linear indexing form. The size vector `sz` defines the extents of the target array, and you must supply one subscript array per dimension.

## How does the `sub2ind` function behave in MATLAB / RunMat?
- Subscripts can be scalars or arrays. When arrays are provided, they must share the same size. Scalars broadcast to that common shape.
- All subscripts must be positive integers within the corresponding dimension's range.
- The size vector can be a row or column vector. Each element must be a positive integer.
- Complex, NaN, or infinite values are rejected.
- The result uses the same shape as the subscript arrays. Scalars produce a scalar double.
- When any input is a GPU tensor, RunMat computes on the host (to reuse integer semantics) and uploads the resulting indices back to the GPU so fusion and downstream kernels keep operating on device.

## `sub2ind` Function GPU Execution Behaviour
When a WGPU-backed provider is active, `sub2ind` executes entirely on the GPU. The shader mirrors MATLAB's validation rules: it rejects non-finite values, non-integer subscripts, and out-of-range indices, surfacing the same diagnostic messages as the CPU path. Providers that do not yet implement the hook fall back to the host implementation; after the indices are computed they are uploaded back to the active provider so downstream fused kernels continue operating on device data.

## Examples of using the `sub2ind` function in MATLAB / RunMat

### Converting a single matrix subscript to a linear index

```matlab
idx = sub2ind([3 4], 2, 3);
```

Expected output:

```matlab
idx = 8;
```

### Mapping multiple subscripts into one-dimensional indices

```matlab
rows = [1; 2; 3];
cols = [3; 3; 3];
idx = sub2ind([3 5], rows, cols);
```

Expected output:

```matlab
idx =
     7
     8
     9
```

### Handling higher-dimensional array subscripts

```matlab
row = [1 1];
col = [2 3];
page = [1 2];
idx = sub2ind([2 3 4], row, col, page);
```

Expected output:

```matlab
idx = [3 11];
```

### Broadcasting scalar subscripts across array inputs

```matlab
rows = [1 2 3];
idx = sub2ind([3 4], rows, 4);
```

Expected output:

```matlab
idx = [10 11 12];
```

### Retaining GPU residency for batched index conversions

```matlab
rows = gpuArray((1:100)');
cols = gpuArray(ones(100, 1) * 4);
idx = sub2ind([100 4], rows, cols);
```

Expected behavior:

```matlab
% idx remains a gpuArray containing the column-major indices.
disp(gather(idx(1:5)));
% Output:
%    301
%    302
%    303
%    304
%    305
```

### Detecting invalid out-of-range subscripts

```matlab
try
    idx = sub2ind([3 4], 4, 1);
catch ME
    disp(ME.message);
end
```

Expected output:

```matlab
Index exceeds the number of rows in dimension 1.
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You typically do **not** need to call `gpuArray` yourself. When the active provider implements the `sub2ind` hook (WGPU today), the entire conversion runs on the GPU and returns a device tensor. If no provider is available, or the provider lacks the hook, RunMat falls back to the host implementation and uploads the resulting indices back to the GPU so residency is maintained automatically.

## FAQ

### What data types does `sub2ind` accept?
Numeric and logical inputs are accepted. Logical values are converted to doubles before validation. Complex, NaN, and infinite values are rejected with a descriptive error.

### Can the size vector contain zeros?
No. Every dimension size must be a positive integer. This matches MATLAB's behavior for index conversion.

### Do subscripts have to be the same size?
Yes. All non-scalar subscripts must share the same size (shape). Scalars broadcast to that common shape.

### What happens when subscripts are out of range?
`sub2ind` throws an error explaining which dimension failed the bounds check. This mirrors MATLAB's run-time error.

### Does the function support GPU arrays?
Yes. With the WGPU provider the conversion happens entirely on device, including validation. Other providers gather the data to the host, compute the indices, and upload them back to the device automatically.

### Are fractional subscripts rounded?
No. Non-integer, NaN, or infinite subscripts raise an error.

### How is the linear index computed?
The output uses MATLAB's column-major convention: `1 + sum((s_k - 1) * stride_k)` where `stride_k` is the product of the preceding dimensions.

### Can I call `sub2ind` with more subscripts than dimensions?
No. You must pass exactly one subscript per dimension listed in the size vector.

### What about empty outputs?
If the subscript arrays are empty, `sub2ind` returns an empty double array with the same shape.

### Does `sub2ind` change the orientation of row/column vectors?
No. The output preserves the orientation (shape) of the subscript arrays, so row vectors stay row vectors and column vectors stay column vectors.

## See Also
[ind2sub](./ind2sub), [find](./find), [size](../../introspection/size), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `sub2ind` function is available at: [`crates/runmat-runtime/src/builtins/array/indexing/sub2ind.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/indexing/sub2ind.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sub2ind",
    op_kind: GpuOpKind::Custom("indexing"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("sub2ind")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers can implement the custom `sub2ind` hook to execute on device; runtimes fall back to host computation otherwise.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sub2ind",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Index conversion executes eagerly on the host; fusion does not apply.",
};

#[runtime_builtin(
    name = "sub2ind",
    category = "array/indexing",
    summary = "Convert N-D subscripts into MATLAB-style column-major linear indices.",
    keywords = "sub2ind,linear index,column major,gpu indexing",
    accel = "custom"
)]
fn sub2ind_builtin(dims_val: Value, rest: Vec<Value>) -> Result<Value, String> {
    let (dims_value, dims_was_gpu) = materialize_value(dims_val)?;
    let dims = parse_dims(&dims_value)?;
    if dims.is_empty() {
        return Err("Size vector must have at least one element.".to_string());
    }

    if rest.len() != dims.len() {
        return Err("The number of subscripts supplied must equal the number of dimensions in the size vector.".to_string());
    }

    if let Some(value) = try_gpu_sub2ind(&dims, &rest)? {
        return Ok(value);
    }

    let mut saw_gpu = dims_was_gpu;
    let mut subscripts: Vec<Tensor> = Vec::with_capacity(rest.len());
    for value in rest {
        let (materialised, was_gpu) = materialize_value(value)?;
        saw_gpu |= was_gpu;
        let tensor = tensor::value_into_tensor_for("sub2ind", materialised)?;
        subscripts.push(tensor);
    }

    let (result_data, result_shape) = compute_indices(&dims, &subscripts)?;
    let want_gpu_output = saw_gpu && runmat_accelerate_api::provider().is_some();

    if want_gpu_output {
        #[cfg(all(test, feature = "wgpu"))]
        {
            if runmat_accelerate_api::provider().is_none() {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
        let shape = result_shape.clone().unwrap_or_else(|| vec![1, 1]);
        if let Some(provider) = runmat_accelerate_api::provider() {
            let view = HostTensorView {
                data: &result_data,
                shape: &shape,
            };
            if let Ok(handle) = provider.upload(&view) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }

    build_host_value(result_data, result_shape)
}

fn try_gpu_sub2ind(dims: &[usize], subs: &[Value]) -> Result<Option<Value>, String> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if subs
            .iter()
            .any(|v| matches!(v, Value::GpuTensor(h) if h.device_id != 0))
        {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };
    if !subs
        .iter()
        .all(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Ok(None);
    }
    if dims.is_empty() {
        return Ok(None);
    }

    let mut handles: Vec<&GpuTensorHandle> = Vec::with_capacity(subs.len());
    for value in subs {
        if let Value::GpuTensor(handle) = value {
            handles.push(handle);
        }
    }

    if handles.len() != dims.len() {
        return Err("The number of subscripts supplied must equal the number of dimensions in the size vector.".to_string());
    }

    let mut scalar_mask: Vec<bool> = Vec::with_capacity(handles.len());
    let mut target_shape: Option<Vec<usize>> = None;
    let mut result_len: usize = 1;
    let mut saw_non_scalar = false;

    for handle in &handles {
        let len = tensor::element_count(&handle.shape);
        let is_scalar = len == 1;
        scalar_mask.push(is_scalar);
        if !is_scalar {
            saw_non_scalar = true;
            if let Some(existing) = &target_shape {
                if existing != &handle.shape {
                    return Err("Subscript inputs must have the same size.".to_string());
                }
            } else {
                target_shape = Some(handle.shape.clone());
                result_len = len;
            }
        }
    }

    if !saw_non_scalar {
        target_shape = Some(vec![1, 1]);
        result_len = 1;
    } else if let Some(shape) = &target_shape {
        result_len = tensor::element_count(shape);
    }

    let strides = build_strides(dims)?;
    if dims.iter().any(|&d| d > u32::MAX as usize)
        || strides.iter().any(|&s| s > u32::MAX as usize)
        || result_len > u32::MAX as usize
    {
        return Ok(None);
    }

    let output_shape = target_shape.clone().unwrap_or_else(|| vec![1, 1]);
    match provider.sub2ind(
        dims,
        &strides,
        &handles,
        &scalar_mask,
        result_len,
        &output_shape,
    ) {
        Ok(handle) => Ok(Some(Value::GpuTensor(handle))),
        Err(err) => Err(err.to_string()),
    }
}

fn compute_indices(
    dims: &[usize],
    subscripts: &[Tensor],
) -> Result<(Vec<f64>, Option<Vec<usize>>), String> {
    let mut target_shape: Option<Vec<usize>> = None;
    let mut result_len: usize = 1;
    let mut has_non_scalar = false;

    for tensor in subscripts {
        if tensor.data.len() != 1 {
            has_non_scalar = true;
            if let Some(shape) = &target_shape {
                if &tensor.shape != shape {
                    return Err("Subscript inputs must have the same size.".to_string());
                }
            } else {
                target_shape = Some(tensor.shape.clone());
                result_len = tensor.data.len();
            }
        }
    }

    if !has_non_scalar {
        // All scalars -> scalar output
        target_shape = Some(vec![1, 1]);
        result_len = 1;
    }

    if result_len == 0 {
        return Ok((Vec::new(), target_shape));
    }

    let strides = build_strides(dims)?;
    let mut output = Vec::with_capacity(result_len);

    for idx in 0..result_len {
        let mut offset: usize = 0;
        for (dim_index, (&dim, tensor)) in dims.iter().zip(subscripts.iter()).enumerate() {
            let raw = subscript_value(tensor, idx);
            let coerced = coerce_subscript(raw, dim_index + 1, dim)?;
            let term = coerced
                .checked_sub(1)
                .and_then(|v| v.checked_mul(strides[dim_index]))
                .ok_or_else(|| "Index exceeds array dimensions.".to_string())?;
            offset = offset
                .checked_add(term)
                .ok_or_else(|| "Index exceeds array dimensions.".to_string())?;
        }
        output.push((offset + 1) as f64);
    }

    Ok((output, target_shape))
}

fn subscript_value(tensor: &Tensor, idx: usize) -> f64 {
    if tensor.data.len() == 1 {
        tensor.data[0]
    } else {
        tensor.data[idx]
    }
}

fn coerce_subscript(value: f64, dim_number: usize, dim_size: usize) -> Result<usize, String> {
    if !value.is_finite() {
        return Err(
            "Subscript indices must either be real positive integers or logicals.".to_string(),
        );
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(
            "Subscript indices must either be real positive integers or logicals.".to_string(),
        );
    }
    if rounded < 1.0 {
        return Err(
            "Subscript indices must either be real positive integers or logicals.".to_string(),
        );
    }
    if rounded > dim_size as f64 {
        return Err(dimension_bounds_error(dim_number));
    }
    Ok(rounded as usize)
}

fn dimension_bounds_error(dim_number: usize) -> String {
    match dim_number {
        1 => format!("Index exceeds the number of rows in dimension {dim_number}."),
        2 => format!("Index exceeds the number of columns in dimension {dim_number}."),
        3 => format!("Index exceeds the number of pages in dimension {dim_number}."),
        _ => "Index exceeds array dimensions.".to_string(),
    }
}

fn build_host_value(data: Vec<f64>, shape: Option<Vec<usize>>) -> Result<Value, String> {
    let shape = shape.unwrap_or_else(|| vec![1, 1]);
    if data.len() == 1 && tensor::element_count(&shape) == 1 {
        Ok(Value::Num(data[0]))
    } else {
        let tensor = Tensor::new(data, shape)
            .map_err(|e| format!("Unable to construct sub2ind output: {e}"))?;
        Ok(Value::Tensor(tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{IntValue, Tensor, Value};

    #[test]
    fn converts_scalar_indices() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result =
            sub2ind_builtin(Value::Tensor(dims), vec![Value::Num(2.0), Value::Num(3.0)]).unwrap();
        assert_eq!(result, Value::Num(8.0));
    }

    #[test]
    fn broadcasts_scalars_over_vectors() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let rows = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = sub2ind_builtin(
            Value::Tensor(dims),
            vec![Value::Tensor(rows), Value::Num(4.0)],
        )
        .unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.data, vec![10.0, 11.0, 12.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn handles_three_dimensions() {
        let dims = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let row = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let col = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let page = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = sub2ind_builtin(
            Value::Tensor(dims),
            vec![Value::Tensor(row), Value::Tensor(col), Value::Tensor(page)],
        )
        .unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![3.0, 11.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn rejects_out_of_range_subscripts() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err = sub2ind_builtin(Value::Tensor(dims), vec![Value::Num(4.0), Value::Num(1.0)])
            .unwrap_err();
        assert!(
            err.contains("Index exceeds"),
            "expected index bounds error, got {err}"
        );
    }

    #[test]
    fn rejects_shape_mismatch() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let rows = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let cols = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = sub2ind_builtin(
            Value::Tensor(dims),
            vec![Value::Tensor(rows), Value::Tensor(cols)],
        )
        .unwrap_err();
        assert!(
            err.contains("same size"),
            "expected size mismatch error, got {err}"
        );
    }

    #[test]
    fn rejects_non_integer_subscripts() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err = sub2ind_builtin(Value::Tensor(dims), vec![Value::Num(1.5), Value::Num(1.0)])
            .unwrap_err();
        assert!(
            err.contains("real positive integers"),
            "expected integer coercion error, got {err}"
        );
    }

    #[test]
    fn accepts_integer_value_variants() {
        let dims = Value::Tensor(Tensor::new(vec![3.0], vec![1, 1]).unwrap());
        let result = sub2ind_builtin(dims, vec![Value::Int(IntValue::I32(2))]).expect("sub2ind");
        assert_eq!(result, Value::Num(2.0));
    }

    #[test]
    fn sub2ind_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
            let rows = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let cols = Tensor::new(vec![4.0, 4.0, 4.0], vec![3, 1]).unwrap();

            let dims_handle = provider
                .upload(&HostTensorView {
                    data: &dims.data,
                    shape: &dims.shape,
                })
                .expect("upload dims");
            let rows_handle = provider
                .upload(&HostTensorView {
                    data: &rows.data,
                    shape: &rows.shape,
                })
                .expect("upload rows");
            let cols_handle = provider
                .upload(&HostTensorView {
                    data: &cols.data,
                    shape: &cols.shape,
                })
                .expect("upload cols");

            let result = sub2ind_builtin(
                Value::GpuTensor(dims_handle),
                vec![Value::GpuTensor(rows_handle), Value::GpuTensor(cols_handle)],
            )
            .expect("sub2ind");

            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).unwrap();
                    assert_eq!(gathered.shape, vec![3, 1]);
                    assert_eq!(gathered.data, vec![10.0, 11.0, 12.0]);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn sub2ind_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            panic!("wgpu provider not available");
        };

        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let rows = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let cols = Tensor::new(vec![4.0, 4.0, 4.0], vec![3, 1]).unwrap();

        let cpu = sub2ind_builtin(
            Value::Tensor(dims.clone()),
            vec![Value::Tensor(rows.clone()), Value::Tensor(cols.clone())],
        )
        .expect("cpu sub2ind");

        let rows_handle = provider
            .upload(&HostTensorView {
                data: &rows.data,
                shape: &rows.shape,
            })
            .expect("upload rows");
        let cols_handle = provider
            .upload(&HostTensorView {
                data: &cols.data,
                shape: &cols.shape,
            })
            .expect("upload cols");

        let result = sub2ind_builtin(
            Value::Tensor(dims),
            vec![Value::GpuTensor(rows_handle), Value::GpuTensor(cols_handle)],
        )
        .expect("wgpu sub2ind");

        let gathered = test_support::gather(result).expect("gather");
        let expected = match cpu {
            Value::Tensor(t) => t,
            Value::Num(v) => Tensor::new(vec![v], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu result {other:?}"),
        };
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
