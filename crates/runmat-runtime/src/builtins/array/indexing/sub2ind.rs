//! MATLAB-compatible `sub2ind` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::HostTensorView;
#[cfg(not(target_arch = "wasm32"))]
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use super::common::{build_strides, materialize_value, parse_dims};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::indexing::sub2ind")]
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::indexing::sub2ind")]
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
    accel = "custom",
    builtin_path = "crate::builtins::array::indexing::sub2ind"
)]
async fn sub2ind_builtin(dims_val: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (dims_value, dims_was_gpu) = materialize_value(dims_val, "sub2ind").await?;
    let dims = parse_dims(&dims_value, "sub2ind").await?;
    if dims.is_empty() {
        return Err(sub2ind_error("Size vector must have at least one element."));
    }

    if rest.len() != dims.len() {
        return Err(sub2ind_error(
            "The number of subscripts supplied must equal the number of dimensions in the size vector.",
        ));
    }

    if let Some(value) = try_gpu_sub2ind(&dims, &rest)? {
        return Ok(value);
    }

    let mut saw_gpu = dims_was_gpu;
    let mut subscripts: Vec<Tensor> = Vec::with_capacity(rest.len());
    for value in rest {
        let (materialised, was_gpu) = materialize_value(value, "sub2ind").await?;
        saw_gpu |= was_gpu;
        let tensor = tensor::value_into_tensor_for("sub2ind", materialised)
            .map_err(|message| sub2ind_error(message))?;
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

fn try_gpu_sub2ind(dims: &[usize], subs: &[Value]) -> crate::BuiltinResult<Option<Value>> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (dims, subs);
        return Ok(None);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
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
        return Err(sub2ind_error(
            "The number of subscripts supplied must equal the number of dimensions in the size vector.",
        ));
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
                    return Err(sub2ind_error("Subscript inputs must have the same size."));
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

    let strides = build_strides(dims, "sub2ind")?;
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
        Err(err) => Err(sub2ind_error(err.to_string())),
        }
    }
}

fn compute_indices(
    dims: &[usize],
    subscripts: &[Tensor],
) -> crate::BuiltinResult<(Vec<f64>, Option<Vec<usize>>)> {
    let mut target_shape: Option<Vec<usize>> = None;
    let mut result_len: usize = 1;
    let mut has_non_scalar = false;

    for tensor in subscripts {
        if tensor.data.len() != 1 {
            has_non_scalar = true;
            if let Some(shape) = &target_shape {
                if &tensor.shape != shape {
                    return Err(sub2ind_error("Subscript inputs must have the same size."));
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

    let strides = build_strides(dims, "sub2ind")?;
    let mut output = Vec::with_capacity(result_len);

    for idx in 0..result_len {
        let mut offset: usize = 0;
        for (dim_index, (&dim, tensor)) in dims.iter().zip(subscripts.iter()).enumerate() {
            let raw = subscript_value(tensor, idx);
            let coerced = coerce_subscript(raw, dim_index + 1, dim)?;
            let term = coerced
                .checked_sub(1)
                .and_then(|v| v.checked_mul(strides[dim_index]))
                .ok_or_else(|| sub2ind_error("Index exceeds array dimensions."))?;
            offset = offset
                .checked_add(term)
                .ok_or_else(|| sub2ind_error("Index exceeds array dimensions."))?;
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

fn coerce_subscript(value: f64, dim_number: usize, dim_size: usize) -> crate::BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(sub2ind_error(
            "Subscript indices must either be real positive integers or logicals.",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(sub2ind_error(
            "Subscript indices must either be real positive integers or logicals.",
        ));
    }
    if rounded < 1.0 {
        return Err(sub2ind_error(
            "Subscript indices must either be real positive integers or logicals.",
        ));
    }
    if rounded > dim_size as f64 {
        return Err(dimension_bounds_error(dim_number));
    }
    Ok(rounded as usize)
}

fn dimension_bounds_error(dim_number: usize) -> RuntimeError {
    let message = match dim_number {
        1 => format!("Index exceeds the number of rows in dimension {dim_number}."),
        2 => format!("Index exceeds the number of columns in dimension {dim_number}."),
        3 => format!("Index exceeds the number of pages in dimension {dim_number}."),
        _ => "Index exceeds array dimensions.".to_string(),
    };
    sub2ind_error(message)
}

fn build_host_value(data: Vec<f64>, shape: Option<Vec<usize>>) -> crate::BuiltinResult<Value> {
    let shape = shape.unwrap_or_else(|| vec![1, 1]);
    if data.len() == 1 && tensor::element_count(&shape) == 1 {
        Ok(Value::Num(data[0]))
    } else {
        let tensor = Tensor::new(data, shape)
            .map_err(|e| sub2ind_error(format!("Unable to construct sub2ind output: {e}")))?;
        Ok(Value::Tensor(tensor))
    }
}

fn sub2ind_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("sub2ind").build()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, Tensor, Value};

    fn sub2ind_builtin(dims_val: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::sub2ind_builtin(dims_val, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn converts_scalar_indices() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result =
            sub2ind_builtin(Value::Tensor(dims), vec![Value::Num(2.0), Value::Num(3.0)]).unwrap();
        assert_eq!(result, Value::Num(8.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_out_of_range_subscripts() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err = sub2ind_builtin(Value::Tensor(dims), vec![Value::Num(4.0), Value::Num(1.0)])
            .unwrap_err();
        assert!(
            err.to_string().contains("Index exceeds"),
            "expected index bounds error, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
            err.to_string().contains("same size"),
            "expected size mismatch error, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_integer_subscripts() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err = sub2ind_builtin(Value::Tensor(dims), vec![Value::Num(1.5), Value::Num(1.0)])
            .unwrap_err();
        assert!(
            err.to_string().contains("real positive integers"),
            "expected integer coercion error, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accepts_integer_value_variants() {
        let dims = Value::Tensor(Tensor::new(vec![3.0], vec![1, 1]).unwrap());
        let result = sub2ind_builtin(dims, vec![Value::Int(IntValue::I32(2))]).expect("sub2ind");
        assert_eq!(result, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
}
