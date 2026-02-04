//! MATLAB-compatible `ind2sub` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use super::common::{build_strides, dims_from_tokens, materialize_value, parse_dims, total_elements};
use crate::builtins::common::arg_tokens::tokens_from_context;
use crate::builtins::array::type_resolvers::size_vector_len;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, make_cell, RuntimeError};

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

fn ind2sub_type(args: &[Type], ctx: &ResolveContext) -> Type {
    let Some(dims) = args.first() else {
        return Type::Unknown;
    };
    let length = dims_from_tokens(&tokens_from_context(ctx))
        .map(|values| values.len())
        .or_else(|| size_vector_len(dims));
    Type::Cell {
        element_type: Some(Box::new(Type::tensor())),
        length,
    }
}

#[runtime_builtin(
    name = "ind2sub",
    category = "array/indexing",
    summary = "Convert MATLAB column-major linear indices into per-dimension subscript arrays.",
    keywords = "ind2sub,linear index,subscripts,column major,gpu indexing",
    accel = "custom",
    type_resolver(ind2sub_type),
    type_resolver_context = true,
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
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (dims, strides, total, indices);
        Ok(None)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
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
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{ResolveContext, Tensor, Type, Value};

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

    #[test]
    fn ind2sub_type_infers_cell_length() {
        let dims = Type::Tensor {
            shape: Some(vec![Some(1), Some(3)]),
        };
        assert_eq!(
            super::ind2sub_type(&[dims, Type::Num], &ResolveContext::empty()),
            Type::Cell {
                element_type: Some(Box::new(Type::tensor())),
                length: Some(3)
            }
        );
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
