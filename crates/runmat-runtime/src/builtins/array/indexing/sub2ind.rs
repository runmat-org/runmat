//! MATLAB-compatible `sub2ind` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, SUB2IND_ERROR_INDEX_BOUNDS, SUB2IND_ERROR_INTERNAL,
    SUB2IND_ERROR_INVALID_INPUT, SUB2IND_ERROR_PROVIDER,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Tensor, Value};

use super::common::{
    build_strides, fits_positive_platform_index, materialize_value, parse_dims, IndexingOperation,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeError};

#[path = "sub2ind/resident_owner.rs"]
mod resident_owner;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::indexing::sub2ind")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sub2ind",
    op_kind: GpuOpKind::Custom("indexing"),
    supported_precisions: &[ScalarType::F64],
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

const BUILTIN_NAME: &str = "sub2ind";

fn sub2ind_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sub2ind_input_error(message: impl Into<String>) -> RuntimeError {
    sub2ind_error_with_message(message, &SUB2IND_ERROR_INVALID_INPUT)
}

fn sub2ind_bounds_error(message: impl Into<String>) -> RuntimeError {
    sub2ind_error_with_message(message, &SUB2IND_ERROR_INDEX_BOUNDS)
}

fn sub2ind_provider_error(message: impl Into<String>) -> RuntimeError {
    sub2ind_error_with_message(message, &SUB2IND_ERROR_PROVIDER)
}

fn sub2ind_internal_error(message: impl Into<String>) -> RuntimeError {
    sub2ind_error_with_message(message, &SUB2IND_ERROR_INTERNAL)
}

#[runtime_builtin(
    name = "sub2ind",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::indexing::sub2ind"
)]
async fn sub2ind_builtin(
    dims_val: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let resident_source = resident_owner::source(&dims_val, rest.view());
    let resident_inputs: Vec<GpuTensorHandle> = std::iter::once(&dims_val)
        .chain(rest.iter())
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect();
    let (dims_value, _) = materialize_value(dims_val, IndexingOperation::Sub2Ind).await?;
    let dims = parse_dims(&dims_value, IndexingOperation::Sub2Ind).await?;
    if dims.is_empty() {
        return Err(sub2ind_error("Size vector must have at least one element."));
    }

    if rest.len() != dims.len() {
        return Err(sub2ind_error(
            "The number of subscripts supplied must equal the number of dimensions in the size vector.",
        ));
    }

    if let Some(value) = try_gpu_sub2ind(&dims, rest.view(), resident_source.as_ref()).await? {
        return Ok(value);
    }

    let mut subscripts: Vec<Tensor> = Vec::with_capacity(rest.len());
    for value in rest {
        let (materialised, _) = materialize_value(value, IndexingOperation::Sub2Ind).await?;
        let tensor = tensor::value_into_tensor_for("sub2ind", materialised)
            .map_err(|message| sub2ind_error(message))?;
        subscripts.push(tensor);
    }

    let (result_data, result_shape) = compute_indices(&dims, &subscripts)?;
    let host = build_host_value(result_data, result_shape)?;
    let protected: Vec<_> = resident_inputs.iter().collect();
    resident_owner::restore_host(resident_source.as_ref(), host, &protected)
}

async fn try_gpu_sub2ind(
    dims: &[usize],
    subs: crate::call::FunctionArgumentView<'_>,
    source: Option<&GpuTensorHandle>,
) -> crate::BuiltinResult<Option<Value>> {
    let Some(source) = source else {
        return Ok(None);
    };
    if !subs
        .iter()
        .all(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Ok(None);
    }
    if subs.iter().any(
            |value| matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some()),
        ) {
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

    gpu_helpers::require_resident_compute_owners(&handles, false, || {
        sub2ind_provider_error("sub2ind: no exact provider owns the resident subscripts")
    })?;
    let Some(provider) = resident_owner::direct_owner(source, &handles) else {
        return Ok(None);
    };

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
        let len = tensor::element_count(handle.shape());
        let is_scalar = len == 1;
        scalar_mask.push(is_scalar);
        if !is_scalar {
            saw_non_scalar = true;
            if let Some(existing) = &target_shape {
                if existing != handle.shape() {
                    return Err(sub2ind_error("Subscript inputs must have the same size."));
                }
            } else {
                target_shape = Some(handle.shape().to_vec());
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

    let strides = build_strides(dims, IndexingOperation::Sub2Ind)?;
    if dims.iter().any(|&d| d > u32::MAX as usize)
        || strides.iter().any(|&s| s > u32::MAX as usize)
        || result_len > u32::MAX as usize
    {
        return Ok(None);
    }

    let output_shape = target_shape.clone().unwrap_or_else(|| vec![1, 1]);
    match provider
        .sub2ind(
            dims,
            &strides,
            &handles,
            &scalar_mask,
            result_len,
            &output_shape,
        )
        .await
    {
        Ok(handle) => {
            resident_owner::publish_direct(handle, &output_shape, source, &handles, provider)
                .map(Some)
        }
        Err(err) if err.is_unsupported() => Ok(None),
        Err(err) => Err(crate::runtime_error::with_provider_execution_source(
            sub2ind_provider_error(err.to_string()),
            err,
        )),
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
        let len = tensor_element_len(tensor);
        if len != 1 {
            has_non_scalar = true;
            if let Some(shape) = &target_shape {
                if tensor.shape() != shape {
                    return Err(sub2ind_error("Subscript inputs must have the same size."));
                }
            } else {
                target_shape = Some(tensor.shape().to_vec());
                result_len = len;
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

    let strides = build_strides(dims, IndexingOperation::Sub2Ind)?;
    let mut output = Vec::with_capacity(result_len);

    for idx in 0..result_len {
        let mut offset: usize = 0;
        for (dim_index, (&dim, tensor)) in dims.iter().zip(subscripts.iter()).enumerate() {
            let raw = subscript_value(tensor, idx);
            let coerced = coerce_subscript_value(raw, dim_index + 1, dim)?;
            let term = coerced
                .checked_sub(1)
                .and_then(|v| v.checked_mul(strides[dim_index]))
                .ok_or_else(|| sub2ind_bounds_error("Index exceeds array dimensions."))?;
            offset = offset
                .checked_add(term)
                .ok_or_else(|| sub2ind_bounds_error("Index exceeds array dimensions."))?;
        }
        output.push((offset + 1) as f64);
    }

    Ok((output, target_shape))
}

fn tensor_element_len(tensor: &Tensor) -> usize {
    tensor.len()
}

enum SubscriptValue {
    Float(f64),
    Integer(IntValue),
}

fn subscript_value(tensor: &Tensor, idx: usize) -> SubscriptValue {
    if let Some(storage) = tensor.integer_storage() {
        let index = if storage.len() == 1 { 0 } else { idx };
        return SubscriptValue::Integer(
            storage
                .value_at(index)
                .expect("subscript index is within integer storage bounds"),
        );
    }
    if tensor::is_scalar_tensor(tensor) {
        SubscriptValue::Float(tensor::tensor_value_f64(tensor, 0))
    } else {
        SubscriptValue::Float(tensor::tensor_value_f64(tensor, idx))
    }
}

fn coerce_subscript_value(
    value: SubscriptValue,
    dim_number: usize,
    dim_size: usize,
) -> crate::BuiltinResult<usize> {
    match value {
        SubscriptValue::Float(value) => coerce_subscript(value, dim_number, dim_size),
        SubscriptValue::Integer(value) => coerce_integer_subscript(&value, dim_number, dim_size),
    }
}

fn coerce_integer_subscript(
    value: &IntValue,
    dim_number: usize,
    dim_size: usize,
) -> crate::BuiltinResult<usize> {
    let Some(index) = value.try_to_usize() else {
        return Err(sub2ind_error(
            "Subscript indices must either be real positive integers or logicals.",
        ));
    };
    if index < 1 {
        return Err(sub2ind_error(
            "Subscript indices must either be real positive integers or logicals.",
        ));
    }
    if index > dim_size {
        return Err(dimension_bounds_error(dim_number));
    }
    Ok(index)
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
    if !fits_positive_platform_index(rounded) {
        return Err(sub2ind_error(
            "Subscript indices exceed the maximum supported index range.",
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
    sub2ind_bounds_error(message)
}

fn build_host_value(data: Vec<f64>, shape: Option<Vec<usize>>) -> crate::BuiltinResult<Value> {
    let shape = shape.unwrap_or_else(|| vec![1, 1]);
    if data.len() == 1 && tensor::element_count(&shape) == 1 {
        Ok(Value::Num(data[0]))
    } else {
        let tensor = Tensor::new(data, shape).map_err(|e| {
            sub2ind_internal_error(format!("Unable to construct sub2ind output: {e}"))
        })?;
        Ok(Value::Tensor(tensor))
    }
}

fn sub2ind_error(message: impl Into<String>) -> RuntimeError {
    sub2ind_input_error(message)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{IntValue, IntegerStorage, Tensor, Value};

    fn sub2ind_builtin(dims_val: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
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
                assert_eq!(t.shape(), [3, 1]);
                assert_eq!(t.materialize_f64(), vec![10.0, 11.0, 12.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sub2ind_typed_integer_subscripts_read_exact_storage() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let rows = Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1]).unwrap();
        let cols = Tensor::new_integer(IntegerStorage::I16(vec![4]), vec![1, 1]).unwrap();

        let result = sub2ind_builtin(
            Value::Tensor(dims),
            vec![Value::Tensor(rows), Value::Tensor(cols)],
        )
        .unwrap();

        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), [3, 1]);
                assert_eq!(tensor.materialize_f64(), vec![10.0, 11.0, 12.0]);
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
                assert_eq!(t.shape(), [1, 2]);
                assert_eq!(t.materialize_f64(), vec![3.0, 11.0]);
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
        assert_eq!(
            err.identifier(),
            super::SUB2IND_ERROR_INDEX_BOUNDS.identifier
        );
    }

    #[test]
    fn sub2ind_accepts_every_integer_class_and_returns_double() {
        for prototype in [
            IntegerStorage::I8(Vec::new()),
            IntegerStorage::I16(Vec::new()),
            IntegerStorage::I32(Vec::new()),
            IntegerStorage::I64(Vec::new()),
            IntegerStorage::U8(Vec::new()),
            IntegerStorage::U16(Vec::new()),
            IntegerStorage::U32(Vec::new()),
            IntegerStorage::U64(Vec::new()),
        ] {
            let typed = |values: &[i8], shape| {
                let values = values
                    .iter()
                    .map(|value| prototype.cast_exact_assignment(&IntValue::I8(*value)))
                    .collect();
                Tensor::new_integer(
                    prototype
                        .from_same_class_values(values)
                        .expect("same-class values"),
                    shape,
                )
                .expect("typed tensor")
            };
            let result = sub2ind_builtin(
                Value::Tensor(typed(&[2, 3], vec![1, 2])),
                vec![
                    Value::Tensor(typed(&[1, 2], vec![1, 2])),
                    Value::Tensor(typed(&[3], vec![1, 1])),
                ],
            )
            .expect("sub2ind");
            let Value::Tensor(result) = result else {
                panic!("expected double tensor output");
            };
            assert_eq!(result.shape(), [1, 2]);
            assert_eq!(result.materialize_f64(), vec![5.0, 6.0]);
            assert!(result.integer_storage().is_none());
        }
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
        assert_eq!(
            err.identifier(),
            super::SUB2IND_ERROR_INVALID_INPUT.identifier
        );
    }

    #[test]
    fn rejects_trailing_singleton_rank_mismatch_for_non_scalar_subscripts() {
        let dims = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let rows = Tensor::new(vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0], vec![2, 3]).unwrap();
        let cols = Tensor::new(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], vec![2, 3, 1]).unwrap();
        let error = sub2ind_builtin(
            Value::Tensor(dims),
            vec![Value::Tensor(rows), Value::Tensor(cols)],
        )
        .expect_err("runtime requires exact non-scalar shape and rank");
        assert_eq!(error.identifier(), SUB2IND_ERROR_INVALID_INPUT.identifier);
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
        assert_eq!(
            err.identifier(),
            super::SUB2IND_ERROR_INVALID_INPUT.identifier
        );
    }

    #[test]
    fn rejects_oversized_float_subscripts_before_casting() {
        let dims = Value::Int(IntValue::U64(usize::MAX.saturating_sub(1) as u64));

        let err = sub2ind_builtin(dims.clone(), vec![Value::Num(1.0e300)])
            .expect_err("huge float subscript must reject");
        assert_eq!(
            err.identifier(),
            super::SUB2IND_ERROR_INVALID_INPUT.identifier
        );

        let err = sub2ind_builtin(dims, vec![Value::Num(usize::MAX as f64)])
            .expect_err("platform boundary float subscript must reject");
        assert_eq!(
            err.identifier(),
            super::SUB2IND_ERROR_INVALID_INPUT.identifier
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
                    data: &dims.materialize_f64(),
                    shape: dims.shape(),
                })
                .expect("upload dims");
            let rows_handle = provider
                .upload(&HostTensorView {
                    data: &rows.materialize_f64(),
                    shape: rows.shape(),
                })
                .expect("upload rows");
            let cols_handle = provider
                .upload(&HostTensorView {
                    data: &cols.materialize_f64(),
                    shape: cols.shape(),
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
                    assert_eq!(gathered.shape(), [3, 1]);
                    assert_eq!(gathered.materialize_f64(), vec![10.0, 11.0, 12.0]);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn sub2ind_integer_gpu_inputs_fall_back_exactly_to_resident_double() {
        test_support::with_test_provider(|provider| {
            let dims =
                Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![1, 2]).expect("dims");
            let rows = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[1, 2]),
                    shape: &[1, 2],
                })
                .expect("rows");
            let cols = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[3]),
                    shape: &[1, 1],
                })
                .expect("cols");
            let result = sub2ind_builtin(
                Value::Tensor(dims),
                vec![Value::GpuTensor(rows), Value::GpuTensor(cols)],
            )
            .expect("sub2ind");
            let Value::GpuTensor(handle) = &result else {
                panic!("expected resident double output");
            };
            assert_eq!(runmat_accelerate_api::handle_integer_type(handle), None);
            let result = test_support::gather(result).expect("gather result");
            assert_eq!(result.shape(), [1, 2]);
            assert_eq!(result.materialize_f64(), vec![5.0, 6.0]);
            assert!(result.integer_storage().is_none());
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
                data: &rows.materialize_f64(),
                shape: rows.shape(),
            })
            .expect("upload rows");
        let cols_handle = provider
            .upload(&HostTensorView {
                data: &cols.materialize_f64(),
                shape: cols.shape(),
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
        assert_eq!(gathered.shape(), expected.shape());
        assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    }
}
