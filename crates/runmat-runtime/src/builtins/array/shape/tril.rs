//! MATLAB-compatible `tril` builtin with GPU-aware semantics for RunMat.
//!
//! This module implements the `tril` function, mirroring MathWorks MATLAB
//! behaviour across real, logical, and complex tensors, including paged
//! matrices. It honours diagonal offsets, keeps higher-dimensional slices
//! independent, and preserves gpuArray residency whenever an acceleration
//! provider is registered.

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, provider_restore, tensor};
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, TRIL_DESCRIPTOR, TRIL_ERROR_INTERNAL, TRIL_ERROR_INVALID_OFFSET,
    TRIL_ERROR_TOO_MANY_INPUTS, TRIL_ERROR_TOO_MANY_OUTPUTS, TRIL_ERROR_UNSUPPORTED_INPUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::CharArray;
use runmat_value::{ComplexStorage, ComplexTensor, LogicalArray, NumericStorage, Tensor, Value};

const BUILTIN_NAME: &str = "tril";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::tril")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tril",
    op_kind: GpuOpKind::Custom("tril"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("tril")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement tril directly; the runtime falls back to gather→compute→upload when unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::tril")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tril",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Triangular masking is not currently fused; fusion planner treats tril nodes as boundaries.",
};

fn tril_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    tril_error_with_message(error.message, error)
}

fn tril_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "tril",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::shape::tril"
)]
async fn tril_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &TRIL_DESCRIPTOR,
        &TRIL_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )?;
    if rest.len() > 1 {
        return Err(tril_error(&TRIL_ERROR_TOO_MANY_INPUTS));
    }
    let offset = parse_diagonal_offset(rest.view()).await?;
    match value {
        Value::GpuTensor(handle) => tril_gpu(handle, offset).await,
        value => tril_host_value(value, offset),
    }
}

fn tril_host_value(value: Value, offset: isize) -> crate::BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => Ok(tril_tensor(tensor, offset).map(tensor::tensor_into_value)?),
        Value::LogicalArray(array) => {
            Ok(tril_logical_array(array, offset).map(Value::LogicalArray)?)
        }
        Value::ComplexTensor(tensor) => {
            Ok(tril_complex_tensor(tensor, offset).map(Value::ComplexTensor)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| tril_error_with_message(format!("tril: {e}"), &TRIL_ERROR_INTERNAL))?;
            Ok(tril_complex_tensor(tensor, offset).map(complex_tensor_into_value)?)
        }
        Value::Num(n) => {
            let tensor = tensor::value_into_tensor_for("tril", Value::Num(n))
                .map_err(|e| tril_error_with_message(e, &TRIL_ERROR_INTERNAL))?;
            Ok(tril_tensor(tensor, offset).map(tensor::tensor_into_value)?)
        }
        Value::Int(i) => {
            let tensor = tensor::value_into_tensor_for("tril", Value::Int(i.clone()))
                .map_err(|e| tril_error_with_message(e, &TRIL_ERROR_INTERNAL))?;
            Ok(tril_tensor(tensor, offset).map(tensor::tensor_into_value)?)
        }
        Value::Bool(flag) => {
            let tensor = tensor::value_into_tensor_for("tril", Value::Bool(flag))
                .map_err(|e| tril_error_with_message(e, &TRIL_ERROR_INTERNAL))?;
            Ok(tril_tensor(tensor, offset).map(tensor::tensor_into_value)?)
        }
        Value::CharArray(chars) => Ok(Value::CharArray(tril_char_array(chars, offset)?)),
        Value::GpuTensor(_) => unreachable!("resident values are handled before host masking"),
        Value::String(_) | Value::StringArray(_) => Err(tril_error_with_message(
            "tril: string arrays are not supported",
            &TRIL_ERROR_UNSUPPORTED_INPUT,
        )),
        Value::Cell(_) => Err(tril_error_with_message(
            "tril: cell arrays are not supported",
            &TRIL_ERROR_UNSUPPORTED_INPUT,
        )),
        Value::ObjectArray(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::SparseTensor(_)
        | Value::Struct(_)
        | Value::StructArray(_)
        | Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::Future(_)
        | Value::Task(_)
        | Value::Pool(_)
        | Value::Job(_)
        | Value::Distributed(_)
        | Value::Composite(_)
        | Value::Foreign(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::OutputList(_) => Err(tril_error(&TRIL_ERROR_UNSUPPORTED_INPUT)),
    }
}

async fn parse_diagonal_offset(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<isize> {
    if args.is_empty() {
        return Ok(0);
    }
    let gathered = crate::dispatcher::gather_if_needed_async(&args[0])
        .await
        .map_err(|e| tril_error_with_message(format!("tril: {e}"), &TRIL_ERROR_INTERNAL))?;
    scalar_to_isize(&gathered, "tril")
}

fn scalar_to_isize(value: &Value, name: &str) -> crate::BuiltinResult<isize> {
    match value {
        Value::Int(i) => i.try_to_isize().ok_or_else(|| {
            tril_error_with_message(
                format!("{name}: diagonal offset is outside the supported range"),
                &TRIL_ERROR_INVALID_OFFSET,
            )
        }),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(tril_error_with_message(
                    format!("{name}: diagonal offset must be finite"),
                    &TRIL_ERROR_INVALID_OFFSET,
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(tril_error_with_message(
                    format!("{name}: diagonal offset must be an integer"),
                    &TRIL_ERROR_INVALID_OFFSET,
                ));
            }
            if rounded < isize::MIN as f64
                || rounded > isize::MAX as f64
                || (isize::BITS == 64 && rounded == isize::MAX as f64)
            {
                return Err(tril_error_with_message(
                    format!("{name}: diagonal offset is outside the supported range"),
                    &TRIL_ERROR_INVALID_OFFSET,
                ));
            }
            Ok(rounded as isize)
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(storage) = t.integer_storage() {
                return scalar_to_isize(
                    &Value::Int(
                        storage
                            .value_at(0)
                            .expect("scalar integer storage has one element"),
                    ),
                    name,
                );
            }
            scalar_to_isize(&Value::Num(tensor::tensor_value_f64(t, 0)), name)
        }
        Value::Bool(flag) => Ok(if *flag { 1 } else { 0 }),
        other => Err(tril_error_with_message(
            format!("{name}: diagonal offset must be a scalar numeric value, got {other:?}"),
            &TRIL_ERROR_INVALID_OFFSET,
        )),
    }
}

fn tril_tensor(tensor: Tensor, offset: isize) -> crate::BuiltinResult<Tensor> {
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| tril_error_with_message(format!("tril: {e}"), &TRIL_ERROR_INTERNAL))?;
    let storage = map_tril_storage(storage, &shape, offset)?;
    Tensor::from_numeric_storage(storage, shape)
        .map_err(|e| tril_error_with_message(format!("tril: {e}"), &TRIL_ERROR_INTERNAL))
}

fn map_tril_storage(
    storage: NumericStorage,
    shape: &[usize],
    offset: isize,
) -> crate::BuiltinResult<NumericStorage> {
    macro_rules! apply {
        ($values:expr, $variant:ident) => {
            NumericStorage::$variant(apply_tril_values($values, shape, offset)?)
        };
    }
    Ok(match storage {
        NumericStorage::F64(values) => apply!(values, F64),
        NumericStorage::F32(values) => apply!(values, F32),
        NumericStorage::I8(values) => apply!(values, I8),
        NumericStorage::I16(values) => apply!(values, I16),
        NumericStorage::I32(values) => apply!(values, I32),
        NumericStorage::I64(values) => apply!(values, I64),
        NumericStorage::U8(values) => apply!(values, U8),
        NumericStorage::U16(values) => apply!(values, U16),
        NumericStorage::U32(values) => apply!(values, U32),
        NumericStorage::U64(values) => apply!(values, U64),
    })
}

fn apply_tril_values<T: Clone + Default>(
    mut values: Vec<T>,
    shape: &[usize],
    offset: isize,
) -> crate::BuiltinResult<Vec<T>> {
    apply_tril_inplace(&mut values, shape, offset, T::default())?;
    Ok(values)
}

fn tril_logical_array(
    mut array: LogicalArray,
    offset: isize,
) -> crate::BuiltinResult<LogicalArray> {
    apply_tril_inplace(&mut array.data, &array.shape, offset, 0u8)?;
    Ok(array)
}

fn tril_char_array(array: CharArray, offset: isize) -> crate::BuiltinResult<CharArray> {
    super::triangular_character::mask(
        crate::context::legacy::active().as_ref(),
        array,
        offset,
        super::triangular_character::Triangle::Lower,
    )
}

fn tril_complex_tensor(
    tensor: ComplexTensor,
    offset: isize,
) -> crate::BuiltinResult<ComplexTensor> {
    let (storage, shape) = tensor.into_complex_storage_and_shape();
    let storage = match storage {
        ComplexStorage::F64(mut values) => {
            apply_tril_inplace(&mut values, &shape, offset, (0.0, 0.0).into())?;
            ComplexStorage::F64(values)
        }
        ComplexStorage::F32(mut values) => {
            apply_tril_inplace(&mut values, &shape, offset, (0.0, 0.0).into())?;
            ComplexStorage::F32(values)
        }
        ComplexStorage::Integer(storage) => {
            let zero = storage
                .real()
                .zeros_like(1)
                .value_at(0)
                .expect("one typed integer zero");
            ComplexStorage::Integer(
                storage
                    .reorder(|values| {
                        let mut values = values.to_vec();
                        apply_tril_inplace(&mut values, &shape, offset, zero.clone())
                            .map_err(|e| e.to_string())?;
                        Ok(values)
                    })
                    .map_err(|e| {
                        tril_error_with_message(format!("tril: {e}"), &TRIL_ERROR_INTERNAL)
                    })?,
            )
        }
    };
    ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| tril_error_with_message(format!("tril: {e}"), &TRIL_ERROR_INTERNAL))
}

async fn tril_gpu(handle: GpuTensorHandle, offset: isize) -> crate::BuiltinResult<Value> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id() != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if runmat_accelerate_api::handle_integer_type(&handle).is_none()
            && runmat_accelerate_api::handle_storage(&handle)
                == runmat_accelerate_api::GpuTensorStorage::Real
            && !runmat_accelerate_api::handle_is_logical(&handle)
        {
            match provider.tril(&handle, offset).await {
                Ok(mut out) => {
                    if !triangular_provider_output_valid(provider, &out, &handle) {
                        return Err(gpu_helpers::reject_gpu_output(
                            out,
                            tril_error_with_message(
                                "tril: provider returned malformed or aliased output",
                                &TRIL_ERROR_INTERNAL,
                            ),
                        ));
                    }
                    if let Err(error) =
                        gpu_helpers::propagate_output_provenance(&mut out, [&handle])
                    {
                        return Err(gpu_helpers::reject_gpu_output(out, error));
                    }
                    return Ok(Value::GpuTensor(out));
                }
                Err(error) if error.is_unsupported() => {}
                Err(error) => {
                    let fallback = tril_error_with_message(
                        format!("tril: admitted provider operation failed: {error}"),
                        &TRIL_ERROR_INTERNAL,
                    );
                    return Err(crate::runtime_error::with_provider_execution_source(
                        fallback, error,
                    ));
                }
            }
        }
        provider_restore::gather_value_compute_restore(handle, BUILTIN_NAME, move |value| {
            tril_host_value(value, offset)
        })
        .await
    } else {
        let value = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
        tril_host_value(value, offset)
    }
}

fn triangular_provider_output_valid(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
) -> bool {
    output.allocation_lease().validate_live().is_ok()
        && output.shape() == input.shape()
        && output.device_id() == input.device_id()
        && (input.shape().contains(&0) || !output.shares_allocation_with(input))
        && runmat_accelerate_api::handle_storage(output)
            == runmat_accelerate_api::handle_storage(input)
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(input)
        && runmat_accelerate_api::handle_integer_type(output)
            == runmat_accelerate_api::handle_integer_type(input)
        && runmat_accelerate_api::handle_is_logical(output)
            == runmat_accelerate_api::handle_is_logical(input)
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
}

fn apply_tril_inplace<T>(
    data: &mut [T],
    shape: &[usize],
    offset: isize,
    zero: T,
) -> crate::BuiltinResult<()>
where
    T: Clone,
{
    if data.is_empty() {
        return Ok(());
    }
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    let plane = rows.saturating_mul(cols);
    let pages = if shape.len() <= 2 {
        1
    } else {
        shape[2..].iter().product::<usize>()
    };
    if plane == 0 || pages == 0 {
        return Ok(());
    }
    let expected = plane.checked_mul(pages).ok_or_else(|| {
        tril_error_with_message("tril: dimension product overflow", &TRIL_ERROR_INTERNAL)
    })?;
    if expected != data.len() {
        return Err(tril_error_with_message(
            "tril: tensor data length mismatch",
            &TRIL_ERROR_INTERNAL,
        ));
    }
    for page in 0..pages {
        let base = page * plane;
        for col in 0..cols {
            let col_base = base + col * rows;
            for row in 0..rows {
                triangular_checkpoint(col_base.saturating_add(row))?;
                let row_idx = row as i128;
                let col_idx = col as i128;
                let offset_idx = offset as i128;
                if row_idx < col_idx - offset_idx {
                    data[col_base + row] = zero.clone();
                }
            }
        }
    }
    Ok(())
}

fn triangular_checkpoint(index: usize) -> crate::BuiltinResult<()> {
    if !index.is_multiple_of(1_024) {
        return Ok(());
    }
    let Some(runtime) = crate::context::legacy::active() else {
        return Ok(());
    };
    let cancellation = runtime.cancellation_context();
    if cancellation.is_cancelled() {
        return Err(crate::builtin::builtin_control_error(
            Some(&cancellation),
            BUILTIN_NAME,
            crate::context::host_allocation::HostAllocationError::Cancelled,
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn tril_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::tril_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::{
        HostIntegerDataView, HostIntegerTensorView, HostTensorView, IntegerElementType,
    };
    use runmat_value::{IntValue, IntegerStorage, LogicalArray};

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        Value::Tensor(tensor)
    }

    #[test]
    fn tril_offset_parser_preserves_signed_values_and_rejects_unrepresentable_uint64() {
        assert_eq!(
            scalar_to_isize(&Value::Int(IntValue::I64(-1)), "tril").expect("signed offset"),
            -1
        );
        let err = scalar_to_isize(&Value::Int(IntValue::U64(u64::MAX)), "tril")
            .expect_err("unrepresentable typed offset must not saturate");
        assert_eq!(err.identifier(), TRIL_ERROR_INVALID_OFFSET.identifier);
    }

    #[test]
    fn tril_gpu_integer_fallback_preserves_exact_storage_resident() {
        test_support::with_test_provider(|provider| {
            let values = [u64::MAX, 9_007_199_254_740_993, 7_u64, 0_u64];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&values),
                    shape: &[2, 2],
                })
                .expect("upload integer");
            let Value::GpuTensor(result) =
                tril_builtin(Value::GpuTensor(handle), Vec::new()).expect("tril integer gpu")
            else {
                panic!("expected resident gpuArray");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&result),
                Some(IntegerElementType::U64)
            );
            let gathered = block_on(gpu_helpers::gather_tensor_async(&result)).expect("gather");
            assert_eq!(gathered.shape(), [2, 2]);
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![
                    u64::MAX,
                    9_007_199_254_740_993,
                    0,
                    0,
                ]))
            );
        });
    }

    #[test]
    fn tril_offset_parser_reads_typed_integer_tensor_storage_exactly() {
        assert_eq!(
            scalar_to_isize(
                &poisoned_int_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1]),
                "tril",
            )
            .expect("typed integer tensor offset"),
            -1
        );
        let err = scalar_to_isize(
            &poisoned_int_tensor(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]),
            "tril",
        )
        .expect_err("unrepresentable typed integer tensor offset");
        assert_eq!(err.identifier(), TRIL_ERROR_INVALID_OFFSET.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_main_diagonal() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let value = tril_builtin(Value::Tensor(tensor), Vec::new()).expect("tril");
        match value {
            Value::Tensor(result) => {
                assert_eq!(result.shape(), [2, 3]);
                assert_eq!(result.materialize_f64(), vec![1.0, 4.0, 0.0, 5.0, 0.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn tril_preserves_native_single_storage() {
        let tensor = Tensor::from_f32(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let Value::Tensor(result) = tril_builtin(Value::Tensor(tensor), Vec::new()).expect("tril")
        else {
            panic!("expected tensor result");
        };
        assert_eq!(
            result.into_numeric_storage().expect("single storage"),
            NumericStorage::F32(vec![1.0, 4.0, 0.0, 5.0])
        );
    }

    #[test]
    fn tril_masks_exact_integer_storage() {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993, 7, 0]),
            vec![2, 2],
        )
        .expect("input");
        let Value::Tensor(result) = tril_builtin(Value::Tensor(tensor), Vec::new()).expect("tril")
        else {
            panic!("expected tensor result");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                u64::MAX,
                9_007_199_254_740_993,
                0,
                0
            ]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_with_positive_offset_keeps_super_diagonal() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let offset = Value::Int(IntValue::I32(1));
        let value =
            tril_builtin(Value::Tensor(tensor), vec![offset]).expect("tril with positive offset");
        match value {
            Value::Tensor(result) => {
                assert_eq!(result.materialize_f64(), vec![1.0, 4.0, 2.0, 5.0, 0.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_negative_offset_drops_main_diagonal() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let offset = Value::Int(IntValue::I32(-1));
        let value =
            tril_builtin(Value::Tensor(tensor), vec![offset]).expect("tril with negative offset");
        match value {
            Value::Tensor(result) => {
                assert_eq!(result.materialize_f64(), vec![0.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_logical_array_preserves_type() {
        let logical =
            LogicalArray::new(vec![1, 0, 1, 1, 1, 1], vec![2, 3]).expect("logical creation");
        let value =
            tril_builtin(Value::LogicalArray(logical), Vec::new()).expect("tril logical array");
        match value {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 3]);
                assert_eq!(array.data, vec![1, 0, 0, 1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn tril_character_matrix_preserves_class_and_row_major_layout() {
        let chars = CharArray::from_code_units(
            [0xD800, 65, 0xDC00, 0xFFFF]
                .map(runmat_value::Utf16CodeUnit::from_raw)
                .to_vec(),
            2,
            2,
        )
        .expect("character matrix");
        let value = tril_builtin(Value::CharArray(chars), Vec::new()).expect("tril char");
        let Value::CharArray(result) = value else {
            panic!("expected character result");
        };
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);
        assert_eq!(
            result
                .data()
                .iter()
                .map(|unit| unit.raw())
                .collect::<Vec<_>>(),
            vec![0xD800, 0, 0xDC00, 0xFFFF]
        );

        let pages =
            CharArray::from_unicode_scalars_with_shape("abcdefgh".chars().collect(), vec![2, 2, 2])
                .unwrap();
        let original = pages.clone();
        let masked = tril_char_array(pages, 0).unwrap();
        assert_eq!(masked.shape.as_slice(), &[2, 2, 2]);
        assert_eq!(
            masked.data(),
            runmat_value::encode_utf16_code_units("a\0cde\0gh")
        );
        assert_eq!(
            original.data(),
            runmat_value::encode_utf16_code_units("abcdefgh")
        );
        let pointer = masked.data().as_ptr();
        let reused = tril_char_array(masked, isize::MAX).unwrap();
        assert_eq!(reused.data().as_ptr(), pointer);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_complex_tensor_masks_values() {
        let data = vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let value = tril_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("tril complex");
        match value {
            Value::ComplexTensor(result) => {
                assert_eq!(result.shape(), [2, 2]);
                assert_eq!(result.materialize_f64()[0], (1.0, 2.0));
                assert_eq!(result.materialize_f64()[1], (3.0, 4.0));
                assert_eq!(result.materialize_f64()[2], (0.0, 0.0));
                assert_eq!(result.materialize_f64()[3], (7.0, 8.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_scalar_with_negative_offset_returns_zero() {
        let value =
            tril_builtin(Value::Num(5.0), vec![Value::Int(IntValue::I32(-1))]).expect("tril");
        match value {
            Value::Num(result) => assert_eq!(result, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_rejects_string_offset_with_stable_identifier() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = tril_builtin(Value::Tensor(tensor), vec![Value::from("diagonal")]).unwrap_err();
        assert_eq!(err.identifier(), TRIL_ERROR_INVALID_OFFSET.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_rejects_extra_arguments_with_stable_identifier() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = tril_builtin(
            Value::Tensor(tensor),
            vec![Value::Num(1.0), Value::Num(2.0)],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), TRIL_ERROR_TOO_MANY_INPUTS.identifier);
    }

    #[test]
    fn tril_rejects_excess_outputs_before_masking() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output scope");
        let err = tril_builtin(Value::Num(3.0), Vec::new()).expect_err("two outputs");
        assert_eq!(err.identifier(), TRIL_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tril_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let value = tril_builtin(Value::GpuTensor(handle), Vec::new()).expect("tril gpu");
            let gathered = test_support::gather(value).expect("gather");
            assert_eq!(gathered.shape(), [2, 2]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 2.0, 0.0, 4.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn tril_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor =
            Tensor::new((1..=16).map(|v| v as f64).collect::<Vec<_>>(), vec![4, 4]).unwrap();
        let cpu = tril_tensor(tensor.clone(), -1).expect("cpu tril");
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu = block_on(super::tril_gpu(handle, -1)).expect("gpu tril");
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape(), cpu.shape());
        assert_eq!(gathered.materialize_f64(), cpu.materialize_f64());
    }
}
