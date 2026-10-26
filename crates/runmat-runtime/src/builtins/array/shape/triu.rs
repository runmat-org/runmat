//! MATLAB-compatible `triu` builtin with GPU-aware semantics for RunMat.
//!
//! Mirrors MathWorks MATLAB behaviour across numeric, logical, and complex
//! tensors while preserving gpuArray residency whenever possible. The builtin
//! honours diagonal offsets and applies the mask independently to every matrix
//! page of higher-dimensional tensors.

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, provider_restore, tensor};
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, TRIU_DESCRIPTOR, TRIU_ERROR_INTERNAL, TRIU_ERROR_INVALID_OFFSET,
    TRIU_ERROR_TOO_MANY_INPUTS, TRIU_ERROR_TOO_MANY_OUTPUTS, TRIU_ERROR_UNSUPPORTED_INPUT,
    TRIU_PAGED_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::CharArray;
use runmat_value::{ComplexStorage, ComplexTensor, LogicalArray, NumericStorage, Tensor, Value};

const BUILTIN_NAME: &str = "triu";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::triu")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "triu",
    op_kind: GpuOpKind::Custom("triu"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("triu")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement triu directly; the runtime falls back to gather→mask→upload when unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::triu")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "triu",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Triangular masking is currently treated as a fusion boundary.",
};

fn triu_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    triu_error_with_message(error.message, error)
}

fn triu_error_with_message(
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
    name = "triu",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::shape::triu"
)]
async fn triu_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &TRIU_DESCRIPTOR,
        &TRIU_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )?;
    if rest.len() > 1 {
        return Err(triu_error(&TRIU_ERROR_TOO_MANY_INPUTS));
    }
    if triu_value_is_paged(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TRIU_PAGED_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let offset = parse_diagonal_offset(rest.view()).await?;
    match value {
        Value::GpuTensor(handle) => triu_gpu(handle, offset).await,
        value => triu_host_value(value, offset),
    }
}

fn triu_host_value(value: Value, offset: isize) -> crate::BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => Ok(triu_tensor(tensor, offset).map(tensor::tensor_into_value)?),
        Value::LogicalArray(array) => {
            Ok(triu_logical_array(array, offset).map(Value::LogicalArray)?)
        }
        Value::ComplexTensor(tensor) => {
            Ok(triu_complex_tensor(tensor, offset).map(Value::ComplexTensor)?)
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| triu_error_with_message(format!("triu: {e}"), &TRIU_ERROR_INTERNAL))?;
            Ok(triu_complex_tensor(tensor, offset).map(complex_tensor_into_value)?)
        }
        Value::Num(n) => {
            let tensor = tensor::value_into_tensor_for("triu", Value::Num(n))
                .map_err(|e| triu_error_with_message(e, &TRIU_ERROR_INTERNAL))?;
            Ok(triu_tensor(tensor, offset).map(tensor::tensor_into_value)?)
        }
        Value::Int(i) => {
            let tensor = tensor::value_into_tensor_for("triu", Value::Int(i.clone()))
                .map_err(|e| triu_error_with_message(e, &TRIU_ERROR_INTERNAL))?;
            Ok(triu_tensor(tensor, offset).map(tensor::tensor_into_value)?)
        }
        Value::Bool(flag) => {
            let tensor = tensor::value_into_tensor_for("triu", Value::Bool(flag))
                .map_err(|e| triu_error_with_message(e, &TRIU_ERROR_INTERNAL))?;
            Ok(triu_tensor(tensor, offset).map(tensor::tensor_into_value)?)
        }
        Value::CharArray(chars) => Ok(Value::CharArray(triu_char_array(chars, offset)?)),
        Value::GpuTensor(_) => unreachable!("resident values are handled before host masking"),
        Value::String(_) | Value::StringArray(_) => Err(triu_error_with_message(
            "triu: string arrays are not supported",
            &TRIU_ERROR_UNSUPPORTED_INPUT,
        )),
        Value::Cell(_) => Err(triu_error_with_message(
            "triu: cell arrays are not supported",
            &TRIU_ERROR_UNSUPPORTED_INPUT,
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
        | Value::OutputList(_) => Err(triu_error(&TRIU_ERROR_UNSUPPORTED_INPUT)),
    }
}

fn triu_value_is_paged(value: &Value) -> bool {
    match value {
        Value::Tensor(value) => value.shape().len() > 2,
        Value::LogicalArray(value) => value.shape.len() > 2,
        Value::ComplexTensor(value) => value.shape().len() > 2,
        Value::CharArray(value) => value.shape().len() > 2,
        Value::GpuTensor(value) => value.shape().len() > 2,
        _ => false,
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
        .map_err(|e| triu_error_with_message(format!("triu: {e}"), &TRIU_ERROR_INTERNAL))?;
    scalar_to_isize(&gathered, "triu")
}

fn scalar_to_isize(value: &Value, name: &str) -> crate::BuiltinResult<isize> {
    match value {
        Value::Int(i) => i.try_to_isize().ok_or_else(|| {
            triu_error_with_message(
                format!("{name}: diagonal offset is outside the supported range"),
                &TRIU_ERROR_INVALID_OFFSET,
            )
        }),
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(triu_error_with_message(
                    format!("{name}: diagonal offset must be finite"),
                    &TRIU_ERROR_INVALID_OFFSET,
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(triu_error_with_message(
                    format!("{name}: diagonal offset must be an integer"),
                    &TRIU_ERROR_INVALID_OFFSET,
                ));
            }
            if rounded < isize::MIN as f64
                || rounded > isize::MAX as f64
                || (isize::BITS == 64 && rounded == isize::MAX as f64)
            {
                return Err(triu_error_with_message(
                    format!("{name}: diagonal offset is outside the supported range"),
                    &TRIU_ERROR_INVALID_OFFSET,
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
        other => Err(triu_error_with_message(
            format!("{name}: diagonal offset must be a scalar numeric value, got {other:?}"),
            &TRIU_ERROR_INVALID_OFFSET,
        )),
    }
}

fn triu_tensor(tensor: Tensor, offset: isize) -> crate::BuiltinResult<Tensor> {
    let shape = tensor.shape().to_vec();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|e| triu_error_with_message(format!("triu: {e}"), &TRIU_ERROR_INTERNAL))?;
    let storage = map_triu_storage(storage, &shape, offset)?;
    Tensor::from_numeric_storage(storage, shape)
        .map_err(|e| triu_error_with_message(format!("triu: {e}"), &TRIU_ERROR_INTERNAL))
}

fn map_triu_storage(
    storage: NumericStorage,
    shape: &[usize],
    offset: isize,
) -> crate::BuiltinResult<NumericStorage> {
    macro_rules! apply {
        ($values:expr, $variant:ident) => {
            NumericStorage::$variant(apply_triu_values($values, shape, offset)?)
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

fn apply_triu_values<T: Clone + Default>(
    mut values: Vec<T>,
    shape: &[usize],
    offset: isize,
) -> crate::BuiltinResult<Vec<T>> {
    apply_triu_inplace(&mut values, shape, offset, T::default())?;
    Ok(values)
}

fn triu_logical_array(
    mut array: LogicalArray,
    offset: isize,
) -> crate::BuiltinResult<LogicalArray> {
    apply_triu_inplace(&mut array.data, &array.shape, offset, 0u8)?;
    Ok(array)
}

fn triu_char_array(array: CharArray, offset: isize) -> crate::BuiltinResult<CharArray> {
    super::triangular_character::mask(
        crate::context::legacy::active().as_ref(),
        array,
        offset,
        super::triangular_character::Triangle::Upper,
    )
}

fn triu_complex_tensor(
    tensor: ComplexTensor,
    offset: isize,
) -> crate::BuiltinResult<ComplexTensor> {
    let (storage, shape) = tensor.into_complex_storage_and_shape();
    let storage = match storage {
        ComplexStorage::F64(mut values) => {
            apply_triu_inplace(&mut values, &shape, offset, (0.0, 0.0).into())?;
            ComplexStorage::F64(values)
        }
        ComplexStorage::F32(mut values) => {
            apply_triu_inplace(&mut values, &shape, offset, (0.0, 0.0).into())?;
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
                        apply_triu_inplace(&mut values, &shape, offset, zero.clone())
                            .map_err(|e| e.to_string())?;
                        Ok(values)
                    })
                    .map_err(|e| {
                        triu_error_with_message(format!("triu: {e}"), &TRIU_ERROR_INTERNAL)
                    })?,
            )
        }
    };
    ComplexTensor::from_complex_storage(storage, shape)
        .map_err(|e| triu_error_with_message(format!("triu: {e}"), &TRIU_ERROR_INTERNAL))
}

async fn triu_gpu(handle: GpuTensorHandle, offset: isize) -> crate::BuiltinResult<Value> {
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
            match provider.triu(&handle, offset).await {
                Ok(mut out) => {
                    if !triangular_provider_output_valid(provider, &out, &handle) {
                        return Err(gpu_helpers::reject_gpu_output(
                            out,
                            triu_error_with_message(
                                "triu: provider returned malformed or aliased output",
                                &TRIU_ERROR_INTERNAL,
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
                    let fallback = triu_error_with_message(
                        format!("triu: admitted provider operation failed: {error}"),
                        &TRIU_ERROR_INTERNAL,
                    );
                    return Err(crate::runtime_error::with_provider_execution_source(
                        fallback, error,
                    ));
                }
            }
        }
        provider_restore::gather_value_compute_restore(handle, BUILTIN_NAME, move |value| {
            triu_host_value(value, offset)
        })
        .await
    } else {
        let value = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
        triu_host_value(value, offset)
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

fn apply_triu_inplace<T>(
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
        1usize
    } else {
        shape[2..].iter().product::<usize>()
    };
    if plane == 0 || pages == 0 {
        return Ok(());
    }
    let expected = plane.checked_mul(pages).ok_or_else(|| {
        triu_error_with_message("triu: dimension product overflow", &TRIU_ERROR_INTERNAL)
    })?;
    if expected != data.len() {
        return Err(triu_error_with_message(
            "triu: tensor data length mismatch",
            &TRIU_ERROR_INTERNAL,
        ));
    }

    let offset_i128 = offset as i128;
    for page in 0..pages {
        let base = page * plane;
        for col in 0..cols {
            let col_base = base + col * rows;
            let col_i = col as i128;
            for row in 0..rows {
                triangular_checkpoint(col_base.saturating_add(row))?;
                let row_i = row as i128;
                let diff = col_i - row_i;
                if diff < offset_i128 {
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

    fn triu_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::triu_builtin(
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
    fn triu_offset_parser_preserves_signed_values_and_rejects_unrepresentable_uint64() {
        assert_eq!(
            scalar_to_isize(&Value::Int(IntValue::I64(-1)), "triu").expect("signed offset"),
            -1
        );
        let err = scalar_to_isize(&Value::Int(IntValue::U64(u64::MAX)), "triu")
            .expect_err("unrepresentable typed offset must not saturate");
        assert_eq!(err.identifier(), TRIU_ERROR_INVALID_OFFSET.identifier);
    }

    #[test]
    fn triu_gpu_integer_fallback_preserves_exact_storage_resident() {
        test_support::with_test_provider(|provider| {
            let values = [u64::MAX, 9_007_199_254_740_993, 7_u64, 0_u64];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&values),
                    shape: &[2, 2],
                })
                .expect("upload integer");
            let Value::GpuTensor(result) =
                triu_builtin(Value::GpuTensor(handle), Vec::new()).expect("triu integer gpu")
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
                Some(&IntegerStorage::U64(vec![u64::MAX, 0, 7, 0]))
            );
        });
    }

    #[test]
    fn triu_offset_parser_reads_typed_integer_tensor_storage_exactly() {
        assert_eq!(
            scalar_to_isize(
                &poisoned_int_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1]),
                "triu",
            )
            .expect("typed integer tensor offset"),
            -1
        );
        let err = scalar_to_isize(
            &poisoned_int_tensor(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]),
            "triu",
        )
        .expect_err("unrepresentable typed integer tensor offset");
        assert_eq!(err.identifier(), TRIU_ERROR_INVALID_OFFSET.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_main_diagonal() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let value = triu_builtin(Value::Tensor(tensor), Vec::new()).expect("triu");
        match value {
            Value::Tensor(result) => {
                assert_eq!(result.shape(), [2, 3]);
                assert_eq!(result.materialize_f64(), vec![1.0, 0.0, 2.0, 5.0, 3.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn triu_preserves_native_single_storage() {
        let tensor = Tensor::from_f32(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let Value::Tensor(result) = triu_builtin(Value::Tensor(tensor), Vec::new()).expect("triu")
        else {
            panic!("expected tensor result");
        };
        assert_eq!(
            result.into_numeric_storage().expect("single storage"),
            NumericStorage::F32(vec![1.0, 0.0, 2.0, 5.0])
        );
    }

    #[test]
    fn triu_masks_exact_integer_storage() {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993, 7, 0]),
            vec![2, 2],
        )
        .expect("input");
        let Value::Tensor(result) = triu_builtin(Value::Tensor(tensor), Vec::new()).expect("triu")
        else {
            panic!("expected tensor result");
        };
        assert_eq!(
            result.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 0, 7, 0]))
        );
    }

    #[test]
    fn triu_character_matrix_preserves_class_and_row_major_layout() {
        let chars = CharArray::from_code_units(
            [0xD800, 65, 0xDC00, 0xFFFF]
                .map(runmat_value::Utf16CodeUnit::from_raw)
                .to_vec(),
            2,
            2,
        )
        .expect("character matrix");
        let value = triu_builtin(Value::CharArray(chars), Vec::new()).expect("triu char");
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
            vec![0xD800, 65, 0, 0xFFFF]
        );

        let pages =
            CharArray::from_unicode_scalars_with_shape("abcdefgh".chars().collect(), vec![2, 2, 2])
                .unwrap();
        let original = pages.clone();
        let masked = triu_char_array(pages, 0).unwrap();
        assert_eq!(masked.shape.as_slice(), &[2, 2, 2]);
        assert_eq!(
            masked.data(),
            runmat_value::encode_utf16_code_units("ab\0def\0h")
        );
        assert_eq!(
            original.data(),
            runmat_value::encode_utf16_code_units("abcdefgh")
        );
        let pointer = masked.data().as_ptr();
        let reused = triu_char_array(masked, isize::MIN).unwrap();
        assert_eq!(reused.data().as_ptr(), pointer);
        let mut invalid = original;
        invalid.rows = 0;
        assert!(triu_char_array(invalid, 0).is_err());
    }

    #[test]
    fn triu_paged_arrays_are_an_explicit_language_extension() {
        let tensor =
            Tensor::new((1..=8).map(f64::from).collect(), vec![2, 2, 2]).expect("paged tensor");
        let chars =
            CharArray::from_unicode_scalars_with_shape("abcdefgh".chars().collect(), vec![2, 2, 2])
                .unwrap();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = triu_builtin(Value::Tensor(tensor.clone()), Vec::new())
            .expect_err("paged input must gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:TriuPagedInputExtension")
        );
        let error = triu_builtin(Value::CharArray(chars.clone()), Vec::new())
            .expect_err("paged character input must gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:TriuPagedInputExtension")
        );
        drop(_strict);
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let Value::Tensor(output) =
            triu_builtin(Value::Tensor(tensor), Vec::new()).expect("paged triu extension")
        else {
            panic!("expected tensor output");
        };
        assert_eq!(output.shape(), [2, 2, 2]);
        assert_eq!(
            output.materialize_f64(),
            vec![1.0, 0.0, 3.0, 4.0, 5.0, 0.0, 7.0, 8.0]
        );
        let Value::CharArray(output) =
            triu_builtin(Value::CharArray(chars), Vec::new()).expect("paged character extension")
        else {
            panic!("expected character output")
        };
        assert_eq!(output.shape.as_slice(), &[2, 2, 2]);
        assert_eq!(
            output.data(),
            runmat_value::encode_utf16_code_units("ab\0def\0h")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_positive_offset_drops_main_diagonal() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let offset = Value::Int(IntValue::I32(1));
        let value =
            triu_builtin(Value::Tensor(tensor), vec![offset]).expect("triu with positive offset");
        match value {
            Value::Tensor(result) => {
                assert_eq!(result.materialize_f64(), vec![0.0, 0.0, 2.0, 0.0, 3.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_negative_offset_includes_sub_diagonal() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let offset = Value::Int(IntValue::I32(-1));
        let value =
            triu_builtin(Value::Tensor(tensor), vec![offset]).expect("triu with negative offset");
        match value {
            Value::Tensor(result) => {
                assert_eq!(result.materialize_f64(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_logical_array_preserves_type() {
        let logical =
            LogicalArray::new(vec![1, 0, 1, 1, 1, 1], vec![2, 3]).expect("logical creation");
        let value =
            triu_builtin(Value::LogicalArray(logical), Vec::new()).expect("triu logical array");
        match value {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 3]);
                assert_eq!(array.data, vec![1, 0, 1, 1, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_complex_tensor_masks_values() {
        let data = vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let value = triu_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("triu complex");
        match value {
            Value::ComplexTensor(result) => {
                assert_eq!(result.shape(), [2, 2]);
                assert_eq!(result.materialize_f64()[0], (1.0, 2.0));
                assert_eq!(result.materialize_f64()[1], (0.0, 0.0));
                assert_eq!(result.materialize_f64()[2], (5.0, 6.0));
                assert_eq!(result.materialize_f64()[3], (7.0, 8.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_scalar_with_positive_offset_returns_zero() {
        let value =
            triu_builtin(Value::Num(5.0), vec![Value::Int(IntValue::I32(1))]).expect("triu");
        match value {
            Value::Num(result) => assert_eq!(result, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let value = triu_builtin(Value::GpuTensor(handle), Vec::new()).expect("triu gpu");
            let gathered = test_support::gather(value).expect("gather");
            assert_eq!(gathered.shape(), [2, 2]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 0.0, 3.0, 4.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_rejects_string_offset() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = triu_builtin(Value::Tensor(tensor), vec![Value::from("diagonal")]).unwrap_err();
        assert!(
            err.to_string()
                .contains("diagonal offset must be a scalar numeric value"),
            "unexpected error: {err}"
        );
        assert_eq!(err.identifier(), TRIU_ERROR_INVALID_OFFSET.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn triu_rejects_extra_arguments() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = triu_builtin(
            Value::Tensor(tensor),
            vec![Value::Num(1.0), Value::Num(2.0)],
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("too many input arguments"),
            "unexpected error: {err}"
        );
        assert_eq!(err.identifier(), TRIU_ERROR_TOO_MANY_INPUTS.identifier);
    }

    #[test]
    fn triu_rejects_excess_outputs_before_compatibility_or_masking() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output scope");
        let err = triu_builtin(Value::Num(3.0), Vec::new()).expect_err("two outputs");
        assert_eq!(err.identifier(), TRIU_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn triu_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor =
            Tensor::new((1..=16).map(|v| v as f64).collect::<Vec<_>>(), vec![4, 4]).unwrap();
        let cpu = triu_tensor(tensor.clone(), 1).expect("cpu triu");
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu = block_on(super::triu_gpu(handle, 1)).expect("gpu triu");
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape(), cpu.shape());
        assert_eq!(gathered.materialize_f64(), cpu.materialize_f64());
    }
}
