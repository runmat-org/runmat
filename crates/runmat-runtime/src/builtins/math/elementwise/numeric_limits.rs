//! MATLAB-compatible numeric limit query builtins.

use runmat_builtins::{
    FloatingLimitKind, IntegerLimitKind, NUMERIC_LIMIT_ERROR_INTERNAL,
    NUMERIC_LIMIT_ERROR_INVALID_CLASS, NUMERIC_LIMIT_ERROR_INVALID_SYNTAX,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, NumericDType, SparseTensor,
    Tensor, Value,
};

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runtime_builtin(
    name = "intmax",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn intmax_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    integer_limit(rest, IntegerLimitKind::Maximum, "intmax")
}

#[runtime_builtin(
    name = "intmin",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn intmin_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    integer_limit(rest, IntegerLimitKind::Minimum, "intmin")
}

#[runtime_builtin(
    name = "realmax",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn realmax_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    floating_limit(rest, FloatingLimitKind::LargestFinite, "realmax")
}

#[runtime_builtin(
    name = "realmin",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn realmin_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    floating_limit(rest, FloatingLimitKind::SmallestNormal, "realmin")
}

#[runtime_builtin(
    name = "flintmax",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::elementwise::numeric_limits"
)]
fn flintmax_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    floating_limit(
        rest,
        FloatingLimitKind::LargestConsecutiveInteger,
        "flintmax",
    )
}

fn floating_limit(
    args: Vec<Value>,
    kind: FloatingLimitKind,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match args.as_slice() {
        [] => floating_limit_value(
            runmat_types::NumericClass::Double,
            false,
            false,
            kind,
            builtin,
        ),
        [class] if text_value(class).is_some() => {
            let text = text_value(class).expect("guarded text value");
            let class = parse_floating_class_name(&text, builtin)?;
            floating_limit_value(class, false, false, kind, builtin)
        }
        [keyword, prototype]
            if text_value(keyword).is_some_and(|text| text.eq_ignore_ascii_case("like")) =>
        {
            floating_limit_like(prototype, kind, builtin)
        }
        _ => Err(limit_syntax_error(
            builtin,
            "expected no arguments, a floating-point class name, or \"like\", prototype",
        )),
    }
}

fn parse_floating_class_name(
    text: &str,
    builtin: &'static str,
) -> BuiltinResult<runmat_types::NumericClass> {
    runmat_types::NumericClass::from_class_name(text.trim())
        .filter(|class| {
            matches!(
                class,
                runmat_types::NumericClass::Double | runmat_types::NumericClass::Single
            )
        })
        .ok_or_else(|| {
            limit_error(
                builtin,
                format!("unsupported floating-point class '{text}'"),
            )
        })
}

fn floating_limit_like(
    prototype: &Value,
    kind: FloatingLimitKind,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match prototype {
        Value::Num(_) => floating_limit_value(
            runmat_types::NumericClass::Double,
            false,
            false,
            kind,
            builtin,
        ),
        Value::Tensor(tensor) => floating_limit_value(
            floating_class(tensor.numeric_dtype(), builtin)?,
            false,
            false,
            kind,
            builtin,
        ),
        Value::Complex(_, _) => floating_limit_value(
            runmat_types::NumericClass::Double,
            true,
            false,
            kind,
            builtin,
        ),
        Value::ComplexTensor(tensor) => floating_limit_value(
            floating_class(tensor.numeric_dtype(), builtin)?,
            true,
            false,
            kind,
            builtin,
        ),
        Value::SparseTensor(tensor) => {
            let class = floating_class(
                tensor
                    .numeric_dtype()
                    .ok_or_else(|| invalid_floating_prototype(builtin))?,
                builtin,
            )?;
            floating_limit_value(class, tensor.is_complex(), true, kind, builtin)
        }
        Value::GpuTensor(handle) => floating_gpu_limit_like(handle, kind, builtin),
        _ => Err(invalid_floating_prototype(builtin)),
    }
}

fn floating_class(
    dtype: NumericDType,
    builtin: &'static str,
) -> BuiltinResult<runmat_types::NumericClass> {
    match dtype {
        NumericDType::F64 => Ok(runmat_types::NumericClass::Double),
        NumericDType::F32 => Ok(runmat_types::NumericClass::Single),
        _ => Err(invalid_floating_prototype(builtin)),
    }
}

fn floating_limit_value(
    class: runmat_types::NumericClass,
    complex: bool,
    sparse: bool,
    kind: FloatingLimitKind,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    let shape = vec![1, 1];
    match (class, complex, sparse) {
        (runmat_types::NumericClass::Double, false, false) => {
            Ok(Value::Num(floating_limit_f64(kind)))
        }
        (runmat_types::NumericClass::Single, false, false) => {
            Tensor::from_f32(vec![floating_limit_f32(kind)], shape)
                .map(Value::Tensor)
                .map_err(|error| internal_limit_error(builtin, error))
        }
        (runmat_types::NumericClass::Double, true, false) => {
            Ok(Value::Complex(floating_limit_f64(kind), 0.0))
        }
        (runmat_types::NumericClass::Single, true, false) => {
            ComplexTensor::from_f32(vec![(floating_limit_f32(kind), 0.0)], shape)
                .map(Value::ComplexTensor)
                .map_err(|error| internal_limit_error(builtin, error))
        }
        (runmat_types::NumericClass::Double, false, true) => {
            SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![floating_limit_f64(kind)])
                .map(Value::SparseTensor)
                .map_err(|error| internal_limit_error(builtin, error))
        }
        (runmat_types::NumericClass::Single, false, true) => {
            SparseTensor::new_f32(1, 1, vec![0, 1], vec![0], vec![floating_limit_f32(kind)])
                .map(Value::SparseTensor)
                .map_err(|error| internal_limit_error(builtin, error))
        }
        (runmat_types::NumericClass::Double, true, true) => SparseTensor::new_complex(
            1,
            1,
            vec![0, 1],
            vec![0],
            vec![(floating_limit_f64(kind), 0.0)],
        )
        .map(Value::SparseTensor)
        .map_err(|error| internal_limit_error(builtin, error)),
        (runmat_types::NumericClass::Single, true, true) => SparseTensor::new_complex_f32(
            1,
            1,
            vec![0, 1],
            vec![0],
            vec![(floating_limit_f32(kind), 0.0)],
        )
        .map(Value::SparseTensor)
        .map_err(|error| internal_limit_error(builtin, error)),
        _ => Err(invalid_floating_prototype(builtin)),
    }
}

fn floating_limit_f64(kind: FloatingLimitKind) -> f64 {
    match kind {
        FloatingLimitKind::SmallestNormal => f64::MIN_POSITIVE,
        FloatingLimitKind::LargestFinite => f64::MAX,
        FloatingLimitKind::LargestConsecutiveInteger => 2f64.powi(53),
    }
}

fn floating_limit_f32(kind: FloatingLimitKind) -> f32 {
    match kind {
        FloatingLimitKind::SmallestNormal => f32::MIN_POSITIVE,
        FloatingLimitKind::LargestFinite => f32::MAX,
        FloatingLimitKind::LargestConsecutiveInteger => 2f32.powi(24),
    }
}

fn floating_gpu_limit_like(
    prototype: &runmat_accelerate_api::GpuTensorHandle,
    kind: FloatingLimitKind,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    use runmat_accelerate_api::{GpuTensorStorage, ProviderPrecision};

    let precision = runmat_accelerate_api::handle_precision(prototype)
        .ok_or_else(|| invalid_floating_prototype(builtin))?;
    let storage = runmat_accelerate_api::handle_storage(prototype);
    if !matches!(
        storage,
        GpuTensorStorage::Real | GpuTensorStorage::ComplexInterleaved
    ) || runmat_accelerate_api::handle_integer_type(prototype).is_some()
        || runmat_accelerate_api::handle_is_logical(prototype)
        || !gpu_helpers::gpu_class_metadata_matches(prototype, Some(precision), None, false)
    {
        return Err(limit_error(
            builtin,
            "floating-point gpuArray prototype has contradictory class metadata",
        ));
    }
    let provider = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        limit_error(
            builtin,
            "floating-point gpuArray prototype has no registered provider",
        )
    })?;
    let shape = vec![1, 1];
    let output = match (precision, storage) {
        (ProviderPrecision::F64, GpuTensorStorage::Real) => gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![floating_limit_f64(kind)], shape.clone())
                .map_err(|error| internal_limit_error(builtin, error))?,
        ),
        (ProviderPrecision::F32, GpuTensorStorage::Real) => gpu_helpers::upload_tensor(
            provider,
            &Tensor::from_f32(vec![floating_limit_f32(kind)], shape.clone())
                .map_err(|error| internal_limit_error(builtin, error))?,
        ),
        (ProviderPrecision::F64, GpuTensorStorage::ComplexInterleaved) => {
            gpu_helpers::upload_complex_tensor(
                provider,
                &ComplexTensor::new(vec![(floating_limit_f64(kind), 0.0)], shape.clone())
                    .map_err(|error| internal_limit_error(builtin, error))?,
            )
            .map_err(|error| error.message().to_string())
        }
        (ProviderPrecision::F32, GpuTensorStorage::ComplexInterleaved) => {
            gpu_helpers::upload_complex_tensor(
                provider,
                &ComplexTensor::from_f32(vec![(floating_limit_f32(kind), 0.0)], shape.clone())
                    .map_err(|error| internal_limit_error(builtin, error))?,
            )
            .map_err(|error| error.message().to_string())
        }
    }
    .map_err(|error| {
        internal_limit_error(builtin, format!("GPU limit creation failed: {error}"))
    })?;
    let valid = output.shape == shape
        && output.device_id == prototype.device_id
        && !gpu_helpers::same_gpu_handle(prototype, &output)
        && gpu_helpers::exact_provider_for_handle(&output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(&output) == storage
        && runmat_accelerate_api::handle_precision(&output) == Some(precision)
        && gpu_helpers::gpu_class_metadata_matches(&output, Some(precision), None, false);
    if !valid {
        gpu_helpers::free_unprotected_exact_owner(&output, &[prototype]);
        return Err(internal_limit_error(
            builtin,
            "GPU limit creation returned an invalid provider result",
        ));
    }
    let mut output = output;
    let provenance = runmat_accelerate_api::handle_provenance(prototype)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    runmat_accelerate_api::set_handle_provenance(&mut output, provenance);
    Ok(gpu_helpers::resident_gpu_value(output))
}

fn integer_limit(
    args: Vec<Value>,
    kind: IntegerLimitKind,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match args.as_slice() {
        [] => Ok(Value::Int(limit_scalar(NumericDType::I32, kind))),
        [class] if text_value(class).is_some() => {
            let text = text_value(class).expect("guarded text value");
            let class = runmat_types::NumericClass::from_class_name(text.trim())
                .filter(|class| {
                    !matches!(
                        class,
                        runmat_types::NumericClass::Double | runmat_types::NumericClass::Single
                    )
                })
                .ok_or_else(|| {
                    limit_error(builtin, format!("unsupported integer class '{text}'"))
                })?;
            let dtype = NumericDType::from(class);
            Ok(Value::Int(limit_scalar(dtype, kind)))
        }
        [keyword, prototype]
            if text_value(keyword).is_some_and(|text| text.eq_ignore_ascii_case("like")) =>
        {
            integer_limit_like(prototype, kind, builtin)
        }
        _ => Err(limit_syntax_error(
            builtin,
            "expected no arguments, an integer class name, or \"like\", prototype",
        )),
    }
}

fn integer_limit_like(
    prototype: &Value,
    kind: IntegerLimitKind,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match prototype {
        Value::Int(value) => Ok(Value::Int(limit_scalar(value.numeric_dtype(), kind))),
        Value::Tensor(tensor) => {
            let dtype = tensor.numeric_dtype();
            if matches!(dtype, NumericDType::F32 | NumericDType::F64) {
                return Err(invalid_integer_prototype(builtin));
            }
            Ok(Value::Int(limit_scalar(dtype, kind)))
        }
        Value::ComplexTensor(tensor) => {
            let Some(prototype_storage) = tensor.integer_storage() else {
                return Err(invalid_integer_prototype(builtin));
            };
            let value = limit_scalar(prototype_storage.real.numeric_dtype(), kind);
            let real = IntegerStorage::from_scalar(value);
            let imag = real.zeros_like(1);
            let storage = IntegerComplexStorage::new(real, imag)
                .map_err(|error| limit_error(builtin, error))?;
            ComplexTensor::new_integer(storage, vec![1, 1])
                .map(Value::ComplexTensor)
                .map_err(|error| limit_error(builtin, error))
        }
        Value::GpuTensor(handle) => {
            let Some(element_type) = runmat_accelerate_api::handle_integer_type(handle) else {
                return Err(invalid_integer_prototype(builtin));
            };
            let prototype_storage = runmat_accelerate_api::handle_storage(handle);
            if !matches!(
                prototype_storage,
                runmat_accelerate_api::GpuTensorStorage::Real
                    | runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            ) || runmat_accelerate_api::handle_precision(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
                || !gpu_helpers::gpu_class_metadata_matches(handle, None, Some(element_type), false)
            {
                return Err(limit_error(
                    builtin,
                    "integer gpuArray prototype has contradictory class metadata",
                ));
            }
            let dtype = dtype_from_integer_element_type(element_type);
            let storage = IntegerStorage::from_scalar(limit_scalar(dtype, kind));
            let shape = [1usize, 1usize];
            let provider = gpu_helpers::exact_provider_for_handle(handle).ok_or_else(|| {
                limit_error(
                    builtin,
                    "integer gpuArray prototype has no registered provider",
                )
            })?;
            let provenance = runmat_accelerate_api::handle_provenance(handle)
                .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            let input_metadata = gpu_helpers::snapshot_handle_metadata(handle);
            let output = match prototype_storage {
                runmat_accelerate_api::GpuTensorStorage::Real => {
                    let view = integer_tensor_view(&storage, &shape);
                    provider
                        .upload_integer(&view)
                        .map_err(|error| error.to_string())
                }
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved => {
                    let imaginary = storage.zeros_like(1);
                    let complex = ComplexTensor::new_integer(
                        IntegerComplexStorage::new(storage, imaginary)
                            .map_err(|error| internal_limit_error(builtin, error))?,
                        shape.to_vec(),
                    )
                    .map_err(|error| internal_limit_error(builtin, error))?;
                    gpu_helpers::upload_complex_tensor(provider, &complex)
                        .map_err(|error| error.message().to_string())
                }
            };
            gpu_helpers::restore_handle_metadata(handle, &input_metadata);
            let output = output.map_err(|error| {
                limit_error(builtin, format!("GPU limit creation failed: {error}"))
            })?;
            let valid = output.shape == shape
                && output.device_id == handle.device_id
                && !gpu_helpers::same_gpu_handle(handle, &output)
                && gpu_helpers::exact_provider_for_handle(&output)
                    .is_some_and(|owner| std::ptr::eq(owner, provider))
                && runmat_accelerate_api::handle_storage(&output) == prototype_storage
                && runmat_accelerate_api::handle_integer_type(&output) == Some(element_type)
                && runmat_accelerate_api::handle_precision(&output).is_none()
                && !runmat_accelerate_api::handle_is_logical(&output)
                && gpu_helpers::gpu_class_metadata_matches(
                    &output,
                    None,
                    Some(element_type),
                    false,
                );
            if !valid {
                gpu_helpers::free_unprotected_exact_owner(&output, &[handle]);
                return Err(limit_error(
                    builtin,
                    "GPU limit creation returned an invalid provider result",
                ));
            }
            let mut output = output;
            runmat_accelerate_api::set_handle_provenance(&mut output, provenance);
            Ok(gpu_helpers::resident_gpu_value(output))
        }
        _ => Err(invalid_integer_prototype(builtin)),
    }
}

fn text_value(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        _ => None,
    }
}

fn limit_scalar(dtype: NumericDType, kind: IntegerLimitKind) -> IntValue {
    match (dtype, kind) {
        (NumericDType::I8, IntegerLimitKind::Minimum) => IntValue::I8(i8::MIN),
        (NumericDType::I8, IntegerLimitKind::Maximum) => IntValue::I8(i8::MAX),
        (NumericDType::I16, IntegerLimitKind::Minimum) => IntValue::I16(i16::MIN),
        (NumericDType::I16, IntegerLimitKind::Maximum) => IntValue::I16(i16::MAX),
        (NumericDType::I32, IntegerLimitKind::Minimum) => IntValue::I32(i32::MIN),
        (NumericDType::I32, IntegerLimitKind::Maximum) => IntValue::I32(i32::MAX),
        (NumericDType::I64, IntegerLimitKind::Minimum) => IntValue::I64(i64::MIN),
        (NumericDType::I64, IntegerLimitKind::Maximum) => IntValue::I64(i64::MAX),
        (NumericDType::U8, IntegerLimitKind::Minimum) => IntValue::U8(0),
        (NumericDType::U8, IntegerLimitKind::Maximum) => IntValue::U8(u8::MAX),
        (NumericDType::U16, IntegerLimitKind::Minimum) => IntValue::U16(0),
        (NumericDType::U16, IntegerLimitKind::Maximum) => IntValue::U16(u16::MAX),
        (NumericDType::U32, IntegerLimitKind::Minimum) => IntValue::U32(0),
        (NumericDType::U32, IntegerLimitKind::Maximum) => IntValue::U32(u32::MAX),
        (NumericDType::U64, IntegerLimitKind::Minimum) => IntValue::U64(0),
        (NumericDType::U64, IntegerLimitKind::Maximum) => IntValue::U64(u64::MAX),
        (NumericDType::F32 | NumericDType::F64, _) => {
            unreachable!("limit_scalar is only called for integer dtypes")
        }
    }
}

fn dtype_from_integer_element_type(
    element_type: runmat_accelerate_api::IntegerElementType,
) -> NumericDType {
    use runmat_accelerate_api::IntegerElementType;
    match element_type {
        IntegerElementType::I8 => NumericDType::I8,
        IntegerElementType::I16 => NumericDType::I16,
        IntegerElementType::I32 => NumericDType::I32,
        IntegerElementType::I64 => NumericDType::I64,
        IntegerElementType::U8 => NumericDType::U8,
        IntegerElementType::U16 => NumericDType::U16,
        IntegerElementType::U32 => NumericDType::U32,
        IntegerElementType::U64 => NumericDType::U64,
    }
}

fn integer_tensor_view<'a>(
    storage: &'a IntegerStorage,
    shape: &'a [usize],
) -> runmat_accelerate_api::HostIntegerTensorView<'a> {
    use runmat_accelerate_api::HostIntegerDataView;
    let data = match storage {
        IntegerStorage::I8(values) => HostIntegerDataView::I8(values),
        IntegerStorage::I16(values) => HostIntegerDataView::I16(values),
        IntegerStorage::I32(values) => HostIntegerDataView::I32(values),
        IntegerStorage::I64(values) => HostIntegerDataView::I64(values),
        IntegerStorage::U8(values) => HostIntegerDataView::U8(values),
        IntegerStorage::U16(values) => HostIntegerDataView::U16(values),
        IntegerStorage::U32(values) => HostIntegerDataView::U32(values),
        IntegerStorage::U64(values) => HostIntegerDataView::U64(values),
    };
    runmat_accelerate_api::HostIntegerTensorView { data, shape }
}

fn invalid_integer_prototype(builtin: &'static str) -> RuntimeError {
    limit_error(
        builtin,
        "like prototype must be an integer variable of class int8, int16, int32, int64, uint8, uint16, uint32, or uint64",
    )
}

fn invalid_floating_prototype(builtin: &'static str) -> RuntimeError {
    limit_error(builtin, "like prototype must have class double or single")
}

fn limit_syntax_error(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = NUMERIC_LIMIT_ERROR_INVALID_SYNTAX.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn limit_error(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = NUMERIC_LIMIT_ERROR_INVALID_CLASS.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn internal_limit_error(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = NUMERIC_LIMIT_ERROR_INTERNAL.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView};
    use runmat_value::Tensor;

    #[test]
    fn integer_limits_support_common_classes() {
        assert_eq!(
            intmax_builtin(Vec::new()).unwrap(),
            Value::Int(IntValue::I32(i32::MAX))
        );
        assert_eq!(
            intmin_builtin(vec![Value::from("uint16")]).unwrap(),
            Value::Int(IntValue::U16(0))
        );
        assert_eq!(
            intmax_builtin(vec![Value::from("uint32")]).unwrap(),
            Value::Int(IntValue::U32(u32::MAX))
        );
    }

    #[test]
    fn integer_limits_support_all_class_names_and_exact_wide_bounds() {
        let cases = [
            ("int8", IntValue::I8(i8::MIN), IntValue::I8(i8::MAX)),
            ("int16", IntValue::I16(i16::MIN), IntValue::I16(i16::MAX)),
            ("int32", IntValue::I32(i32::MIN), IntValue::I32(i32::MAX)),
            ("int64", IntValue::I64(i64::MIN), IntValue::I64(i64::MAX)),
            ("uint8", IntValue::U8(0), IntValue::U8(u8::MAX)),
            ("uint16", IntValue::U16(0), IntValue::U16(u16::MAX)),
            ("uint32", IntValue::U32(0), IntValue::U32(u32::MAX)),
            ("uint64", IntValue::U64(0), IntValue::U64(u64::MAX)),
        ];

        for (class, minimum, maximum) in cases {
            assert_eq!(
                intmin_builtin(vec![Value::from(class)]).unwrap(),
                Value::Int(minimum)
            );
            assert_eq!(
                intmax_builtin(vec![Value::from(class)]).unwrap(),
                Value::Int(maximum)
            );
        }
    }

    #[test]
    fn integer_limit_like_copies_every_integer_prototype_class() {
        let prototypes = [
            IntegerStorage::I8(vec![7]),
            IntegerStorage::I16(vec![7]),
            IntegerStorage::I32(vec![7]),
            IntegerStorage::I64(vec![9_007_199_254_740_993]),
            IntegerStorage::U8(vec![7]),
            IntegerStorage::U16(vec![7]),
            IntegerStorage::U32(vec![7]),
            IntegerStorage::U64(vec![9_007_199_254_740_993]),
        ];

        for storage in prototypes {
            let dtype = storage.numeric_dtype();
            let prototype = Tensor::new_integer(storage, vec![1, 1]).unwrap();
            assert_eq!(
                intmin_builtin(vec![Value::from("like"), Value::Tensor(prototype.clone())])
                    .unwrap(),
                Value::Int(limit_scalar(dtype, IntegerLimitKind::Minimum))
            );
            assert_eq!(
                intmax_builtin(vec![Value::from("like"), Value::Tensor(prototype)]).unwrap(),
                Value::Int(limit_scalar(dtype, IntegerLimitKind::Maximum))
            );
        }
    }

    #[test]
    fn integer_limit_like_copies_complexity_without_losing_uint64_max() {
        let prototype = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![9_007_199_254_740_993]),
                IntegerStorage::U64(vec![1]),
            )
            .unwrap(),
            vec![1, 1],
        )
        .unwrap();

        let output =
            intmax_builtin(vec![Value::from("like"), Value::ComplexTensor(prototype)]).unwrap();
        let Value::ComplexTensor(output) = output else {
            panic!("expected complex integer scalar")
        };
        assert_eq!(output.shape, vec![1, 1]);
        assert_eq!(
            output.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX]),
                    IntegerStorage::U64(vec![0]),
                )
                .unwrap()
            )
        );
    }

    #[test]
    fn integer_limit_like_rejects_noninteger_prototypes_and_bad_syntax() {
        assert!(intmin_builtin(vec![Value::from("like"), Value::Num(0.0)]).is_err());
        assert!(intmax_builtin(vec![Value::Bool(true)]).is_err());
        assert!(intmax_builtin(vec![Value::from("uint8"), Value::from("uint16")]).is_err());
        assert!(intmin_builtin(vec![Value::from("int")]).is_err());
    }

    #[test]
    fn integer_limit_like_preserves_gpu_class_and_wide_value() {
        test_support::with_test_provider(|provider| {
            let shape = [1usize, 1usize];
            let prototype = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[9_007_199_254_740_993]),
                    shape: &shape,
                })
                .expect("integer prototype upload");

            let output = intmax_builtin(vec![
                Value::from("like"),
                Value::GpuTensor(prototype.clone()),
            ])
            .expect("gpu intmax like");
            let Value::GpuTensor(output) = output else {
                panic!("expected resident integer output")
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&output),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let downloaded = block_on(provider.download_integer(&output)).expect("download");
            assert_eq!(downloaded.data, HostIntegerDataOwned::U64(vec![u64::MAX]));
            assert_eq!(downloaded.shape, vec![1, 1]);
            provider.free(&prototype).ok();
            provider.free(&output).ok();
        });
    }

    #[test]
    fn integer_limit_like_rejects_contradictory_resident_class_metadata() {
        test_support::with_test_provider(|provider| {
            let prototype = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[1]),
                    shape: &[1, 1],
                })
                .expect("integer prototype upload");
            runmat_accelerate_api::set_handle_class_identity(&prototype, "double");
            let error = intmax_builtin(vec![
                Value::from("like"),
                Value::GpuTensor(prototype.clone()),
            ])
            .expect_err("contradictory resident prototype must reject");
            assert!(error.message().contains("contradictory class metadata"));
            provider.free(&prototype).ok();
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn integer_limit_like_preserves_wgpu_class_and_wide_value() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let shape = [1usize, 1usize];
        let prototype = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::I64(&[9_007_199_254_740_993]),
                shape: &shape,
            })
            .expect("WGPU integer prototype upload");

        let output = intmin_builtin(vec![
            Value::from("like"),
            Value::GpuTensor(prototype.clone()),
        ])
        .expect("WGPU intmin like");
        let Value::GpuTensor(output) = output else {
            panic!("expected resident integer output")
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(runmat_accelerate_api::IntegerElementType::I64)
        );
        let downloaded = block_on(provider.download_integer(&output)).expect("download");
        assert_eq!(downloaded.data, HostIntegerDataOwned::I64(vec![i64::MIN]));
        provider.free(&prototype).ok();
        provider.free(&output).ok();
    }

    #[test]
    fn floating_limits_support_single_and_double() {
        assert_eq!(realmax_builtin(Vec::new()).unwrap(), Value::Num(f64::MAX));
        for (output, expected) in [
            (
                realmin_builtin(vec![Value::from("single")]).unwrap(),
                f32::MIN_POSITIVE,
            ),
            (
                flintmax_builtin(vec![Value::from("single")]).unwrap(),
                2f32.powi(24),
            ),
        ] {
            let Value::Tensor(output) = output else {
                panic!("single limit must retain native single storage")
            };
            assert_eq!(output.as_f32_slice(), Some([expected].as_slice()));
            assert_eq!(output.shape, vec![1, 1]);
        }
    }

    #[test]
    fn floating_limit_like_preserves_complexity_and_sparse_single_storage() {
        let complex = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let output = realmax_builtin(vec![Value::from("like"), Value::ComplexTensor(complex)])
            .expect("complex single like form");
        let Value::ComplexTensor(output) = output else {
            panic!("expected complex single scalar")
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        assert_eq!(output.shape, vec![1, 1]);

        let sparse = SparseTensor::new_f32(2, 2, vec![0, 1, 1], vec![0], vec![3.0]).unwrap();
        let output = realmin_builtin(vec![Value::from("like"), Value::SparseTensor(sparse)])
            .expect("sparse single like form");
        let Value::SparseTensor(output) = output else {
            panic!("expected sparse single scalar")
        };
        assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
        assert_eq!(output.as_f32_slice(), Some([f32::MIN_POSITIVE].as_slice()));
        assert_eq!((output.rows, output.cols), (1, 1));

        let sparse =
            SparseTensor::new_complex_f32(2, 1, vec![0, 1], vec![1], vec![(3.0, -4.0)]).unwrap();
        let output = flintmax_builtin(vec![Value::from("like"), Value::SparseTensor(sparse)])
            .expect("sparse complex single like form");
        let Value::SparseTensor(output) = output else {
            panic!("expected sparse complex single scalar")
        };
        assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
        assert!(output.is_complex());
        assert_eq!(
            output.as_complex_f32_slice(),
            Some([runmat_value::ComplexElement(2f32.powi(24), 0.0)].as_slice())
        );
    }

    #[test]
    fn floating_limit_like_preserves_resident_single_representation_and_intent() {
        test_support::with_f32_test_provider(|provider| {
            let prototype = Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).expect("prototype");
            let mut handle = gpu_helpers::upload_tensor(provider, &prototype).expect("upload");
            runmat_accelerate_api::set_handle_provenance(
                &mut handle,
                runmat_accelerate_api::GpuHandleProvenance::Explicit,
            );

            let output =
                realmax_builtin(vec![Value::from("like"), Value::GpuTensor(handle.clone())])
                    .expect("resident realmax like");
            let Value::GpuTensor(output) = output else {
                panic!("expected resident single output")
            };
            assert_eq!(output.shape, vec![1, 1]);
            assert_eq!(output.device_id, handle.device_id);
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output),
                runmat_accelerate_api::GpuTensorStorage::Real
            );
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(runmat_accelerate_api::ProviderPrecision::F32)
            );
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                &Value::GpuTensor(output.clone()),
            ))
            .expect("gather");
            let Value::Tensor(gathered) = gathered else {
                panic!("expected gathered single tensor")
            };
            assert_eq!(gathered.as_f32_slice(), Some([f32::MAX].as_slice()));
            provider.free(&handle).ok();
            provider.free(&output).ok();
        });

        test_support::with_f32_test_provider(|provider| {
            let prototype =
                ComplexTensor::from_f32(vec![(1.0, -2.0)], vec![1, 1]).expect("prototype");
            let handle = gpu_helpers::upload_complex_tensor(provider, &prototype).expect("upload");
            let output =
                flintmax_builtin(vec![Value::from("like"), Value::GpuTensor(handle.clone())])
                    .expect("resident complex flintmax like");
            let Value::GpuTensor(output) = output else {
                panic!("expected resident complex single output")
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(runmat_accelerate_api::ProviderPrecision::F32)
            );
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                &Value::GpuTensor(output.clone()),
            ))
            .expect("gather");
            let Value::ComplexTensor(gathered) = gathered else {
                panic!("expected gathered complex single tensor")
            };
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(
                gathered.as_f32_slice(),
                Some([runmat_value::ComplexElement(2f32.powi(24), 0.0)].as_slice())
            );
            provider.free(&handle).ok();
            provider.free(&output).ok();
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn floating_limit_like_preserves_wgpu_single_class_and_value() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let prototype = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("prototype");
        let handle = gpu_helpers::upload_tensor(provider, &prototype).expect("upload");

        let output = realmin_builtin(vec![Value::from("like"), Value::GpuTensor(handle.clone())])
            .expect("WGPU realmin like");
        let Value::GpuTensor(output) = output else {
            panic!("expected resident single output")
        };
        assert_eq!(
            runmat_accelerate_api::handle_precision(&output),
            Some(runmat_accelerate_api::ProviderPrecision::F32)
        );
        let gathered = block_on(provider.download(&output)).expect("download");
        assert_eq!(gathered.data, vec![f64::from(f32::MIN_POSITIVE)]);
        assert_eq!(gathered.shape, vec![1, 1]);
        provider.free(&handle).ok();
        provider.free(&output).ok();
    }
}
