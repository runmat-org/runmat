use super::*;

pub(super) fn adjust_output_for_like(
    data: Value,
    prototype: &Value,
    precision: PrecisionSpec,
) -> BuiltinResult<Value> {
    if let Value::GpuTensor(prototype) = prototype {
        return match data {
            Value::Tensor(tensor) => tensor_to_gpu_value(tensor, prototype),
            Value::CharArray(_) => Err(fread_error_with_detail(
                &FREAD_ERROR_INVALID_OPTION,
                "character output cannot be returned on the GPU via 'like'",
            )),
            other => Ok(other),
        };
    }

    let result = (|| -> Result<Value, String> {
        match prototype {
            Value::LogicalArray(_) | Value::Bool(_) => convert_to_logical_value(data),
            Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
                if !matches!(precision.output, OutputKind::Char) {
                    return Err(
                        "fread: character prototypes require a character precision such as '*char'"
                            .to_string(),
                    );
                }
                ensure_char_result(data)
            }
            Value::Tensor(tensor) => tensor_to_numeric_like(data, tensor.numeric_dtype()),
            Value::Int(value) => tensor_to_numeric_like(
                data,
                IntegerStorage::from_scalar(value.clone()).numeric_dtype(),
            ),
            Value::Num(_) => tensor_to_numeric_like(data, NumericDType::F64),
            Value::ComplexTensor(_) | Value::Complex(_, _) => {
                Err("fread: complex prototypes are not supported yet".to_string())
            }
            Value::Cell(_) => Err("fread: cell prototypes are not supported".to_string()),
            _ => Ok(data),
        }
    })();
    map_string_result(result, &FREAD_ERROR_INVALID_OPTION)
}

pub(super) fn tensor_to_numeric_like(data: Value, dtype: NumericDType) -> Result<Value, String> {
    match data {
        Value::Tensor(tensor) => Ok(Value::Tensor(tensor::coerce_tensor_dtype(tensor, dtype))),
        Value::CharArray(_) => Err(
            "fread: character output cannot be converted to a numeric 'like' prototype".to_string(),
        ),
        other => Ok(other),
    }
}

pub(super) fn ensure_char_result(data: Value) -> Result<Value, String> {
    match data {
        Value::CharArray(_) => Ok(data),
        _ => Err("fread: expected character output when using a character prototype".to_string()),
    }
}

pub(super) fn tensor_to_gpu_value(
    tensor: Tensor,
    prototype: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Value> {
    use runmat_accelerate_api::{GpuTensorStorage, IntegerElementType, ProviderPrecision};

    let fail = |message: &str| fread_error_with_detail(&FREAD_ERROR_INVALID_OPTION, message);
    if prototype.physical().storage() != GpuTensorStorage::Real {
        return Err(fail("complex resident 'like' prototypes are not supported"));
    }
    let provider = gpu_helpers::exact_provider_for_handle(prototype)
        .ok_or_else(|| fail("resident 'like' prototype has no registered owning provider"))?;
    if prototype.view().is_logical() {
        let bits = (0..tensor.len())
            .map(|index| {
                u8::from(
                    tensor
                        .numeric_value_at(index)
                        .expect("validated tensor index")
                        .materialize_f64()
                        != 0.0,
                )
            })
            .collect::<Vec<_>>();
        let output = runmat_accelerate_api::upload_logical_protecting(
            provider,
            &runmat_accelerate_api::HostLogicalTensorView {
                data: &bits,
                shape: tensor.shape(),
            },
            &[prototype],
        )
        .map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                fail("logical Like upload failed"),
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        })?;
        return Ok(Value::GpuTensor(output));
    }
    let expected_integer = runmat_accelerate_api::handle_integer_type(prototype);
    let expected_logical = false;
    let expected_precision = runmat_accelerate_api::handle_precision(prototype);
    let tensor = if let Some(integer) = expected_integer {
        let dtype = match integer {
            IntegerElementType::I8 => NumericDType::I8,
            IntegerElementType::I16 => NumericDType::I16,
            IntegerElementType::I32 => NumericDType::I32,
            IntegerElementType::I64 => NumericDType::I64,
            IntegerElementType::U8 => NumericDType::U8,
            IntegerElementType::U16 => NumericDType::U16,
            IntegerElementType::U32 => NumericDType::U32,
            IntegerElementType::U64 => NumericDType::U64,
        };
        tensor::coerce_tensor_dtype(tensor, dtype)
    } else {
        let dtype = match expected_precision {
            Some(ProviderPrecision::F32) => NumericDType::F32,
            Some(ProviderPrecision::F64) => NumericDType::F64,
            None => {
                return Err(fail(
                    "resident 'like' prototype is missing precision metadata",
                ))
            }
        };
        tensor::coerce_tensor_dtype(tensor, dtype)
    };
    let expected_shape = tensor.shape().to_vec();
    let output = gpu_helpers::upload_tensor_protecting(provider, &tensor, &[prototype]).map_err(
        |error| {
            crate::runtime_error::with_provider_execution_source(
                fail("prototype owner upload failed"),
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        },
    )?;
    if let Err(message) = validate_resident_like_output(
        prototype,
        &output,
        &expected_shape,
        provider,
        expected_precision,
        expected_integer,
        expected_logical,
    ) {
        return Err(gpu_helpers::reject_gpu_output(output, fail(&message)));
    }
    Ok(Value::GpuTensor(output))
}

pub(super) fn validate_resident_like_output(
    prototype: &runmat_accelerate_api::GpuTensorHandle,
    output: &runmat_accelerate_api::GpuTensorHandle,
    expected_shape: &[usize],
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    expected_precision: Option<runmat_accelerate_api::ProviderPrecision>,
    expected_integer: Option<runmat_accelerate_api::IntegerElementType>,
    expected_logical: bool,
) -> Result<(), String> {
    let aliases_prototype = output.shares_allocation_with(prototype);
    let valid = !aliases_prototype
        && output.device_id() == prototype.device_id()
        && output.shape() == expected_shape
        && runmat_accelerate_api::provider_for_handle(output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output) == expected_precision
        && runmat_accelerate_api::handle_integer_type(output) == expected_integer
        && runmat_accelerate_api::handle_is_logical(output) == expected_logical;
    if valid {
        return Ok(());
    }
    Err("fread: resident 'like' upload returned an invalid or aliased handle".to_string())
}

pub(super) fn convert_to_logical_value(data: Value) -> Result<Value, String> {
    match data {
        Value::LogicalArray(_) => Ok(data),
        Value::Tensor(tensor) => {
            let bits = (0..tensor.len())
                .map(|index| {
                    let value = tensor
                        .numeric_value_at(index)
                        .expect("validated fread numeric tensor index");
                    match value {
                        NumericScalar::F64(value) => u8::from(value != 0.0),
                        NumericScalar::F32(value) => u8::from(value != 0.0),
                        value => u8::from(
                            !value
                                .into_int_value()
                                .expect("non-floating numeric scalar is integer")
                                .is_zero(),
                        ),
                    }
                })
                .collect();
            LogicalArray::new(bits, tensor.shape().to_vec())
                .map(Value::LogicalArray)
                .map_err(|e| format!("fread: {e}"))
        }
        Value::CharArray(ca) => {
            let total = ca.rows.saturating_mul(ca.cols);
            let mut bits = Vec::with_capacity(total);
            for c in 0..ca.cols {
                for r in 0..ca.rows {
                    let idx = r * ca.cols + c;
                    let ch = ca.data()[idx];
                    bits.push(if ch != runmat_value::Utf16CodeUnit::from_raw(0) {
                        1
                    } else {
                        0
                    });
                }
            }
            LogicalArray::new(bits, vec![ca.rows, ca.cols])
                .map(Value::LogicalArray)
                .map_err(|e| format!("fread: {e}"))
        }
        _ => Err(
            "fread: logical prototypes require numeric or character output from the read"
                .to_string(),
        ),
    }
}

// Limits a single synchronous Vec fill/transposition run; the execution
// service, not this value, decides when a charged chunk yields a turn.
pub(super) const STORAGE_FILL_CHUNK: usize = 4096;

pub(super) async fn finalize_numeric(
    size_spec: &SizeSpec,
    count_read: usize,
    mut values: NumericStorage,
    cooperation: &mut super::super::cooperation::FileCooperation,
) -> BuiltinResult<(NumericStorage, usize, usize)> {
    let (rows, cols, target) = match size_spec {
        SizeSpec::All | SizeSpec::Count(_) => (count_read, 1, count_read),
        SizeSpec::Matrix {
            rows,
            cols: Some(c),
        } => (*rows, *c, rows.saturating_mul(*c)),
        SizeSpec::Matrix { rows, cols: None } => {
            if *rows == 0 {
                (0, 0, 0)
            } else {
                let cols = if count_read == 0 {
                    0
                } else {
                    count_read.div_ceil(*rows)
                };
                (*rows, cols, rows.saturating_mul(cols))
            }
        }
    };
    resize_numeric_storage(&mut values, target, cooperation).await?;
    Ok((values, rows, cols))
}

pub(super) async fn resize_numeric_storage(
    storage: &mut NumericStorage,
    len: usize,
    cooperation: &mut super::super::cooperation::FileCooperation,
) -> BuiltinResult<()> {
    macro_rules! resize {
        ($values:expr, $zero:expr, $width:expr) => {{
            if $values.len() < len {
                $values.try_reserve(len - $values.len()).map_err(|error| {
                    build_runtime_error("fread: output storage cannot be allocated")
                        .with_builtin(BUILTIN_NAME)
                        .with_source(error)
                        .build()
                })?;
                while $values.len() < len {
                    let end = len.min($values.len().saturating_add(STORAGE_FILL_CHUNK));
                    let added = end - $values.len();
                    $values.resize(end, $zero);
                    cooperation.completed(added.saturating_mul($width)).await?;
                }
            } else {
                $values.truncate(len);
            }
        }};
    }
    match storage {
        NumericStorage::F64(values) => resize!(values, 0.0, 8),
        NumericStorage::F32(values) => resize!(values, 0.0, 4),
        NumericStorage::I8(values) => resize!(values, 0, 1),
        NumericStorage::I16(values) => resize!(values, 0, 2),
        NumericStorage::I32(values) => resize!(values, 0, 4),
        NumericStorage::I64(values) => resize!(values, 0, 8),
        NumericStorage::U8(values) => resize!(values, 0, 1),
        NumericStorage::U16(values) => resize!(values, 0, 2),
        NumericStorage::U32(values) => resize!(values, 0, 4),
        NumericStorage::U64(values) => resize!(values, 0, 8),
    }
    Ok(())
}
