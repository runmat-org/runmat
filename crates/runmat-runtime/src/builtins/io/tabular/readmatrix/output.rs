use super::*;

pub(super) fn finalize_output(tensor: Tensor, options: &ReadMatrixOptions) -> BuiltinResult<Value> {
    match &options.output_template {
        OutputTemplate::Numeric(dtype) => {
            let tensor = tensor::coerce_tensor_dtype(tensor, *dtype);
            Ok(Value::Tensor(tensor))
        }
        OutputTemplate::Logical => tensor_to_logical(tensor),
        OutputTemplate::Like(proto) => finalize_like(tensor, proto),
    }
}

fn tensor_to_logical(tensor: Tensor) -> BuiltinResult<Value> {
    let shape = tensor.shape().to_vec();
    let storage = tensor.into_numeric_storage().map_err(|error| {
        readmatrix_error_with(
            &READMATRIX_ERROR_TENSOR_BUILD,
            format!("readmatrix: {error}"),
        )
    })?;
    let NumericStorage::F64(values) = storage else {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_TENSOR_BUILD,
            "readmatrix: parsed logical source was not double",
        ));
    };
    let data = values
        .into_iter()
        .map(|value| if value == 0.0 { 0 } else { 1 })
        .collect();
    let logical = LogicalArray::new(data, shape).map_err(|e| {
        readmatrix_error_with(&READMATRIX_ERROR_TENSOR_BUILD, format!("readmatrix: {e}"))
    })?;
    Ok(Value::LogicalArray(logical))
}

pub(super) fn finalize_like(tensor: Tensor, proto: &Value) -> BuiltinResult<Value> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) => tensor_to_logical(tensor),
        Value::GpuTensor(handle) => tensor_to_gpu(tensor, handle),
        Value::Tensor(prototype) => Ok(Value::Tensor(tensor::coerce_tensor_dtype(
            tensor,
            prototype.numeric_dtype(),
        ))),
        Value::Int(value) => {
            tensor_to_integer_like(tensor, &IntegerStorage::from_scalar(value.clone()))
        }
        Value::Num(_) => Ok(Value::Tensor(tensor)),
        Value::ComplexTensor(_) | Value::Complex(_, _) => Ok(Value::Tensor(tensor)),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => Ok(Value::Tensor(tensor)),
        Value::Cell(_) => Ok(Value::Tensor(tensor)),
        _ => Ok(Value::Tensor(tensor)),
    }
}

fn tensor_to_integer_like(tensor: Tensor, prototype: &IntegerStorage) -> BuiltinResult<Value> {
    if tensor.numeric_dtype() == prototype.numeric_dtype() {
        return Ok(Value::Tensor(tensor));
    }
    let shape = tensor.shape().to_vec();
    let storage = tensor.into_numeric_storage().map_err(|e| {
        readmatrix_error_with(&READMATRIX_ERROR_TENSOR_BUILD, format!("readmatrix: {e}"))
    })?;
    let NumericStorage::F64(values) = storage else {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_TENSOR_BUILD,
            "readmatrix: parsed integer-like source was not double",
        ));
    };
    let tensor =
        crate::builtins::common::tensor::integer_tensor_from_f64_like(prototype, values, &shape)
            .map_err(|e| {
                readmatrix_error_with(&READMATRIX_ERROR_TENSOR_BUILD, format!("readmatrix: {e}"))
            })?;
    Ok(Value::Tensor(tensor))
}

pub(super) fn tensor_to_gpu(
    tensor: Tensor,
    prototype: &runmat_accelerate_api::GpuTensorHandle,
) -> BuiltinResult<Value> {
    use runmat_accelerate_api::ProviderPrecision;

    let owner = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        readmatrix_error_with(
            &READMATRIX_ERROR_OUTPUT_TYPE,
            "readmatrix: no acceleration provider owns the Like prototype",
        )
    })?;
    let expected_logical = prototype.view().is_logical();
    let expected_integer = if expected_logical {
        None
    } else {
        runmat_accelerate_api::handle_integer_type(prototype)
    };
    let expected_precision = runmat_accelerate_api::handle_precision(prototype);
    if expected_logical {
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
            owner,
            &runmat_accelerate_api::HostLogicalTensorView {
                data: &bits,
                shape: tensor.shape(),
            },
            &[prototype],
        )
        .map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                readmatrix_error_with(
                    &READMATRIX_ERROR_OUTPUT_TYPE,
                    format!("readmatrix: logical Like upload failed: {error}"),
                ),
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        })?;
        let output = output
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .map_err(|error| {
                crate::runtime_error::with_provider_execution_source(
                    readmatrix_error_with(
                        &READMATRIX_ERROR_OUTPUT_TYPE,
                        "readmatrix: Like output publication failed",
                    ),
                    runmat_accelerate_api::ProviderExecutionError::from_source(error),
                )
            })?;
        return checked_like_result(output, prototype, tensor.shape());
    }
    let target_dtype = if let Some(integer) = expected_integer {
        integer_element_dtype(integer)
    } else {
        match expected_precision {
            Some(ProviderPrecision::F32) => NumericDType::F32,
            Some(ProviderPrecision::F64) => NumericDType::F64,
            None => {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_OUTPUT_TYPE,
                    "readmatrix: the Like prototype is missing numeric class metadata",
                ))
            }
        }
    };
    let tensor = if tensor.numeric_dtype() == target_dtype {
        tensor
    } else {
        tensor::coerce_tensor_dtype(tensor, target_dtype)
    };
    let expected_shape = tensor.shape().to_vec();
    let mut uploaded = gpu_helpers::upload_tensor_protecting(owner, &tensor, &[prototype])
        .map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                readmatrix_error_with(
                    &READMATRIX_ERROR_OUTPUT_TYPE,
                    format!(
                        "readmatrix: failed to restore Like output to its owning provider: {error}"
                    ),
                ),
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        })?;
    uploaded = uploaded
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                readmatrix_error_with(
                    &READMATRIX_ERROR_OUTPUT_TYPE,
                    "readmatrix: Like output publication failed",
                ),
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        })?;
    checked_like_result(uploaded, prototype, &expected_shape)
}

pub(super) fn checked_like_result(
    uploaded: runmat_accelerate_api::GpuTensorHandle,
    prototype: &runmat_accelerate_api::GpuTensorHandle,
    expected_shape: &[usize],
) -> BuiltinResult<Value> {
    use runmat_accelerate_api::GpuTensorStorage;

    let expected_owner = gpu_helpers::exact_provider_for_handle(prototype);
    let expected_logical = runmat_accelerate_api::handle_is_logical(prototype);
    let expected_integer = runmat_accelerate_api::handle_integer_type(prototype);
    let expected_precision = runmat_accelerate_api::handle_precision(prototype);
    let valid = !uploaded.shares_allocation_with(prototype)
        && uploaded.device_id() == prototype.device_id()
        && uploaded.shape() == expected_shape
        && gpu_helpers::exact_provider_for_handle(&uploaded).is_some_and(|uploaded_owner| {
            expected_owner.is_some_and(|expected_owner| {
                runmat_accelerate_api::same_provider_instance(uploaded_owner, expected_owner)
            })
        })
        && runmat_accelerate_api::handle_storage(&uploaded) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(&uploaded) == expected_integer
        && runmat_accelerate_api::handle_precision(&uploaded) == expected_precision
        && runmat_accelerate_api::handle_is_logical(&uploaded) == expected_logical;
    if !valid {
        return Err(gpu_helpers::reject_gpu_output(uploaded, readmatrix_error_with(
            &READMATRIX_ERROR_OUTPUT_TYPE,
            "readmatrix: the Like provider returned an invalid class, shape, owner, or storage representation",
        )));
    }
    Ok(Value::GpuTensor(uploaded))
}

pub(super) fn integer_element_dtype(
    integer: runmat_accelerate_api::IntegerElementType,
) -> NumericDType {
    use runmat_accelerate_api::IntegerElementType;
    match integer {
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
