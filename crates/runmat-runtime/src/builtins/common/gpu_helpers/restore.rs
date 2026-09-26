use super::*;

/// Restore a class-preserving host value to the provider that owns `source`.
///
/// If the owner cannot physically represent the floating class, the host value
/// is returned instead of relabelling it. Integer and logical storage retain
/// their exact metadata independently of the provider's floating precision.
pub fn restore_class_preserving_value(
    source: &GpuTensorHandle,
    value: Value,
    builtin: &str,
) -> crate::BuiltinResult<Value> {
    restore_class_preserving_value_protecting(source, value, builtin, &[])
}

/// Restore through the source owner without adopting or releasing any borrowed
/// operand. The source is protected even when it is absent from `protected`.
pub fn restore_class_preserving_value_protecting(
    source: &GpuTensorHandle,
    value: Value,
    builtin: &str,
    protected: &[&GpuTensorHandle],
) -> crate::BuiltinResult<Value> {
    let value = match value {
        Value::Num(number) => {
            let tensor = Tensor::new(vec![number], vec![1, 1]).map_err(|error| {
                build_runtime_error(format!("{builtin}: invalid scalar result: {error}"))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
            })?;
            return match restore_class_preserving_value_protecting(
                source,
                Value::Tensor(tensor),
                builtin,
                protected,
            )? {
                Value::Tensor(_) => Ok(Value::Num(number)),
                restored => Ok(restored),
            };
        }
        Value::Complex(real, imag) => {
            let tensor = ComplexTensor::new(vec![(real, imag)], vec![1, 1]).map_err(|error| {
                build_runtime_error(format!("{builtin}: invalid complex scalar result: {error}"))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
            })?;
            return match restore_class_preserving_value_protecting(
                source,
                Value::ComplexTensor(tensor),
                builtin,
                protected,
            )? {
                Value::ComplexTensor(_) => Ok(Value::Complex(real, imag)),
                restored => Ok(restored),
            };
        }
        Value::Bool(bit) => {
            let logical = LogicalArray::new(vec![u8::from(bit)], vec![1, 1]).map_err(|error| {
                build_runtime_error(format!("{builtin}: invalid logical scalar result: {error}"))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
            })?;
            return match restore_class_preserving_value_protecting(
                source,
                Value::LogicalArray(logical),
                builtin,
                protected,
            )? {
                Value::LogicalArray(_) => Ok(Value::Bool(bit)),
                restored => Ok(restored),
            };
        }
        value => value,
    };
    let provider = exact_provider_for_handle(source).ok_or_else(|| {
        build_runtime_error(format!(
            "{builtin}: no acceleration provider owns the input handle"
        ))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
    })?;
    let source_guard = HandleMetadataRestoreGuard::new(source);
    let protected = std::iter::once(source)
        .chain(protected.iter().copied())
        .collect::<Vec<_>>();
    let _operand_guards = protected
        .iter()
        .map(|handle| HandleMetadataRestoreGuard::new(handle))
        .collect::<Vec<_>>();
    let explicit = protected
        .iter()
        .any(|handle| runmat_accelerate_api::handle_is_explicit(handle));

    let (
        mut output,
        expected_shape,
        expected_storage,
        expected_precision,
        expected_integer,
        logical,
    ) = match &value {
        Value::Tensor(tensor) => {
            let expected_integer = tensor.integer_storage().map(integer_element_type);
            let expected_precision = if expected_integer.is_some() {
                None
            } else {
                Some(match tensor.numeric_dtype() {
                    NumericDType::F32 => ProviderPrecision::F32,
                    _ => ProviderPrecision::F64,
                })
            };
            let transfer = HostNumericTensorView {
                data: tensor_numeric_view(tensor),
                shape: &tensor.shape,
                storage: GpuTensorStorage::Real,
            };
            let output = match provider.upload_numeric(&transfer) {
                Ok(output) => output,
                Err(_) if expected_integer.is_none() && !explicit => return Ok(value),
                Err(error) => {
                    return Err(build_runtime_error(format!(
                        "{builtin}: failed to restore GPU result: {error}"
                    ))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build())
                }
            };
            let output = validate_numeric_upload(provider, output, transfer, &protected).map_err(
                |error| {
                    build_runtime_error(format!("{builtin}: {error}"))
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                },
            )?;
            (
                output,
                tensor.shape.clone(),
                GpuTensorStorage::Real,
                expected_precision,
                expected_integer,
                false,
            )
        }
        Value::LogicalArray(array) => {
            let storage = match provider.precision() {
                ProviderPrecision::F32 => {
                    NumericStorage::F32(array.data.iter().map(|bit| f32::from(*bit != 0)).collect())
                }
                ProviderPrecision::F64 => {
                    NumericStorage::F64(array.data.iter().map(|bit| f64::from(*bit != 0)).collect())
                }
            };
            let tensor =
                Tensor::from_numeric_storage(storage, array.shape.clone()).map_err(|error| {
                    build_runtime_error(format!("{builtin}: invalid logical result: {error}"))
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                })?;
            let output =
                upload_tensor_protecting(provider, &tensor, &protected).map_err(|error| {
                    build_runtime_error(format!("{builtin}: failed to restore GPU result: {error}"))
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                })?;
            runmat_accelerate_api::set_handle_logical(&output, true);
            (
                output,
                array.shape.clone(),
                GpuTensorStorage::Real,
                Some(provider.precision()),
                None,
                true,
            )
        }
        Value::ComplexTensor(tensor) => {
            let expected_integer = tensor
                .integer_storage()
                .map(|storage| integer_element_type(&storage.real));
            let expected_precision =
                expected_integer
                    .is_none()
                    .then(|| match tensor.numeric_dtype() {
                        NumericDType::F32 => ProviderPrecision::F32,
                        _ => ProviderPrecision::F64,
                    });
            let data = interleaved_complex_data(tensor)
                .map_err(|error| build_runtime_error(format!("{builtin}: {error}")).build())?;
            let transfer = HostNumericTensorView {
                data: data.as_view(),
                shape: &tensor.shape,
                storage: GpuTensorStorage::ComplexInterleaved,
            };
            let output = match provider.upload_numeric(&transfer) {
                Ok(output) => output,
                Err(_) if expected_integer.is_none() && !explicit => return Ok(value),
                Err(error) => {
                    return Err(build_runtime_error(format!(
                        "{builtin}: failed to restore GPU result: {error}"
                    ))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build())
                }
            };
            let output = validate_numeric_upload(provider, output, transfer, &protected).map_err(
                |error| {
                    build_runtime_error(format!("{builtin}: {error}"))
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                },
            )?;
            (
                output,
                tensor.shape.clone(),
                GpuTensorStorage::ComplexInterleaved,
                expected_precision,
                expected_integer,
                false,
            )
        }
        _ => return Ok(value),
    };

    let aliases_source = protected
        .iter()
        .any(|input| same_gpu_handle(&output, input));
    if aliases_source {
        return Err(build_runtime_error(format!(
            "{builtin}: provider aliased the protected input while restoring the result"
        ))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }

    let valid = output.shape == expected_shape
        && output.device_id == provider.device_id()
        && exact_provider_for_handle(&output).is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(&output) == expected_storage
        && expected_precision.is_none_or(|precision| {
            runmat_accelerate_api::handle_precision(&output) == Some(precision)
        })
        && runmat_accelerate_api::handle_integer_type(&output) == expected_integer
        && runmat_accelerate_api::handle_is_logical(&output) == logical
        && gpu_class_metadata_matches(&output, expected_precision, expected_integer, logical);
    if !valid {
        free_rejected_provider_output(&output, &protected, provider);
        return Err(build_runtime_error(format!(
            "{builtin}: provider returned an invalid restored result"
        ))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    let provenance = if explicit {
        runmat_accelerate_api::GpuHandleProvenance::Explicit
    } else {
        source_guard
            .snapshot
            .provenance
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic)
    };
    output.descriptor.provenance = Some(provenance);
    runmat_accelerate_api::mark_residency(&output);
    Ok(Value::GpuTensor(output))
}
