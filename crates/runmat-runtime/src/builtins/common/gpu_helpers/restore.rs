use super::*;

#[cfg(test)]
#[path = "restore_owned_tests.rs"]
mod owned_tests;

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
    let _source_guard = ResidencyRestoreGuard::new(source);
    let protected = std::iter::once(source)
        .chain(protected.iter().copied())
        .collect::<Vec<_>>();
    let _operand_guards = protected
        .iter()
        .map(|handle| ResidencyRestoreGuard::new(handle))
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
            let output = match runmat_accelerate_api::upload_numeric_protecting(
                provider, &transfer, &protected,
            ) {
                Ok(output) => output,
                Err(error)
                    if expected_integer.is_none()
                        && !explicit
                        && matches!(&error, runmat_accelerate_api::NumericUploadError::Provider(error) if provider_hook_is_unsupported(error)) =>
                {
                    return Ok(value)
                }
                Err(error) => {
                    return Err(build_runtime_error(format!(
                        "{builtin}: failed to restore GPU result: {error}"
                    ))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .with_source(error)
                    .build())
                }
            };
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
            let output = runmat_accelerate_api::upload_logical_protecting(
                provider,
                &runmat_accelerate_api::HostLogicalTensorView {
                    data: &array.data,
                    shape: &array.shape,
                },
                &protected,
            )
            .map_err(|error| {
                build_runtime_error(format!(
                    "{builtin}: failed to restore logical GPU result: {error}"
                ))
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .with_source(error)
                .build()
            })?;
            let precision = runmat_accelerate_api::handle_precision(&output);
            let integer = runmat_accelerate_api::handle_integer_type(&output);
            (
                output,
                array.shape.clone(),
                GpuTensorStorage::Real,
                precision,
                integer,
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
            let output = match runmat_accelerate_api::upload_numeric_protecting(
                provider, &transfer, &protected,
            ) {
                Ok(output) => output,
                Err(error)
                    if expected_integer.is_none()
                        && !explicit
                        && matches!(&error, runmat_accelerate_api::NumericUploadError::Provider(error) if provider_hook_is_unsupported(error)) =>
                {
                    return Ok(value)
                }
                Err(error) => {
                    return Err(build_runtime_error(format!(
                        "{builtin}: failed to restore GPU result: {error}"
                    ))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .with_source(error)
                    .build())
                }
            };
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
        .any(|input| output.shares_allocation_with(input));
    if aliases_source {
        return Err(reject_gpu_output(
            output,
            build_runtime_error(format!(
                "{builtin}: provider aliased the protected input while restoring the result"
            ))
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ));
    }

    let valid = output.shape() == expected_shape
        && output.device_id() == provider.device_id()
        && exact_provider_for_handle(&output).is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(&output) == expected_storage
        && expected_precision.is_none_or(|precision| {
            runmat_accelerate_api::handle_precision(&output) == Some(precision)
        })
        && runmat_accelerate_api::handle_integer_type(&output) == expected_integer
        && runmat_accelerate_api::handle_is_logical(&output) == logical
        && gpu_class_metadata_matches(&output, expected_precision, expected_integer, logical);
    if !valid {
        return Err(reject_gpu_output(
            output,
            build_runtime_error(format!(
                "{builtin}: provider returned an invalid restored result"
            ))
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build(),
        ));
    }
    let provenance = if explicit {
        runmat_accelerate_api::GpuHandleProvenance::Explicit
    } else {
        runmat_accelerate_api::handle_provenance(source)
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic)
    };
    output = output.with_provenance(provenance).map_err(|error| {
        build_runtime_error(format!(
            "{builtin}: cannot preserve restored result intent: {error}"
        ))
        .with_source(error)
        .build()
    })?;
    resident_gpu_value(output)
}
