use super::*;

#[cfg(test)]
#[path = "restore_owned_tests.rs"]
mod owned_tests;

/// Restore a host-computed real scalar after an input has already crossed a
/// provider boundary. A fresh result is required even for automatic inputs:
/// these callers have no host-result fallback once an owning provider exists.
pub(crate) fn restore_real_scalar_after_host(
    source: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    value: f64,
    precision: ProviderPrecision,
    builtin: &str,
    internal_error: impl Fn(String) -> crate::RuntimeError,
) -> crate::BuiltinResult<Value> {
    let terminal = |detail: String| {
        let mut failure = internal_error(format!("{builtin}: {detail}"));
        failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
        failure
    };
    source.allocation_lease().validate_live().map_err(|cause| {
        crate::runtime_error::with_provider_execution_source(
            terminal(format!("input allocation is not live: {cause}")),
            runmat_accelerate_api::ProviderExecutionError::from_source(cause)
                .with_unproven_effects(),
        )
    })?;
    if source.device_id() != provider.device_id()
        || !exact_provider_for_handle(source)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
    {
        return Err(terminal("input has no exact owning provider".to_owned()));
    }

    let single = [value as f32];
    let double = [value];
    let data = match precision {
        ProviderPrecision::F32 => HostNumericDataView::F32(&single),
        ProviderPrecision::F64 => HostNumericDataView::F64(&double),
    };
    let mut output = runmat_accelerate_api::upload_numeric_protecting(
        provider,
        &HostNumericTensorView {
            data,
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        },
        &[source],
    )
    .map_err(|cause| {
        let failure = terminal(format!("scalar upload failed: {cause}"));
        crate::runtime_error::with_provider_execution_source(
            failure,
            cause.into_execution_error().with_unproven_effects(),
        )
    })?;

    if !unary_gpu_output_matches_shape(
        &output,
        source,
        provider,
        &[1, 1],
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: Some(precision),
            integer: None,
            logical: false,
            alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    ) {
        return Err(reject_gpu_output(
            output,
            terminal("provider returned a malformed, aliased, or foreign scalar".to_owned()),
        ));
    }
    if let Err(cause) = source.allocation_lease().validate_live() {
        let failure = crate::runtime_error::with_provider_execution_source(
            terminal(format!(
                "input allocation retired during scalar upload: {cause}"
            )),
            runmat_accelerate_api::ProviderExecutionError::from_source(cause)
                .with_unproven_effects(),
        );
        return Err(reject_gpu_output(output, failure));
    }
    if let Err(cause) = propagate_output_provenance(&mut output, [source]) {
        return Err(reject_gpu_output(output, cause));
    }
    resident_gpu_value(output).map_err(|mut cause| {
        cause.gpu_gather_retry = crate::GpuGatherRetry::Never;
        cause
    })
}

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
    restore_class_preserving_value_with_operands(
        source,
        value,
        builtin,
        RestoreOperandContext::new(protected, protected),
    )
}

/// Separate allocation safety from output intent. `protected` includes every
/// borrowed allocation that the provider must not return as a new result;
/// `intent_sources` contains only operands whose explicit device placement
/// participates in the result contract. The source participates in both.
#[derive(Clone, Copy)]
pub struct RestoreOperandContext<'a> {
    protected: &'a [&'a GpuTensorHandle],
    intent_sources: &'a [&'a GpuTensorHandle],
}

impl<'a> RestoreOperandContext<'a> {
    pub fn new(
        protected: &'a [&'a GpuTensorHandle],
        intent_sources: &'a [&'a GpuTensorHandle],
    ) -> Self {
        Self {
            protected,
            intent_sources,
        }
    }
}

pub fn restore_class_preserving_value_with_operands(
    source: &GpuTensorHandle,
    value: Value,
    builtin: &str,
    operands: RestoreOperandContext<'_>,
) -> crate::BuiltinResult<Value> {
    let value = match value {
        Value::Num(number) => {
            let tensor = Tensor::new(vec![number], vec![1, 1]).map_err(|error| {
                build_runtime_error(format!("{builtin}: invalid scalar result: {error}"))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
            })?;
            return match restore_class_preserving_value_with_operands(
                source,
                Value::Tensor(tensor),
                builtin,
                operands,
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
            return match restore_class_preserving_value_with_operands(
                source,
                Value::ComplexTensor(tensor),
                builtin,
                operands,
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
            return match restore_class_preserving_value_with_operands(
                source,
                Value::LogicalArray(logical),
                builtin,
                operands,
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
        .chain(operands.protected.iter().copied())
        .chain(operands.intent_sources.iter().copied())
        .collect::<Vec<_>>();
    let _operand_guards = protected
        .iter()
        .map(|handle| ResidencyRestoreGuard::new(handle))
        .collect::<Vec<_>>();
    let explicit = std::iter::once(source)
        .chain(operands.intent_sources.iter().copied())
        .any(|handle| runmat_accelerate_api::handle_is_explicit(handle));
    let automatic_host_fallback = std::iter::once(source)
        .chain(operands.intent_sources.iter().copied())
        .all(|handle| {
            runmat_accelerate_api::handle_provenance(handle)
                == Some(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        });

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
                shape: tensor.shape(),
                storage: GpuTensorStorage::Real,
            };
            let output = match runmat_accelerate_api::upload_numeric_protecting(
                provider, &transfer, &protected,
            ) {
                Ok(output) => output,
                Err(error)
                    if expected_integer.is_none()
                        && automatic_host_fallback
                        && matches!(&error, runmat_accelerate_api::NumericUploadError::Provider(error) if error.is_unsupported()) =>
                {
                    return Ok(value);
                }
                Err(error) => {
                    return Err(build_runtime_error(format!(
                        "{builtin}: failed to restore GPU result: {error}"
                    ))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .with_source(error)
                    .build());
                }
            };
            (
                output,
                tensor.shape(),
                GpuTensorStorage::Real,
                expected_precision,
                expected_integer,
                false,
            )
        }
        Value::LogicalArray(array) => {
            let output = match runmat_accelerate_api::upload_logical_protecting(
                provider,
                &runmat_accelerate_api::HostLogicalTensorView {
                    data: &array.data,
                    shape: &array.shape,
                },
                &protected,
            ) {
                Ok(output) => output,
                Err(runmat_accelerate_api::NumericUploadError::Provider(error))
                    if automatic_host_fallback && error.is_unsupported() =>
                {
                    return Ok(value);
                }
                Err(error) => {
                    return Err(build_runtime_error(format!(
                        "{builtin}: failed to restore logical GPU result: {error}"
                    ))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .with_source(error)
                    .build());
                }
            };
            let precision = runmat_accelerate_api::handle_precision(&output);
            let integer = runmat_accelerate_api::handle_integer_type(&output);
            (
                output,
                array.shape.as_slice(),
                GpuTensorStorage::Real,
                precision,
                integer,
                true,
            )
        }
        Value::ComplexTensor(tensor) => {
            let expected_integer = tensor
                .integer_storage()
                .map(|storage| integer_element_type(storage.real()));
            let expected_precision =
                expected_integer
                    .is_none()
                    .then(|| match tensor.numeric_dtype() {
                        NumericDType::F32 => ProviderPrecision::F32,
                        _ => ProviderPrecision::F64,
                    });
            let data = interleaved_complex_data(tensor).map_err(|error| {
                build_runtime_error(format!("{builtin}: {error}"))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
            })?;
            let transfer = HostNumericTensorView {
                data: data.as_view(),
                shape: tensor.shape(),
                storage: GpuTensorStorage::ComplexInterleaved,
            };
            let output = match runmat_accelerate_api::upload_numeric_protecting(
                provider, &transfer, &protected,
            ) {
                Ok(output) => output,
                Err(error)
                    if expected_integer.is_none()
                        && automatic_host_fallback
                        && matches!(&error, runmat_accelerate_api::NumericUploadError::Provider(error) if error.is_unsupported()) =>
                {
                    return Ok(value);
                }
                Err(error) => {
                    return Err(build_runtime_error(format!(
                        "{builtin}: failed to restore GPU result: {error}"
                    ))
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .with_source(error)
                    .build());
                }
            };
            (
                output,
                tensor.shape(),
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
        && exact_provider_for_handle(&output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
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
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
    })?;
    resident_gpu_value(output).map_err(|mut error| {
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}
