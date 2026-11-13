use super::super::inverse_helpers;
use super::*;

#[derive(Clone)]
pub(super) enum OutputTemplate {
    Default,
    Like(Value),
}

pub(super) fn parse_output_template(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<OutputTemplate> {
    match args.len() {
        0 => Ok(OutputTemplate::Default),
        1 => {
            if matches!(keyword_of(&args[0])?.as_deref(), Some("like")) {
                Err(tan_error_with_detail(
                    &TAN_ERROR_INVALID_OPTION,
                    "expected prototype after 'like'",
                ))
            } else {
                Err(tan_error_with_detail(
                    &TAN_ERROR_INVALID_OPTION,
                    "unrecognised argument for tan",
                ))
            }
        }
        2 => {
            if matches!(keyword_of(&args[0])?.as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err(tan_error_with_detail(
                    &TAN_ERROR_INVALID_OPTION,
                    "unsupported option; only 'like' is accepted",
                ))
            }
        }
        _ => Err(tan_error(&TAN_ERROR_ARG_COUNT)),
    }
}

pub(super) fn admit_output_template(
    template: OutputTemplate,
) -> BuiltinResult<inverse_helpers::like::Template> {
    use inverse_helpers::like::{Prototype, PrototypeError, Template};
    match template {
        OutputTemplate::Default => Ok(Template::Default),
        OutputTemplate::Like(value) => {
            Prototype::parse(&value)
                .map(Template::Like)
                .map_err(|error| {
                    let descriptor = if matches!(error, PrototypeError::MissingOwner) {
                        &TAN_ERROR_GPU_UNAVAILABLE
                    } else {
                        &TAN_ERROR_LIKE_PROTOTYPE
                    };
                    tan_error_with_detail(descriptor, error)
                })
        }
    }
}

pub(super) async fn apply_output_template(
    value: Value,
    template: &inverse_helpers::like::Template,
    protected: &[GpuTensorHandle],
) -> BuiltinResult<Value> {
    use inverse_helpers::like::{Prototype, Template};
    match template {
        Template::Default => Ok(value),
        Template::Like(Prototype::HostReal) => convert_to_host_like(value).await,
        Template::Like(Prototype::HostComplex) => convert_to_host_complex(value).await,
        Template::Like(Prototype::Resident {
            handle,
            complex: true,
        }) => convert_to_gpu_complex(value, handle, protected).await,
        Template::Like(Prototype::Resident {
            handle,
            complex: false,
        }) => convert_to_gpu(value, handle, protected).await,
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_gpu(
    value: Value,
    prototype: &GpuTensorHandle,
    protected: &[GpuTensorHandle],
) -> BuiltinResult<Value> {
    let borrowed: Vec<_> = protected.iter().chain(std::iter::once(prototype)).collect();
    let provider = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        tan_error_with_detail(
            &TAN_ERROR_GPU_UNAVAILABLE,
            "GPU output requested via 'like' but no provider owns the prototype",
        )
    })?;
    match value {
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(&handle)
                == runmat_accelerate_api::GpuTensorStorage::Real
                && gpu_helpers::exact_provider_for_handle(&handle).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                }) =>
        {
            gpu_helpers::resident_gpu_value(handle)
        }
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            convert_to_gpu(gathered, prototype, protected).await
        }
        Value::Tensor(tensor) => {
            let handle = gpu_helpers::upload_tensor_protecting(provider, &tensor, &borrowed)
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            gpu_helpers::resident_gpu_value(handle)
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor), prototype, protected).await
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64()), prototype, protected).await,
        Value::Bool(b) => {
            convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 }), prototype, protected).await
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor), prototype, protected).await
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(tan_error_with_detail(
            &TAN_ERROR_LIKE_PROTOTYPE,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        other => Err(tan_error_with_detail(
            &TAN_ERROR_INTERNAL,
            format!("unsupported result type for GPU output via 'like' ({other:?})"),
        )),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_gpu_complex(
    value: Value,
    prototype: &GpuTensorHandle,
    protected: &[GpuTensorHandle],
) -> BuiltinResult<Value> {
    let borrowed: Vec<_> = protected.iter().chain(std::iter::once(prototype)).collect();
    let provider = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        tan_error_with_detail(
            &TAN_ERROR_GPU_UNAVAILABLE,
            "complex GPU output requested via 'like' but no provider owns the prototype",
        )
    })?;
    match value {
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(&handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                && gpu_helpers::exact_provider_for_handle(&handle).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                }) =>
        {
            gpu_helpers::complex_gpu_value(handle)
        }
        Value::GpuTensor(handle) => {
            let same_owner = gpu_helpers::exact_provider_for_handle(&handle).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            });
            if same_owner
                && runmat_accelerate_api::handle_storage(&handle)
                    == runmat_accelerate_api::GpuTensorStorage::Real
            {
                match provider.complex_from_real(&handle).await {
                    Ok(output)
                        if !borrowed
                            .iter()
                            .any(|input| gpu_helpers::same_gpu_handle(input, &output))
                            && gpu_helpers::unary_gpu_output_matches(
                                &output,
                                &handle,
                                provider,
                                gpu_helpers::UnaryGpuOutputContract {
                                    storage:
                                        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
                                    precision: runmat_accelerate_api::handle_precision(&handle),
                                    integer: None,
                                    logical: false,
                                    alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
                                },
                            ) =>
                    {
                        return gpu_helpers::complex_gpu_value(output);
                    }
                    Ok(output) => {
                        let mut error = tan_error_with_detail(
                            &TAN_ERROR_INTERNAL,
                            "provider returned an invalid complex conversion result",
                        );
                        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                        return Err(gpu_helpers::reject_gpu_output(output, error));
                    }
                    Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {}
                    Err(error) => {
                        let mut failure = tan_error_with_detail(
                            &TAN_ERROR_INTERNAL,
                            format!("provider complex conversion failed: {error}"),
                        );
                        failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
                        failure = crate::runtime_error::with_provider_execution_source(
                            failure,
                            error.into(),
                        );
                        return Err(failure);
                    }
                }
            }
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            convert_to_gpu_complex(gathered, prototype, protected).await
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            let handle =
                gpu_helpers::upload_complex_tensor_protecting(provider, &tensor, &borrowed)
                    .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            gpu_helpers::complex_gpu_value(handle)
        }
        Value::ComplexTensor(tensor) => {
            let handle =
                gpu_helpers::upload_complex_tensor_protecting(provider, &tensor, &borrowed)
                    .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            gpu_helpers::complex_gpu_value(handle)
        }
        Value::Num(n) => convert_to_gpu_complex(Value::Complex(n, 0.0), prototype, protected).await,
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64_cow(&tensor);
            let data = values.iter().map(|&re| (re, 0.0)).collect::<Vec<_>>();
            let complex = ComplexTensor::from_f64_values_with_dtype(
                data,
                tensor.shape().to_vec(),
                complex_floating_output_dtype(tensor.numeric_dtype()),
            )
            .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            convert_to_gpu_complex(Value::ComplexTensor(complex), prototype, protected).await
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            convert_to_gpu_complex(Value::Tensor(tensor), prototype, protected).await
        }
        Value::Int(i) => convert_to_gpu_complex(Value::Num(i.to_f64()), prototype, protected).await,
        Value::Bool(b) => {
            convert_to_gpu_complex(Value::Num(if b { 1.0 } else { 0.0 }), prototype, protected)
                .await
        }
        other => Err(tan_error_with_detail(
            &TAN_ERROR_INTERNAL,
            format!("cannot convert value {other:?} to complex GPU output via 'like'"),
        )),
    }
}

async fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value_async(&proxy).await
        }
        other => Ok(other),
    }
}

#[async_recursion::async_recursion(?Send)]
pub(super) async fn convert_to_host_complex(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64_cow(&tensor);
            let data = values.iter().map(|&re| (re, 0.0)).collect::<Vec<_>>();
            let complex = ComplexTensor::from_f64_values_with_dtype(
                data,
                tensor.shape().to_vec(),
                complex_floating_output_dtype(tensor.numeric_dtype()),
            )
            .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            Ok(complex_tensor_into_value(complex))
        }
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            convert_to_host_complex(gathered).await
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            convert_to_host_complex(Value::Tensor(tensor)).await
        }
        Value::Int(i) => convert_to_host_complex(Value::Num(i.to_f64())).await,
        Value::Bool(b) => convert_to_host_complex(Value::Num(if b { 1.0 } else { 0.0 })).await,
        other => Err(tan_error_with_detail(
            &TAN_ERROR_INTERNAL,
            format!("cannot convert value {other:?} to complex output via 'like'"),
        )),
    }
}

fn complex_floating_output_dtype(dtype: NumericDType) -> NumericDType {
    if dtype == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}
