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
            if let Some(keyword) = keyword_of(&args[0])? {
                if keyword.trim() == "like" {
                    return Err(atan_error_with_detail(
                        &ATAN_ERROR_INVALID_OPTION,
                        "expected prototype after 'like'",
                    ));
                }
            }
            Err(atan_error_with_detail(
                &ATAN_ERROR_INVALID_OPTION,
                "unrecognised argument for atan",
            ))
        }
        len if len >= 2 => {
            if let Some(keyword) = keyword_of(&args[0])? {
                if keyword.trim() == "like" {
                    if len == 2 {
                        return Ok(OutputTemplate::Like(args[1].clone()));
                    }
                    return Err(atan_error(&ATAN_ERROR_ARG_COUNT));
                }
            }
            Err(atan_error_with_detail(
                &ATAN_ERROR_INVALID_OPTION,
                "unsupported option; only 'like' is accepted",
            ))
        }
        _ => unreachable!(),
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
                        &ATAN_ERROR_GPU_UNAVAILABLE
                    } else {
                        &ATAN_ERROR_LIKE_PROTOTYPE
                    };
                    atan_error_with_detail(descriptor, error)
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
        Template::Like(Prototype::HostReal) => ensure_host_real(value).await,
        Template::Like(Prototype::HostComplex) => ensure_host_complex(value).await,
        Template::Like(Prototype::Resident { handle, complex }) => {
            ensure_provider_output(value, handle, *complex, protected).await
        }
    }
}

async fn ensure_host_value(value: Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(_) = &value {
        gpu_helpers::gather_value_async(&value).await
    } else {
        Ok(value)
    }
}

async fn ensure_host_real(value: Value) -> BuiltinResult<Value> {
    let host_value = ensure_host_value(value).await?;
    if is_complex_value(&host_value) {
        return Err(atan_error_with_detail(
            &ATAN_ERROR_LIKE_PROTOTYPE,
            "result is complex but 'like' prototype is real",
        ));
    }
    Ok(host_value)
}

async fn ensure_host_complex(value: Value) -> BuiltinResult<Value> {
    let host_value = ensure_host_value(value).await?;
    if is_complex_value(&host_value) {
        Ok(host_value)
    } else {
        convert_real_to_complex(host_value)
    }
}

async fn ensure_provider_output(
    value: Value,
    prototype: &GpuTensorHandle,
    complex: bool,
    protected: &[GpuTensorHandle],
) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        atan_error_with_detail(
            &ATAN_ERROR_GPU_UNAVAILABLE,
            "provider output requested via 'like' but the prototype has no owner",
        )
    })?;
    let expected_storage = if complex {
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    } else {
        runmat_accelerate_api::GpuTensorStorage::Real
    };
    if let Value::GpuTensor(handle) = &value {
        if runmat_accelerate_api::handle_storage(handle) == expected_storage
            && gpu_helpers::exact_provider_for_handle(handle)
                .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        {
            return Ok(value);
        }
    }
    let host = if complex {
        ensure_host_complex(value).await?
    } else {
        ensure_host_real(value).await?
    };
    crate::builtins::common::provider_restore::upload_value_like_protected(
        provider,
        host,
        BUILTIN_NAME,
        prototype,
        protected,
    )
}

fn is_complex_value(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

pub(super) fn convert_real_to_complex(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64_cow(&tensor);
            let data: Vec<(f64, f64)> = values.iter().map(|&v| (v, 0.0)).collect();
            let tensor = ComplexTensor::from_f64_values_with_dtype(
                data,
                tensor.shape().to_vec(),
                tensor.numeric_dtype(),
            )
            .map_err(|e| atan_error_with_detail(&ATAN_ERROR_INTERNAL, e))?;
            Ok(complex_tensor_into_value(tensor))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| atan_error_with_detail(&ATAN_ERROR_INTERNAL, e))?;
            convert_real_to_complex(Value::Tensor(tensor))
        }
        Value::Int(i) => convert_real_to_complex(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_real_to_complex(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            Err(atan_error_with_detail(
                &ATAN_ERROR_LIKE_PROTOTYPE,
                "'like' prototype must be numeric",
            ))
        }
        Value::GpuTensor(_) => Err(atan_error_with_detail(
            &ATAN_ERROR_INTERNAL,
            "internal error converting GPU value to complex output",
        )),
        other => Err(atan_error_with_detail(
            &ATAN_ERROR_INTERNAL,
            format!("cannot convert value {other:?} into a complex result for 'like'"),
        )),
    }
}
