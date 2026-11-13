use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, ProviderPrecision};
use runmat_builtins::{
    BuiltinErrorDescriptor, EXPRND_ERROR_INTERNAL, EXPRND_ERROR_INVALID_ARGUMENT,
    EXPRND_ERROR_MU_MUST_BE_POSITIVE, EXPRND_INTEGER_MU_EXTENSION, EXPRND_INTEGER_SIZE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericDType, Tensor, Value};

use crate::build_runtime_error;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random;
use crate::builtins::common::tensor;
use crate::{gather_if_needed_async, BuiltinResult};
#[path = "exprnd/output.rs"]
mod output;

const BUILTIN_NAME: &str = "exprnd";

fn exprnd_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn exprnd_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    exprnd_error_with(error, error.message)
}

fn exprnd_internal_error(message: impl Into<String>) -> crate::RuntimeError {
    exprnd_error_with(&EXPRND_ERROR_INTERNAL, message)
}

#[runtime_builtin(
    name = "exprnd",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::random::exprnd"
)]
async fn exprnd_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    ensure_output_arity()?;
    let integer_mu = args.first().is_some_and(is_typed_integer_value);
    let integer_size = args.iter().skip(1).any(is_typed_integer_value);
    if integer_mu {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXPRND_INTEGER_MU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if integer_size {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXPRND_INTEGER_SIZE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    reject_unsupported_classes(args.view())?;
    let parsed = parse_args(args).await?;
    if parsed.mu.iter().any(|mu| !mu.is_finite() || *mu <= 0.0) {
        return Err(exprnd_error(&EXPRND_ERROR_MU_MUST_BE_POSITIVE));
    }
    let len = checked_element_count(&parsed.shape)?;
    validate_explicit_restoration(&parsed)?;
    let mut turn = random::acquire_turn(None, BUILTIN_NAME).await?;
    let host = output::generate(&mut turn, &parsed, len).await?;
    build_output(host, parsed)
}

fn checked_element_count(shape: &[usize]) -> BuiltinResult<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |n, dim| n.checked_mul(*dim))
        .filter(|n| *n <= isize::MAX as usize / std::mem::size_of::<f64>())
        .ok_or_else(|| {
            exprnd_error_with(
                &EXPRND_ERROR_INVALID_ARGUMENT,
                "exprnd: output size exceeds the supported allocation range",
            )
        })
}

fn validate_explicit_restoration(parsed: &ParsedExprnd) -> BuiltinResult<()> {
    let Some(source) = &parsed.gpu_source else {
        return Ok(());
    };
    if !runmat_accelerate_api::handle_is_explicit(source) {
        return Ok(());
    }
    let precision = match parsed.output_precision {
        OutputPrecision::Double => ProviderPrecision::F64,
        OutputPrecision::Single => ProviderPrecision::F32,
    };
    if runmat_accelerate_api::provider_for_handle(source).is_some_and(|owner| {
        owner.device_id() == source.device_id() && owner.precision() == precision
    }) {
        return Ok(());
    }
    Err(exprnd_internal_error(
        "exprnd: provider cannot preserve explicit gpuArray output residency",
    ))
}

fn ensure_output_arity() -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: too many output arguments",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputPrecision {
    Double,
    Single,
}

struct ParsedExprnd {
    mu: Vec<f64>,
    shape: Vec<usize>,
    output_precision: OutputPrecision,
    gpu_source: Option<GpuTensorHandle>,
    borrowed_gpu: Vec<GpuTensorHandle>,
}

fn output_precision(value: &Value) -> OutputPrecision {
    match value {
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            OutputPrecision::Single
        }
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(ProviderPrecision::F32) =>
        {
            OutputPrecision::Single
        }
        _ => OutputPrecision::Double,
    }
}

fn build_output(output: Tensor, parsed: ParsedExprnd) -> BuiltinResult<Value> {
    if let Some(source) = parsed.gpu_source {
        let Some(provider) = runmat_accelerate_api::provider_for_handle(&source)
            .filter(|owner| owner.device_id() == source.device_id())
        else {
            return Ok(exprnd_host_output(output, parsed.output_precision));
        };
        let required_precision = match parsed.output_precision {
            OutputPrecision::Double => ProviderPrecision::F64,
            OutputPrecision::Single => ProviderPrecision::F32,
        };
        if provider.precision() != required_precision {
            return Ok(exprnd_host_output(output, parsed.output_precision));
        }
        let protected: Vec<_> = parsed.borrowed_gpu.iter().collect();
        let handle = gpu_helpers::upload_tensor_protecting_typed(provider, &output, &protected)
            .map_err(|err| {
                build_runtime_error(format!("exprnd: {err}"))
                    .with_builtin(BUILTIN_NAME)
                    .with_identifier(
                        EXPRND_ERROR_INTERNAL
                            .identifier
                            .expect("internal identifier"),
                    )
                    .with_source(err)
                    .build()
            })?;
        if !valid_exprnd_gpu_output(
            &handle,
            &source,
            provider,
            output.shape(),
            required_precision,
        ) {
            return Err(gpu_helpers::reject_gpu_output(
                handle,
                exprnd_internal_error("exprnd: provider upload returned malformed resident output"),
            ));
        }
        let handle = gpu_helpers::propagate_owned_output_provenance(handle, [&source])?;
        return gpu_helpers::resident_gpu_value(handle);
    }
    Ok(exprnd_host_output(output, parsed.output_precision))
}

fn exprnd_host_output(output: Tensor, precision: OutputPrecision) -> Value {
    if precision == OutputPrecision::Double {
        tensor::tensor_into_value(output)
    } else {
        Value::Tensor(output)
    }
}

mod parsing;
mod provider;
use parsing::*;
use provider::*;
#[cfg(test)]
mod tests;
