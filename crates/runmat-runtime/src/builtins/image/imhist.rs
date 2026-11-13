//! MATLAB-compatible `imhist` grayscale and indexed-image histograms.

#[cfg(feature = "plot-core")]
use runmat_builtins::IMHIST_ERROR_PLOT_FAILED;
use runmat_builtins::{
    BuiltinErrorDescriptor, IMHIST_ERROR_INTERNAL, IMHIST_ERROR_INVALID_ARGUMENT,
    IMHIST_ERROR_TOO_MANY_OUTPUTS, IMHIST_ERROR_UNSUPPORTED_IMAGE,
    IMHIST_INTEGER_TOLERANCE as INTEGER_TOL, IMHIST_MAX_BINS as MAX_BINS,
    IMHIST_TYPED_BIN_COUNT_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{
    BuiltinIntegerInputAvailability, BuiltinIntegerOutputClassRule, IMHIST_INTEGER_CAPABILITIES,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, LogicalArray, NumericStorage, Tensor, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[cfg(feature = "plot-core")]
use runmat_plot::plots::BarChart;

const NAME: &str = "imhist";
const DEFAULT_GRAYSCALE_BINS: usize = 256;
const LOGICAL_BINS: usize = 2;
#[cfg(feature = "plot-core")]
const MAX_PLOT_BINS: usize = 4096;

fn imhist_error_with_message(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn imhist_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let raw = detail.as_ref().trim();
    let normalized = raw.strip_prefix("imhist:").map(str::trim).unwrap_or(raw);
    let message = if normalized.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {}", error.message, normalized)
    };
    imhist_error_with_message(error, message)
}

fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    imhist_error_with_detail(&IMHIST_ERROR_INVALID_ARGUMENT, detail)
}

fn unsupported(detail: impl AsRef<str>) -> RuntimeError {
    imhist_error_with_detail(&IMHIST_ERROR_UNSUPPORTED_IMAGE, detail)
}

fn internal(detail: impl AsRef<str>) -> RuntimeError {
    imhist_error_with_detail(&IMHIST_ERROR_INTERNAL, detail)
}

fn too_many_outputs() -> RuntimeError {
    imhist_error_with_message(
        &IMHIST_ERROR_TOO_MANY_OUTPUTS,
        IMHIST_ERROR_TOO_MANY_OUTPUTS.message,
    )
}

#[cfg(feature = "plot-core")]
fn plot_failed(detail: impl AsRef<str>) -> RuntimeError {
    imhist_error_with_detail(&IMHIST_ERROR_PLOT_FAILED, detail)
}

fn map_flow(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        let message = format!(
            "{}: {}",
            IMHIST_ERROR_INVALID_ARGUMENT.message,
            err.message()
        );
        build_runtime_error(message)
            .with_builtin(NAME)
            .with_identifier(
                IMHIST_ERROR_INVALID_ARGUMENT
                    .identifier
                    .expect("imhist invalid-argument descriptor identifier"),
            )
            .with_source(err)
            .build()
    }
}

#[runtime_builtin(
    name = "imhist",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::image::imhist"
)]
async fn imhist_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    image: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 2) {
        return Err(too_many_outputs());
    }
    let output_count = crate::output_count::current_output_count().unwrap_or_else(|| {
        if crate::output_context::requested_output_count() == Some(0) {
            0
        } else {
            1
        }
    });
    let mut selected = placement::select(runtime.as_ref(), &image, rest.view(), output_count)?;
    let eval = evaluate_arguments(image, rest.view()).await?;

    if crate::output_context::requested_output_count() == Some(0)
        && crate::output_count::current_output_count().is_none()
    {
        if eval.has_resident_source() {
            return Ok(eval.into_host_counts_value());
        }
        eval.render_plot()?;
        return Ok(Value::output_list(Vec::new()));
    }

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            if eval.has_resident_source() {
                return Ok(eval.into_host_counts_value());
            }
            eval.render_plot()?;
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            let output = placement::publish(&mut selected, || eval.into_counts_value())?;
            return Ok(Value::output_list(vec![output]));
        }
        if out_count == 2 {
            let outputs = placement::publish_many(&mut selected, || eval.into_outputs())?;
            return Ok(Value::output_list(outputs));
        }
        return Err(too_many_outputs());
    }

    placement::publish(&mut selected, || eval.into_counts_value())
}

pub async fn evaluate(image: Value, rest: &[Value]) -> BuiltinResult<ImhistEvaluation> {
    evaluate_arguments(image, crate::call::FunctionArgumentView::from_slice(rest)).await
}

async fn evaluate_arguments(
    image: Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ImhistEvaluation> {
    if rest.len() > 1 {
        return Err(invalid(
            "expected imhist(I), imhist(I, n), or imhist(X, map)",
        ));
    }
    ensure_supported_resident_image_class(&image)?;
    if rest.first().is_some_and(is_typed_integer_scalar) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IMHIST_TYPED_BIN_COUNT_EXTENSION,
            NAME,
        )?;
    }
    let resident_source = match &image {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let indexed_call = rest.first().is_some_and(is_colormap_argument);
    let indexed_argument_is_resident = rest
        .first()
        .is_some_and(|value| matches!(value, Value::GpuTensor(_)));
    if indexed_call && (resident_source.is_some() || indexed_argument_is_resident) {
        return Err(unsupported("indexed gpuArray images are not supported"));
    }
    let image = gather_preserving_resident_value(&image).await?;
    let gathered_control = match rest.first() {
        Some(value) => Some(gather_preserving_resident_value(value).await?),
        None => None,
    };
    let call = parse_call(image, gathered_control.as_slice())?;
    let eval = match call.mode {
        ImhistMode::Grayscale { bins } => {
            let input = GrayscaleInput::from_value(call.image)?;
            input.evaluate(bins)?
        }
        ImhistMode::Indexed { bins } => {
            let input = IndexedInput::from_value(call.image, bins)?;
            input.evaluate()?
        }
    };
    Ok(eval.with_resident_source(resident_source))
}

async fn gather_preserving_resident_value(value: &Value) -> BuiltinResult<Value> {
    let Value::GpuTensor(handle) = value else {
        return Ok(value.clone());
    };
    let _residency = runmat_accelerate_api::ResidencyRestoreGuard::new(handle);
    let result = crate::builtins::common::gpu_helpers::gather_value_async(value).await;
    result.map_err(map_flow)
}

fn ensure_supported_resident_image_class(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Int(IntValue::I64(_) | IntValue::U64(_)) => Err(unsupported(
            "int64 and uint64 image classes are not supported",
        )),
        Value::Tensor(tensor)
            if matches!(
                tensor.numeric_dtype(),
                runmat_value::NumericDType::I64 | runmat_value::NumericDType::U64
            ) =>
        {
            Err(unsupported(
                "int64 and uint64 image classes are not supported",
            ))
        }
        Value::GpuTensor(handle)
            if matches!(
                runmat_accelerate_api::handle_integer_type(handle),
                Some(
                    runmat_accelerate_api::IntegerElementType::I64
                        | runmat_accelerate_api::IntegerElementType::U64
                )
            ) =>
        {
            Err(unsupported(
                "int64 and uint64 image classes are not supported",
            ))
        }
        _ => Ok(()),
    }
}

fn is_colormap_argument(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.shape().len() == 2 && tensor.shape().get(1) == Some(&3),
        Value::GpuTensor(handle) => handle.shape().len() == 2 && handle.shape().get(1) == Some(&3),
        _ => false,
    }
}

mod binning;
mod call;
mod inputs;
mod placement;
mod presentation;
#[cfg(test)]
#[path = "imhist/tests.rs"]
mod tests;

use binning::{count_indexed_values, histogram_counts, linspace};
use call::{is_typed_integer_scalar, parse_call, validate_bin_count, ImhistMode};
#[cfg(test)]
use call::{parse_optional_bin_count, scalar_number};
use inputs::{GrayscaleInput, IndexedInput};
#[cfg(all(test, feature = "plot-core"))]
use presentation::plot_display_bins;
#[cfg(test)]
use presentation::same_gpu_handle;
pub use presentation::ImhistEvaluation;
