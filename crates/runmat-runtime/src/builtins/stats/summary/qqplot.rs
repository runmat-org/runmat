//! Quantile-quantile plot compatibility helper.

use runmat_builtins::{
    BuiltinErrorDescriptor, QqplotInputRole, QQPLOT_ERROR_INTERNAL, QQPLOT_ERROR_INVALID_ARGUMENT,
};
#[cfg(test)]
use runmat_builtins::{
    QQPLOT_EXPLICIT_GPU_EXTENSION, QQPLOT_INTEGER_PVEC_EXTENSION, QQPLOT_INTEGER_SAMPLE_EXTENSION,
    QQPLOT_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::plotting::op_common::{apply_axes_target, split_leading_axes_arguments};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "qqplot";
mod admission;
mod geometry;
mod plot;
mod series;
use plot::render_series;
#[cfg(test)]
use series::columns_from_tensor;
use series::{
    normal_series_from_tensor, probabilities_from_tensor, tensor_values_f64,
    two_sample_series_from_tensors,
};
#[cfg(test)]
mod provider_tests;

fn error(descriptor: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error(&QQPLOT_ERROR_INVALID_ARGUMENT, message)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error(&QQPLOT_ERROR_INTERNAL, message)
}

fn invalid_with_source(message: impl Into<String>, source: RuntimeError) -> RuntimeError {
    let span = source.span;
    let mut builder = build_runtime_error(message)
        .with_builtin(NAME)
        .with_gpu_gather_retry(source.gpu_gather_retry())
        .with_source(source);
    if let Some(span) = span {
        builder = builder.with_span(span);
    }
    if let Some(identifier) = QQPLOT_ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Clone, Debug)]
struct Series {
    theoretical: Vec<f64>,
    observed: Vec<f64>,
    reference_x: Vec<f64>,
    reference_y: Vec<f64>,
    quartile_x: Vec<f64>,
    quartile_y: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlotMode {
    Normal,
    TwoSample,
}

struct QqplotEvaluation {
    mode: PlotMode,
    series: Vec<Series>,
}

#[runtime_builtin(
    name = "qqplot",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::qqplot"
)]
pub(crate) async fn qqplot_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let (target, args) = split_leading_axes_arguments(args, NAME)
        .map_err(|err| invalid_with_source(format!("qqplot: {err}"), err))?;
    admission::admit_host_visible(args.view())?;
    let eval = parse_args(args).await?;
    if eval.series.is_empty() {
        return Err(invalid("qqplot: input contains no finite samples"));
    }
    apply_axes_target(target, NAME).map_err(|err| {
        let message = err.message.clone();
        invalid_with_source(message, err)
    })?;

    let handles = render_series(eval)?;
    if handles.is_empty() {
        return Ok(Value::Tensor(
            Tensor::new(Vec::new(), vec![0, 0])
                .map_err(|err| internal(format!("qqplot: {err}")))?,
        ));
    }
    Tensor::new(handles.clone(), vec![handles.len(), 1])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("qqplot: {err}")))
}

async fn parse_args(args: crate::call::BuiltinArguments) -> BuiltinResult<QqplotEvaluation> {
    let _argument_sources = args.fork();
    match args.len() {
        0 => Err(invalid("qqplot: sample input is required")),
        1 => {
            let x =
                value_to_tensor(args.into_iter().next().unwrap(), QqplotInputRole::SampleX).await?;
            Ok(QqplotEvaluation {
                mode: PlotMode::Normal,
                series: normal_series_from_tensor(x, None)?,
            })
        }
        2 => {
            let mut iter = args.into_iter();
            let x = value_to_tensor(iter.next().unwrap(), QqplotInputRole::SampleX).await?;
            let second = value_to_tensor(iter.next().unwrap(), QqplotInputRole::SampleY).await?;
            Ok(QqplotEvaluation {
                mode: PlotMode::TwoSample,
                series: two_sample_series_from_tensors(x, second, None)?,
            })
        }
        3 => {
            let mut iter = args.into_iter();
            let x = value_to_tensor(iter.next().unwrap(), QqplotInputRole::SampleX).await?;
            let y = value_to_tensor(iter.next().unwrap(), QqplotInputRole::SampleY).await?;
            let p = probabilities_from_tensor(
                value_to_tensor(iter.next().unwrap(), QqplotInputRole::Percentages).await?,
            )?;
            if tensor::tensor_element_len(&y) == 0 {
                Ok(QqplotEvaluation {
                    mode: PlotMode::Normal,
                    series: normal_series_from_tensor(x, Some(p))?,
                })
            } else {
                Ok(QqplotEvaluation {
                    mode: PlotMode::TwoSample,
                    series: two_sample_series_from_tensors(x, y, Some(p))?,
                })
            }
        }
        _ => Err(invalid("qqplot: accepts at most x, y, and pvec inputs")),
    }
}

async fn value_to_tensor(value: Value, role: QqplotInputRole) -> BuiltinResult<Tensor> {
    use crate::builtins::common::validation::{
        native_integer_value_is_exact_f64, value_has_native_integer_class,
    };
    let resident_integer =
        matches!(value, Value::GpuTensor(_)) && value_has_native_integer_class(&value);
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid_with_source(format!("qqplot: {err}"), err))?;
    if resident_integer && !native_integer_value_is_exact_f64(&gathered) {
        return Err(invalid(format!(
            "qqplot: integer {} values must be exactly representable as double",
            role.integer_label()
        )));
    }
    let tensor = tensor::value_into_tensor_for(NAME, gathered)
        .map_err(|err| invalid(format!("qqplot: {err}")))?;
    if tensor.integer_storage().is_some() {
        return Tensor::new(tensor_values_f64(&tensor), tensor.shape().to_vec())
            .map_err(|err| internal(format!("qqplot: {err}")));
    }
    Ok(tensor)
}
#[cfg(test)]
mod tests;
