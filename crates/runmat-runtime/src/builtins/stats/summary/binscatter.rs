//! Binned scatter plot compatibility helper.

mod input;
mod numeric;
mod options;
mod plot;

use input::parse_inputs;
pub(crate) use numeric::validate_face_alpha;
pub(crate) use numeric::{auto_bin_count, build_binscatter_chart};
use options::{
    integer_i128, integer_scalar, tensor_values_f64, validate_limits, BinscatterOptionPlan,
};
pub(crate) use options::{option_bool, parse_limits, parse_num_bins};
use plot::render_binscatter;

use runmat_builtins::{
    BinscatterOptionKind, BuiltinErrorDescriptor, BINSCATTER_ERROR_INTERNAL,
    BINSCATTER_ERROR_INVALID_ARGUMENT, BINSCATTER_GPU_INPUT_EXTENSION, BINSCATTER_MAX_NUM_BINS,
};
use runmat_macros::runtime_builtin;
use runmat_plot::plots::SurfacePlot;
use runmat_value::{IntValue, NumericScalar, ObjectInstance, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::plotting::op_common::{apply_axes_target, split_leading_axes_arguments};
use crate::builtins::plotting::state::color_limits_snapshot;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "binscatter";
const DEFAULT_FACE_ALPHA: f64 = 1.0;
pub(crate) const MAX_NUM_BINS: usize = BINSCATTER_MAX_NUM_BINS;
const AUTO_NUM_BINS_CAP: usize = 100;

#[derive(Clone, Debug)]
struct ParsedInputs {
    x_data: Tensor,
    y_data: Tensor,
}

#[derive(Clone, Debug)]
struct BinscatterOptions {
    show_empty_bins: bool,
    face_alpha: f64,
    display_name: Option<String>,
    num_bins: Option<[usize; 2]>,
    x_limits: Option<Tensor>,
    y_limits: Option<Tensor>,
}

pub(crate) struct BinscatterChart {
    pub(crate) values: Tensor,
    pub(crate) x_bin_edges: Vec<f64>,
    pub(crate) y_bin_edges: Vec<f64>,
    pub(crate) surface: SurfacePlot,
}

fn error(descriptor: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error(&BINSCATTER_ERROR_INVALID_ARGUMENT, message)
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
    if let Some(identifier) = BINSCATTER_ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error(&BINSCATTER_ERROR_INTERNAL, message)
}

fn map_plot_error(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        crate::builtins::plotting::state::with_render_cause(invalid(err.message.clone()), err)
    }
}

#[cfg(test)]
pub(crate) async fn binscatter_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
    let args =
        crate::execution::RootedValues::new(args).map_err(|error| error.into_runtime_error())?;
    binscatter_builtin_registered(None, args.into()).await
}

#[runtime_builtin(
    name = "binscatter",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::summary::binscatter"
)]
pub(crate) async fn binscatter_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<f64> {
    if args
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BINSCATTER_GPU_INPUT_EXTENSION,
            NAME,
        )?;
    }
    let (axes_target, args) = split_leading_axes_arguments(args, NAME).map_err(map_plot_error)?;
    let input_count = if matches!(args.first(), Some(Value::Object(object)) if crate::builtins::table::is_tabular_object(object))
    {
        3
    } else {
        2
    };
    if args.len() < input_count {
        return Err(invalid("binscatter: expected x and y inputs"));
    }
    let prepared_inputs = input::admit_host_visible(args.view().range(..input_count)?)?;
    let _argument_sources = args.fork();
    let option_plan = BinscatterOptionPlan::admit(args.range(input_count..)?)?;
    let inputs = if let Some(inputs) = prepared_inputs {
        inputs
    } else {
        parse_inputs(
            gather_values(args.into_range(..input_count)?, runtime.as_ref())
                .await?
                .view(),
        )?
    };
    let options = option_plan.evaluate().await?;
    let [x_bins, y_bins] = if let Some(num_bins) = options.num_bins {
        num_bins
    } else {
        [
            auto_bin_count(&inputs.x_data, options.x_limits.as_ref(), AUTO_NUM_BINS_CAP)?,
            auto_bin_count(&inputs.y_data, options.y_limits.as_ref(), AUTO_NUM_BINS_CAP)?,
        ]
    };
    let mut chart = build_binscatter_chart(
        &inputs.x_data,
        &inputs.y_data,
        [x_bins, y_bins],
        options.x_limits.as_ref(),
        options.y_limits.as_ref(),
        options.show_empty_bins,
        options.face_alpha,
        options.display_name.as_deref(),
        None,
    )?;
    apply_axes_target(axes_target, NAME).map_err(map_plot_error)?;
    if let Some(limits) = color_limits_snapshot() {
        chart.surface = chart.surface.with_color_limits(Some(limits));
    }
    render_binscatter(inputs, options, [x_bins, y_bins], chart)
}

async fn gather_values(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    args.try_map_preserving_roots_async(args.len(), runtime, async |index| {
        gather_if_needed_async(&args[index])
            .await
            .map_err(|err| invalid_with_source(format!("binscatter: {err}"), err))
    })
    .await
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod provider_tests;
