//! MATLAB-compatible Communications Toolbox `scatterplot` builtin.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, SCATTERPLOT_ERROR_INVALID_ARGUMENT, SCATTERPLOT_ERROR_PLOT_FAILED,
    SCATTERPLOT_INTEGER_CONTROL_EXTENSION, SCATTERPLOT_INTEGER_DATA_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::plotting::state::{figure_handle_exists, select_figure, FigureHandle};
use crate::builtins::plotting::state::{set_axis_equal, set_axis_limits, set_grid_enabled};
use crate::builtins::plotting::style::{parse_line_style_args, LineStyleParseOptions};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::gpu_helpers::axis_bounds;
use super::scatter::scatter_builtin;

const BUILTIN_NAME: &str = "scatterplot";
const DEFAULT_MARKER: &str = "b.";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::scatterplot")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "scatterplot",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "scatterplot extracts GPU-resident real/imag components into provider buffers and forwards them to scatter without host readback when no decimation is requested; decimated or unsupported inputs gather once.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::scatterplot")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "scatterplot",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "scatterplot is a rendering sink and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "scatterplot",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::scatterplot"
)]
pub async fn scatterplot_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    x: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<f64> {
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), BUILTIN_NAME)?;
    let _arguments = rest.fork();
    let result = async {
        crate::builtins::common::validation::reject_typed_complex_integer(&x, BUILTIN_NAME)?;
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            &x,
            &SCATTERPLOT_INTEGER_DATA_EXTENSION,
            BUILTIN_NAME,
            "sample",
        )
        .await?;
        for control in rest.iter().take(2) {
            if crate::builtins::common::validation::value_has_native_integer_class(control) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &SCATTERPLOT_INTEGER_CONTROL_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
        }
        let options = ScatterplotOptions::parse(rest)?;
        let (x_value, y_value, limits) =
            extract_scatter_values(x, options.n, options.offset).await?;
        let marker = options
            .marker
            .unwrap_or_else(|| Value::string(DEFAULT_MARKER.to_string()));

        work.check()?;
        if let Some(figure) = options.figure {
            select_figure(figure);
        }
        work.check()?;
        let coordinates = [x_value, y_value, marker];
        let arguments =
            crate::call::BuiltinArguments::from_borrowed(&coordinates, runtime.as_ref())?;
        let handle = scatter_builtin(runtime.clone(), arguments)
            .await
            .map_err(scatterplot_map_plot_error)?;
        work.check()?;
        set_axis_equal(true);
        set_grid_enabled(true);
        if let Some(limit) = limits {
            set_axis_limits(Some(limit), Some(limit));
        }
        work.finish(Ok(handle))
    }
    .await;
    work.finish(result)
}

#[derive(Clone, Debug)]
struct ScatterplotOptions {
    n: usize,
    offset: usize,
    marker: Option<Value>,
    figure: Option<FigureHandle>,
}

impl ScatterplotOptions {
    fn parse(rest: crate::call::BuiltinArguments) -> BuiltinResult<Self> {
        if rest.len() > 4 {
            return Err(scatterplot_error(
                SCATTERPLOT_ERROR_INVALID_ARGUMENT.message,
                &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
            ));
        }

        let n = match rest.first() {
            Some(value) => parse_nonnegative_integer(value, "n").and_then(|n| {
                if n == 0 {
                    Err(scatterplot_error(
                        "scatterplot: n must be a positive integer",
                        &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
                    ))
                } else {
                    Ok(n)
                }
            })?,
            None => 1,
        };
        let offset = match rest.get(1) {
            Some(value) => parse_nonnegative_integer(value, "offset")?,
            None => 0,
        };
        if offset >= n {
            return Err(scatterplot_error(
                "scatterplot: offset must be less than n",
                &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
            ));
        }
        let marker = match rest.get(2) {
            Some(value) => Some(parse_marker(value)?),
            None => None,
        };
        let figure = match rest.get(3) {
            Some(value) => Some(parse_figure_handle(value)?),
            None => None,
        };
        Ok(Self {
            n,
            offset,
            marker,
            figure,
        })
    }
}

fn parse_marker(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::String(runmat_value::StringElement::Text(_)) => {}
        Value::CharArray(chars) if chars.rows == 1 => {}
        Value::CharArray(_) => {
            return Err(scatterplot_error(
                "scatterplot: marker must be a character row vector or string scalar",
                &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
            ));
        }
        other => {
            return Err(scatterplot_error(
                format!("scatterplot: marker must be a LineSpec string, got {other:?}"),
                &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
            ));
        }
    }

    let opts = LineStyleParseOptions {
        builtin_name: BUILTIN_NAME,
        forbid_leading_numeric: true,
        forbid_interleaved_numeric: true,
        accepts_handle_visibility: false,
    };
    parse_line_style_args(std::slice::from_ref(value), &opts).map_err(|err| {
        scatterplot_error(
            format!("scatterplot: invalid marker LineSpec: {}", err.message()),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        )
    })?;
    Ok(value.clone())
}

fn parse_figure_handle(value: &Value) -> BuiltinResult<FigureHandle> {
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        return Err(scatterplot_error(
            "scatterplot: scatfig must be a double Figure handle",
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    let scalar = parse_numeric_scalar(value, "scatfig")?;
    if !scalar.is_finite() || scalar <= 0.0 || scalar.fract() != 0.0 || scalar > f64::from(u32::MAX)
    {
        return Err(scatterplot_error(
            "scatterplot: scatfig must be a valid Figure handle",
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    let figure = FigureHandle::from(scalar as u32);
    if !figure_handle_exists(figure) {
        return Err(scatterplot_error(
            "scatterplot: scatfig must identify an existing Figure",
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(figure)
}

async fn extract_scatter_values(
    value: Value,
    n: usize,
    offset: usize,
) -> BuiltinResult<(Value, Value, Option<(f64, f64)>)> {
    match value {
        Value::GpuTensor(handle) if n == 1 && offset == 0 => {
            if let Some((real, imag)) = gpu_real_imag_handles(&handle).await? {
                let limits = symmetric_limits_from_gpu_bounds(&real, &imag);
                return Ok((Value::GpuTensor(real), Value::GpuTensor(imag), limits));
            }
            let gathered = gather_gpu_value(handle).await?;
            let (real, imag, limits) = extract_host_points(gathered, n, offset)?;
            Ok((Value::Tensor(real), Value::Tensor(imag), limits))
        }
        Value::GpuTensor(handle) => {
            let gathered = gather_gpu_value(handle).await?;
            let (real, imag, limits) = extract_host_points(gathered, n, offset)?;
            Ok((Value::Tensor(real), Value::Tensor(imag), limits))
        }
        other => {
            let (real, imag, limits) = extract_host_points(other, n, offset)?;
            Ok((Value::Tensor(real), Value::Tensor(imag), limits))
        }
    }
}

async fn gpu_real_imag_handles(
    handle: &GpuTensorHandle,
) -> BuiltinResult<Option<(GpuTensorHandle, GpuTensorHandle)>> {
    // Decomposition is optional preparation before any figure/registry effect.
    // A declined compute route uses the existing mandatory plotting readback,
    // never a whole-builtin replay or permission to compute under that policy.
    use crate::builtins::common::gpu_helpers::{
        resolve_resident_compute_owner, ResidentComputeDecline,
    };
    let provider = match resolve_resident_compute_owner(handle) {
        Ok(provider) => provider,
        Err(ResidentComputeDecline::PolicyDenied) => return Ok(None),
        Err(decline) => {
            return Err(decline.with_domain_error(scatterplot_error(
                "scatterplot: no exact provider owns the sample input",
                &SCATTERPLOT_ERROR_PLOT_FAILED,
            )))
        }
    };
    let Ok(real) = provider.unary_real(handle).await else {
        return Ok(None);
    };
    let Ok(imag) = provider.unary_imag(handle).await else {
        return Ok(None);
    };
    Ok(Some((real, imag)))
}

async fn gather_gpu_value(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let value = Value::GpuTensor(handle);
    crate::gather_if_needed_async(&value)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))
}

type HostPointTensors = (Tensor, Tensor, Option<(f64, f64)>);

fn extract_host_points(value: Value, n: usize, offset: usize) -> BuiltinResult<HostPointTensors> {
    let samples = complex_samples(value)?;
    if samples.is_empty() {
        return Err(scatterplot_error(
            "scatterplot: input samples cannot be empty",
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    let selected: Vec<(f64, f64)> = samples.into_iter().skip(offset).step_by(n).collect();
    if selected.is_empty() {
        return Err(scatterplot_error(
            "scatterplot: decimation selects no samples",
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    let mut real = Vec::with_capacity(selected.len());
    let mut imag = Vec::with_capacity(selected.len());
    for (re, im) in selected {
        real.push(re);
        imag.push(im);
    }
    let limits = symmetric_limits(&real, &imag);
    let shape = vec![real.len(), 1];
    let x = Tensor::new(real, shape.clone()).map_err(|err| {
        scatterplot_error(
            format!("scatterplot: {err}"),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        )
    })?;
    let y = Tensor::new(imag, shape).map_err(|err| {
        scatterplot_error(
            format!("scatterplot: {err}"),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        )
    })?;
    Ok((x, y, limits))
}

fn complex_samples(value: Value) -> BuiltinResult<Vec<(f64, f64)>> {
    match value {
        Value::Complex(re, im) => Ok(vec![(re, im)]),
        Value::ComplexTensor(tensor) => Ok(tensor.materialize_f64()),
        Value::Num(v) => Ok(vec![(v, 0.0)]),
        Value::Int(v) => Ok(vec![(v.to_f64(), 0.0)]),
        Value::Bool(v) => Ok(vec![(if v { 1.0 } else { 0.0 }, 0.0)]),
        Value::Tensor(tensor) => Ok(tensor_utils::tensor_values_f64(&tensor)
            .into_iter()
            .map(|v| (v, 0.0))
            .collect()),
        Value::LogicalArray(logical) => Ok(logical
            .data
            .into_iter()
            .map(|v| (if v != 0 { 1.0 } else { 0.0 }, 0.0))
            .collect()),
        other => Err(scatterplot_error(
            format!("scatterplot: expected numeric or complex samples, got {other:?}"),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn parse_nonnegative_integer(value: &Value, name: &str) -> BuiltinResult<usize> {
    if let Some(integer) = tensor_utils::scalar_integer_value(value) {
        return integer.try_to_usize().ok_or_else(|| {
            scatterplot_error(
                format!("scatterplot: {name} is too large"),
                &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
            )
        });
    }
    let scalar = parse_numeric_scalar(value, name)?;
    if !scalar.is_finite() || scalar < 0.0 || scalar.fract() != 0.0 {
        return Err(scatterplot_error(
            format!("scatterplot: {name} must be a nonnegative integer scalar"),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    if scalar > usize::MAX as f64 || (usize::BITS == 64 && scalar == usize::MAX as f64) {
        return Err(scatterplot_error(
            format!("scatterplot: {name} is too large"),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(scalar as usize)
}

fn parse_numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(v) => Ok(*v),
        Value::Int(v) => Ok(v.to_f64()),
        Value::Bool(v) => Ok(if *v { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Ok(tensor_utils::tensor_value_f64(tensor, 0))
        }
        other => Err(scatterplot_error(
            format!("scatterplot: {name} must be a numeric scalar, got {other:?}"),
            &SCATTERPLOT_ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn symmetric_limits(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let max_abs = x
        .iter()
        .chain(y.iter())
        .copied()
        .filter(|v| v.is_finite())
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if max_abs == 0.0 {
        Some((-1.0, 1.0))
    } else if max_abs.is_finite() {
        let padded = max_abs * 1.05;
        Some((-padded, padded))
    } else {
        None
    }
}

fn symmetric_limits_from_gpu_bounds(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
) -> Option<(f64, f64)> {
    let (xmin, xmax) = axis_bounds(x, BUILTIN_NAME).ok()?;
    let (ymin, ymax) = axis_bounds(y, BUILTIN_NAME).ok()?;
    symmetric_limits(&[xmin as f64, xmax as f64], &[ymin as f64, ymax as f64])
}

fn scatterplot_error(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn scatterplot_map_plot_error(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {}",
        SCATTERPLOT_ERROR_PLOT_FAILED.message,
        err.message()
    ))
    .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = SCATTERPLOT_ERROR_PLOT_FAILED.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.with_source(err).build()
}

#[cfg(test)]
#[path = "scatterplot/tests.rs"]
mod tests;
