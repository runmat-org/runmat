use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::ERRORBAR_ERROR_INVALID_ARGUMENT;
use runmat_macros::runtime_builtin;
use runmat_plot::gpu::errorbar::ErrorBarGpuInputs;
use runmat_plot::gpu::line::{
    self, LineGpuInputs as MarkerGpuInputs, LineGpuParams as MarkerGpuParams,
};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::{ErrorBar, LineMarkerAppearance};
use runmat_value::{Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::{build_runtime_error, RuntimeError};

use super::gpu_helpers::axis_bounds_protecting;
use super::op_common::line_inputs::NumericInput;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{marker_metadata_from_appearance, parse_line_style_args, LineStyleParseOptions};

const BUILTIN_NAME: &str = "errorbar";
type ErrorBarArgs = (
    Option<usize>,
    Value,
    Value,
    Option<Value>,
    Option<Value>,
    Value,
    Value,
    Vec<Value>,
);

fn errorbar_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_errorbar_invalid(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    errorbar_error_with_detail(&ERRORBAR_ERROR_INVALID_ARGUMENT, err.message)
}

fn errorbar_invalid(detail: impl AsRef<str>) -> RuntimeError {
    errorbar_error_with_detail(&ERRORBAR_ERROR_INVALID_ARGUMENT, detail)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::errorbar")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "errorbar",
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
    notes: "errorbar is a plotting sink; GPU inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::errorbar")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "errorbar",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "errorbar performs rendering and terminates fusion graphs.",
};

#[runtime_builtin(
    name = "errorbar",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::errorbar"
)]
pub async fn errorbar_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::ERRORBAR_DESCRIPTOR,
        &runmat_builtins::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
        "errorbar",
    )?;
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "errorbar")?;
    // Keep the original rooted owner alive even when helpers extract native data.
    let _arguments = args.fork();
    work.finish(errorbar_impl(args, &work).await)
}

async fn errorbar_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(errorbar_invalid("too many output arguments"));
    }
    let (target_axes, x, y, x_neg, x_pos, y_neg, y_pos, rest) =
        parse_errorbar_args(args).map_err(map_errorbar_invalid)?;
    let parsed = parse_errorbar_style_args(&rest).map_err(map_errorbar_invalid)?;
    let mut x_in = Some(errorbar_numeric_input(x).map_err(map_errorbar_invalid)?);
    let mut y_in = Some(errorbar_numeric_input(y).map_err(map_errorbar_invalid)?);
    let mut xn_in = x_neg
        .map(errorbar_numeric_input)
        .transpose()
        .map_err(map_errorbar_invalid)?;
    let mut xp_in = x_pos
        .map(errorbar_numeric_input)
        .transpose()
        .map_err(map_errorbar_invalid)?;
    let mut n_in = Some(errorbar_numeric_input(y_neg).map_err(map_errorbar_invalid)?);
    let mut p_in = Some(errorbar_numeric_input(y_pos).map_err(map_errorbar_invalid)?);
    let mut prepared = async move {
        let x_arg = x_in.take().expect("x consumed");
        let y_arg = y_in.take().expect("y consumed");
        let yn_arg = n_in.take().expect("yn consumed");
        let yp_arg = p_in.take().expect("yp consumed");
        let label = parsed.label.clone().unwrap_or_else(|| "Data".into());
        if parsed.orientation == runmat_plot::plots::errorbar::ErrorBarOrientation::Vertical
            && errorbar_gpu_inputs_eligible(
                &x_arg,
                &y_arg,
                xn_in.as_ref(),
                xp_in.as_ref(),
                &yn_arg,
                &yp_arg,
            )
        {
            if let (Some(x_gpu), Some(y_gpu), Some(yn_gpu), Some(yp_gpu)) = (
                x_arg.gpu_handle(),
                y_arg.gpu_handle(),
                yn_arg.gpu_handle(),
                yp_arg.gpu_handle(),
            ) {
                if let Some(plot) = build_errorbar_gpu_plot(
                    BUILTIN_NAME,
                    x_gpu,
                    y_gpu,
                    xn_in.as_ref().and_then(|v| v.gpu_handle()),
                    xp_in.as_ref().and_then(|v| v.gpu_handle()),
                    yn_gpu,
                    yp_gpu,
                    &parsed,
                    &label,
                )
                .await?
                {
                    return Ok(vec![plot]);
                }
            }
        }
        let x = x_arg
            .into_tensor_async(BUILTIN_NAME)
            .await
            .map_err(map_errorbar_invalid)?;
        let y = y_arg
            .into_tensor_async(BUILTIN_NAME)
            .await
            .map_err(map_errorbar_invalid)?;
        let xn = match xn_in.take() {
            Some(value) => Some(
                value
                    .into_tensor_async(BUILTIN_NAME)
                    .await
                    .map_err(map_errorbar_invalid)?,
            ),
            None => None,
        };
        let xp = match xp_in.take() {
            Some(value) => Some(
                value
                    .into_tensor_async(BUILTIN_NAME)
                    .await
                    .map_err(map_errorbar_invalid)?,
            ),
            None => None,
        };
        let yn = yn_arg
            .into_tensor_async(BUILTIN_NAME)
            .await
            .map_err(map_errorbar_invalid)?;
        let yp = yp_arg
            .into_tensor_async(BUILTIN_NAME)
            .await
            .map_err(map_errorbar_invalid)?;
        let plots = build_errorbar_host_plots(x, y, xn, xp, yn, yp, &parsed, &label)
            .map_err(map_errorbar_invalid)?;
        Ok::<_, crate::RuntimeError>(plots)
    }
    .await?;
    let opts = PlotRenderOptions::default();
    let plot_indices_out = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plot_indices_slot = std::rc::Rc::clone(&plot_indices_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    work.check()?;
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let axes = target_axes.unwrap_or(axes);
        for plot in prepared.drain(..) {
            let plot_index = figure.add_errorbar_on_axes(plot, axes);
            plot_indices_slot.borrow_mut().push((axes, plot_index));
        }
        Ok(())
    });
    if plot_indices_out.borrow().is_empty() {
        return render_result.map(|_| f64::NAN);
    }
    let handles: Vec<f64> = plot_indices_out
        .borrow()
        .iter()
        .map(|(axes, plot_index)| {
            crate::builtins::plotting::state::register_errorbar_handle(
                figure_handle,
                *axes,
                *plot_index,
            )
        })
        .collect();
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handles[0]);
        }
        return Err(map_errorbar_invalid(err));
    }
    // The runtime builtin ABI is scalar today. Matrix calls register every
    // series handle and return the first; the descriptor/type ABI must be
    // widened before the complete handle vector can cross this boundary.
    Ok(handles[0])
}

fn errorbar_numeric_input(value: Value) -> crate::BuiltinResult<NumericInput> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(errorbar_invalid(
            "complex coordinate and error data are not supported",
        )),
        Value::LogicalArray(array) => {
            let data = array
                .data
                .iter()
                .map(|value| if *value == 0 { 0.0 } else { 1.0 })
                .collect();
            let tensor = Tensor::new(data, array.shape)
                .map_err(|err| errorbar_invalid(format!("logical input: {err}")))?;
            Ok(NumericInput::Host(tensor))
        }
        other => NumericInput::from_value(other, BUILTIN_NAME),
    }
}

fn errorbar_gpu_inputs_eligible(
    x: &NumericInput,
    y: &NumericInput,
    x_neg: Option<&NumericInput>,
    x_pos: Option<&NumericInput>,
    y_neg: &NumericInput,
    y_pos: &NumericInput,
) -> bool {
    let eligible = |input: &NumericInput| match input.gpu_handle() {
        Some(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::Real
                && handle.shape().iter().filter(|extent| **extent > 1).count() <= 1
        }
        None => false,
    };
    eligible(x)
        && eligible(y)
        && eligible(y_neg)
        && eligible(y_pos)
        && x_neg.is_none_or(eligible)
        && x_pos.is_none_or(eligible)
}

#[derive(Clone, Copy)]
enum ErrorBarSeriesLayout {
    Vector,
    MatrixColumns { rows: usize, cols: usize },
    MatrixRows { rows: usize, cols: usize },
}

#[allow(clippy::too_many_arguments)]
fn build_errorbar_host_plots(
    x: Tensor,
    y: Tensor,
    x_neg: Option<Tensor>,
    x_pos: Option<Tensor>,
    y_neg: Tensor,
    y_pos: Tensor,
    parsed: &ParsedErrorBarStyle,
    label: &str,
) -> crate::BuiltinResult<Vec<ErrorBar>> {
    let tensors = [
        Some(&x),
        Some(&y),
        x_neg.as_ref(),
        x_pos.as_ref(),
        Some(&y_neg),
        Some(&y_pos),
    ];
    for tensor in tensors.into_iter().flatten() {
        if tensor.shape().len() > 2 && tensor.shape().iter().skip(2).any(|extent| *extent != 1) {
            return Err(errorbar_invalid(
                "inputs must be vectors or two-dimensional matrices",
            ));
        }
    }
    let matrix_shape = tensors
        .into_iter()
        .flatten()
        .find(|tensor| tensor.rows() > 1 && tensor.cols() > 1)
        .map(|tensor| (tensor.rows(), tensor.cols()));
    let layout = if let Some((rows, cols)) = matrix_shape {
        for tensor in tensors.into_iter().flatten() {
            if tensor.rows() > 1
                && tensor.cols() > 1
                && (tensor.rows(), tensor.cols()) != (rows, cols)
            {
                return Err(errorbar_invalid(
                    "all matrix inputs must have the same size and orientation",
                ));
            }
        }
        let vectors_match = |points: usize| {
            tensors.into_iter().flatten().all(|tensor| {
                (tensor.rows() > 1 && tensor.cols() > 1)
                    || tensor.len() == points
                    || tensor.is_empty()
            })
        };
        if vectors_match(rows) {
            ErrorBarSeriesLayout::MatrixColumns { rows, cols }
        } else if vectors_match(cols) {
            ErrorBarSeriesLayout::MatrixRows { rows, cols }
        } else {
            return Err(errorbar_invalid(
                "vector lengths must match one dimension of the matrix inputs",
            ));
        }
    } else {
        ErrorBarSeriesLayout::Vector
    };
    let series_count = match layout {
        ErrorBarSeriesLayout::Vector => 1,
        ErrorBarSeriesLayout::MatrixColumns { cols, .. } => cols,
        ErrorBarSeriesLayout::MatrixRows { rows, .. } => rows,
    };
    let mut plots = Vec::with_capacity(series_count);
    for series in 0..series_count {
        let x_values = errorbar_series_values(&x, layout, series)?;
        let y_values = errorbar_series_values(&y, layout, series)?;
        let xn_values = x_neg
            .as_ref()
            .map(|tensor| errorbar_series_values(tensor, layout, series))
            .transpose()?;
        let xp_values = x_pos
            .as_ref()
            .map(|tensor| errorbar_series_values(tensor, layout, series))
            .transpose()?;
        let yn_values = errorbar_series_values(&y_neg, layout, series)?;
        let yp_values = errorbar_series_values(&y_pos, layout, series)?;
        let series_label = if series_count == 1 {
            label.to_string()
        } else {
            format!("{label} {}", series + 1)
        };
        plots.push(build_errorbar_series(
            x_values,
            y_values,
            xn_values,
            xp_values,
            yn_values,
            yp_values,
            parsed,
            series_label,
        )?);
    }
    Ok(plots)
}

fn errorbar_series_values(
    tensor: &Tensor,
    layout: ErrorBarSeriesLayout,
    series: usize,
) -> crate::BuiltinResult<Vec<f64>> {
    let value = |index| {
        tensor
            .numeric_value_at(index)
            .map(runmat_value::NumericScalar::materialize_f64)
            .ok_or_else(|| errorbar_invalid("coordinate/error storage index exceeds input length"))
    };
    let is_matrix = tensor.rows() > 1 && tensor.cols() > 1;
    match layout {
        ErrorBarSeriesLayout::MatrixColumns { rows, .. } if is_matrix => {
            (series * rows..(series + 1) * rows).map(value).collect()
        }
        ErrorBarSeriesLayout::MatrixRows { rows, cols } if is_matrix => {
            (0..cols).map(|col| value(series + rows * col)).collect()
        }
        _ => (0..tensor.len()).map(value).collect(),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_errorbar_series(
    x: Vec<f64>,
    y: Vec<f64>,
    x_neg: Option<Vec<f64>>,
    x_pos: Option<Vec<f64>>,
    y_neg: Vec<f64>,
    y_pos: Vec<f64>,
    parsed: &ParsedErrorBarStyle,
    label: String,
) -> crate::BuiltinResult<ErrorBar> {
    let mut plot = if let (Some(x_neg), Some(x_pos)) = (x_neg, x_pos) {
        ErrorBar::new_both(x, y, x_neg, x_pos, y_neg, y_pos)
    } else {
        match parsed.orientation {
            runmat_plot::plots::errorbar::ErrorBarOrientation::Vertical => {
                ErrorBar::new_vertical(x, y, y_neg, y_pos)
            }
            runmat_plot::plots::errorbar::ErrorBarOrientation::Horizontal => {
                let len = x.len();
                ErrorBar::new_both(x, y, y_neg, y_pos, vec![0.0; len], vec![0.0; len]).map(
                    |mut plot| {
                        plot.orientation =
                            runmat_plot::plots::errorbar::ErrorBarOrientation::Horizontal;
                        plot
                    },
                )
            }
            runmat_plot::plots::errorbar::ErrorBarOrientation::Both => {
                ErrorBar::new_both(x, y, y_neg.clone(), y_pos.clone(), y_neg, y_pos)
            }
        }
    }
    .map_err(|err| errorbar_invalid(&err))?
    .with_style(
        parsed.color,
        parsed.line_width,
        parsed.line_style,
        parsed.cap_size,
    )
    .with_label(label);
    if let Some(marker) = parsed.marker.clone() {
        plot.set_marker(Some(marker));
    }
    if let Some(visible) = parsed.visible {
        plot.set_visible(visible);
    }
    Ok(plot)
}

async fn build_errorbar_gpu_plot(
    name: &'static str,
    x: &runmat_accelerate_api::GpuTensorHandle,
    y: &runmat_accelerate_api::GpuTensorHandle,
    x_neg: Option<&runmat_accelerate_api::GpuTensorHandle>,
    x_pos: Option<&runmat_accelerate_api::GpuTensorHandle>,
    y_neg: &runmat_accelerate_api::GpuTensorHandle,
    y_pos: &runmat_accelerate_api::GpuTensorHandle,
    parsed: &ParsedErrorBarStyle,
    label: &str,
) -> crate::BuiltinResult<Option<ErrorBar>> {
    let mut gpu_exports = super::gpu_helpers::GpuBufferExports::default();
    let precision = runmat_accelerate_api::handle_precision(x);
    if [y, y_neg, y_pos]
        .into_iter()
        .chain(x_neg)
        .chain(x_pos)
        .any(|input| runmat_accelerate_api::handle_precision(input) != precision)
    {
        return Ok(None);
    }
    let (Some(x_ref), Some(y_ref), Some(yn_ref), Some(yp_ref)) = (
        gpu_exports.export(x, name)?,
        gpu_exports.export(y, name)?,
        gpu_exports.export(y_neg, name)?,
        gpu_exports.export(y_pos, name)?,
    ) else {
        return Ok(None);
    };
    let xn_ref = x_neg
        .map(|input| gpu_exports.export(input, name))
        .transpose()?
        .flatten();
    let xp_ref = x_pos
        .map(|input| gpu_exports.export(input, name))
        .transpose()?
        .flatten();
    if (x_neg.is_some() && xn_ref.is_none()) || (x_pos.is_some() && xp_ref.is_none()) {
        return Ok(None);
    }
    let context = super::gpu_helpers::ensure_shared_wgpu_context(name)?;
    if x_ref.len != y_ref.len
        || x_ref.len != yn_ref.len
        || x_ref.len != yp_ref.len
        || xn_ref.as_ref().map(|r| r.len).unwrap_or(x_ref.len) != x_ref.len
        || xp_ref.as_ref().map(|r| r.len).unwrap_or(x_ref.len) != x_ref.len
    {
        return Err(plotting_error(
            name,
            format!("{name}: X, Y, and error inputs must have identical lengths"),
        ));
    }
    if x_ref.precision != y_ref.precision
        || x_ref.precision != yn_ref.precision
        || x_ref.precision != yp_ref.precision
        || xn_ref
            .as_ref()
            .map(|r| r.precision)
            .unwrap_or(x_ref.precision)
            != x_ref.precision
        || xp_ref
            .as_ref()
            .map(|r| r.precision)
            .unwrap_or(x_ref.precision)
            != x_ref.precision
    {
        return Ok(None);
    }
    let scalar =
        ScalarType::from_is_f64(x_ref.precision == runmat_accelerate_api::ProviderPrecision::F64);
    let mut protected = vec![x, y, y_neg, y_pos];
    protected.extend(x_neg);
    protected.extend(x_pos);
    let (min_x, max_x) = axis_bounds_protecting(x, &protected, name).await?;
    let (min_y, max_y) = axis_bounds_protecting(y, &protected, name).await?;
    let (_, max_neg) = axis_bounds_protecting(y_neg, &protected, name).await?;
    let (_, max_pos) = axis_bounds_protecting(y_pos, &protected, name).await?;
    let mut bounds = runmat_plot::core::BoundingBox::new(
        glam::Vec3::new(min_x, min_y - max_neg, 0.0),
        glam::Vec3::new(max_x, max_y + max_pos, 0.0),
    );
    if let (Some(xn), Some(xp)) = (x_neg, x_pos) {
        let (min_xn, max_xn) = axis_bounds_protecting(xn, &protected, name).await?;
        let (min_xp, max_xp) = axis_bounds_protecting(xp, &protected, name).await?;
        bounds.min.x -= max_xn.max(min_xn.abs());
        bounds.max.x += max_xp.max(min_xp.abs());
    }
    let inputs = ErrorBarGpuInputs {
        x_buffer: x_ref.clone().into(),
        y_buffer: y_ref.clone().into(),
        x_neg_buffer: xn_ref.as_ref().map(|r| r.clone().into()),
        x_pos_buffer: xp_ref.as_ref().map(|r| r.clone().into()),
        y_neg_buffer: yn_ref.clone().into(),
        y_pos_buffer: yp_ref.clone().into(),
        len: x_ref.len as u32,
        scalar,
    };
    let mut plot = ErrorBar::from_gpu_inputs(
        parsed.color,
        parsed.line_width,
        parsed.line_style,
        parsed.cap_size,
        if x_neg.is_some() && x_pos.is_some() {
            runmat_plot::plots::errorbar::ErrorBarOrientation::Both
        } else {
            runmat_plot::plots::errorbar::ErrorBarOrientation::Vertical
        },
        inputs,
        bounds,
    )
    .with_label(label);
    if let Some(marker) = parsed.marker.clone() {
        let marker_gpu = context
            .access
            .with_shared(|emission| {
                line::pack_marker_vertices_from_xy(
                    emission,
                    &MarkerGpuInputs {
                        x_buffer: x_ref.clone().into(),
                        y_buffer: y_ref.clone().into(),
                        len: x_ref.len as u32,
                        scalar,
                    },
                    &MarkerGpuParams {
                        color: marker.face_color,
                        half_width_px: 0.0,
                        viewport_width_px: 1.0,
                        viewport_height_px: 1.0,
                        x_min: 0.0,
                        x_span: 1.0,
                        y_min: 0.0,
                        y_span: 1.0,
                        x_log: false,
                        y_log: false,
                        line_style: runmat_plot::plots::LineStyle::Solid,
                        marker_size: marker.size,
                    },
                )
            })
            .map_err(|error| error.to_string())
            .and_then(|result| result)
            .map_err(|e| {
                plotting_error(
                    name,
                    format!("{name}: failed to build marker vertices: {e}"),
                )
            })?;
        plot.set_marker(Some(marker));
        plot.set_marker_gpu_vertices(Some(marker_gpu));
    }
    if let Some(visible) = parsed.visible {
        plot.set_visible(visible);
    }
    Ok(Some(plot))
}

struct ParsedErrorBarStyle {
    visible: Option<bool>,
    color: glam::Vec4,
    line_width: f32,
    line_style: runmat_plot::plots::LineStyle,
    marker: Option<LineMarkerAppearance>,
    label: Option<String>,
    cap_size: f32,
    orientation: runmat_plot::plots::errorbar::ErrorBarOrientation,
}

fn parse_errorbar_style_args(args: &[Value]) -> crate::BuiltinResult<ParsedErrorBarStyle> {
    let mut filtered = Vec::new();
    let mut cap_size = 6.0;
    let mut orientation = runmat_plot::plots::errorbar::ErrorBarOrientation::Vertical;
    let mut idx = 0usize;
    while idx < args.len() {
        if let Some(key) = super::style::value_as_string(&args[idx]).map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "errorbar",
                source.to_string(),
                source,
            )
        })? {
            match (idx, key.trim().to_ascii_lowercase().as_str()) {
                (0, "vertical") => {
                    orientation = runmat_plot::plots::errorbar::ErrorBarOrientation::Vertical;
                    idx += 1;
                    continue;
                }
                (0, "horizontal") => {
                    orientation = runmat_plot::plots::errorbar::ErrorBarOrientation::Horizontal;
                    idx += 1;
                    continue;
                }
                (0, "both") => {
                    orientation = runmat_plot::plots::errorbar::ErrorBarOrientation::Both;
                    idx += 1;
                    continue;
                }
                _ => {}
            }
            if key.trim().eq_ignore_ascii_case("CapSize") && idx + 1 < args.len() {
                cap_size = super::style::value_as_f64(&args[idx + 1]).ok_or_else(|| {
                    plotting_error(BUILTIN_NAME, "errorbar: CapSize must be numeric")
                })? as f32;
                idx += 2;
                continue;
            }
        }
        filtered.push(args[idx].clone());
        idx += 1;
    }
    let parsed = parse_line_style_args(&filtered, &LineStyleParseOptions::generic(BUILTIN_NAME))?;
    let marker = marker_metadata_from_appearance(&parsed.appearance);
    Ok(ParsedErrorBarStyle {
        visible: parsed.appearance.visible,
        color: parsed.appearance.color,
        line_width: parsed.appearance.line_width,
        line_style: parsed.appearance.line_style,
        marker,
        label: parsed.label,
        cap_size,
        orientation,
    })
}

fn parse_errorbar_args(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<ErrorBarArgs> {
    if args.len() < 2 {
        return Err(plotting_error(
            BUILTIN_NAME,
            "errorbar: expected at least y and error inputs",
        ));
    }
    let mut it = args.into_iter();
    let mut target_axes = None;
    let first = it.next().unwrap();
    let first = if let Ok(crate::builtins::plotting::properties::PlotHandle::Axes(_, axes)) =
        crate::builtins::plotting::properties::resolve_plot_handle(&first, BUILTIN_NAME)
    {
        target_axes = Some(axes);
        it.next().ok_or_else(|| {
            plotting_error(BUILTIN_NAME, "errorbar: expected data after axes handle")
        })?
    } else {
        first
    };
    let second = it
        .next()
        .ok_or_else(|| errorbar_invalid("expected data and error inputs after axes handle"))?;
    let third = it.next();
    let fourth = it.next();
    match (third, fourth) {
        (None, _) => {
            let y = first;
            let err = second;
            let x = infer_errorbar_x_from_y(&y)?;
            Ok((target_axes, x, y, None, None, err.clone(), err, Vec::new()))
        }
        (Some(third), None) => {
            if is_styleish(&third) {
                let y = first;
                let err = second;
                let x = infer_errorbar_x_from_y(&y)?;
                Ok((target_axes, x, y, None, None, err.clone(), err, vec![third]))
            } else {
                Ok((
                    target_axes,
                    first,
                    second,
                    None,
                    None,
                    third.clone(),
                    third,
                    Vec::new(),
                ))
            }
        }
        (Some(third), Some(fourth)) => {
            if is_styleish(&third) {
                let y = first;
                let err = second;
                let x = infer_errorbar_x_from_y(&y)?;
                let mut rest = vec![third, fourth];
                rest.extend(it);
                Ok((target_axes, x, y, None, None, err.clone(), err, rest))
            } else if is_styleish(&fourth) {
                let mut rest = vec![fourth];
                rest.extend(it);
                Ok((
                    target_axes,
                    first,
                    second,
                    None,
                    None,
                    third.clone(),
                    third,
                    rest,
                ))
            } else {
                let rest = it.collect::<Vec<_>>();
                match rest.as_slice() {
                    [] => Ok((target_axes, first, second, None, None, third, fourth, rest)),
                    [fifth, ..] if is_styleish(fifth) => {
                        Ok((target_axes, first, second, None, None, third, fourth, rest))
                    }
                    [fifth, sixth, tail @ ..] if is_numericish(fifth) && is_numericish(sixth) => {
                        Ok((
                            target_axes,
                            first,
                            second,
                            Some(fifth.clone()),
                            Some(sixth.clone()),
                            third,
                            fourth,
                            tail.to_vec(),
                        ))
                    }
                    [fifth, ..] if is_numericish(fifth) => Err(plotting_error(
                        BUILTIN_NAME,
                        "errorbar: expected positive X error data after negative X error data",
                    )),
                    _ => Ok((target_axes, first, second, None, None, third, fourth, rest)),
                }
            }
        }
    }
}

fn is_styleish(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::CharArray(_))
}

fn is_numericish(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(_)
            | Value::GpuTensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::LogicalArray(_)
            | Value::Complex(_, _)
            | Value::ComplexTensor(_)
    )
}

fn infer_errorbar_x_from_y(y: &Value) -> crate::BuiltinResult<Value> {
    let len = match y {
        Value::Tensor(tensor) => {
            if tensor.rows() > 1 && tensor.cols() > 1 {
                tensor.rows()
            } else {
                tensor_utils::tensor_element_len(tensor)
            }
        }
        Value::LogicalArray(array) => {
            if array.shape.first().copied().unwrap_or(1) > 1
                && array.shape.get(1).copied().unwrap_or(1) > 1
            {
                array.shape[0]
            } else {
                array.data.len()
            }
        }
        Value::GpuTensor(handle) => {
            if handle.shape().first().copied().unwrap_or(1) > 1
                && handle.shape().get(1).copied().unwrap_or(1) > 1
            {
                handle.shape()[0]
            } else {
                handle.shape().iter().product()
            }
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => 1,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err(plotting_error(
                BUILTIN_NAME,
                "errorbar: complex coordinate data are not supported",
            ))
        }
        other => {
            return Err(plotting_error(
                BUILTIN_NAME,
                format!("errorbar: unsupported Y input {other:?}"),
            ))
        }
    };
    Ok(Value::Tensor(
        Tensor::new((1..=len).map(|i| i as f64).collect(), vec![len])
            .expect("implicit errorbar axis"),
    ))
}

#[cfg(test)]
mod tests {
    use runmat_builtins::BuiltinIntegerInputAvailability;
    use runmat_builtins::{ERRORBAR_DESCRIPTOR, ERRORBAR_INTEGER_CAPABILITIES};
    fn errorbar_builtin(args: Vec<Value>) -> crate::BuiltinResult<f64> {
        futures::executor::block_on(super::errorbar_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::set::set_builtin;
    use crate::builtins::plotting::state;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
        subplot::subplot_builtin,
    };
    use runmat_plot::plots::PlotElement;
    use runmat_value::IntegerStorage;

    fn vec_tensor(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("errorbar test vector")
    }

    #[test]
    fn axes_target_without_error_data_returns_diagnostic_instead_of_panicking() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let axes = state::current_axes_state();
        let handle = state::encode_axes_handle(axes.handle, axes.active_index);
        let error = errorbar_builtin(vec![Value::Num(handle), Value::Num(2.0)])
            .expect_err("missing error data");
        assert_eq!(
            error.identifier(),
            ERRORBAR_ERROR_INVALID_ARGUMENT.identifier
        );
    }

    #[test]
    fn implicit_errorbar_axis_uses_exact_integer_storage_length() {
        let y = Tensor::new_integer(IntegerStorage::U64(vec![7, 8, 9]), vec![1, 3])
            .expect("integer y values");

        let x = infer_errorbar_x_from_y(&Value::Tensor(y)).expect("implicit x");
        let Value::Tensor(x) = x else {
            panic!("expected tensor axis");
        };
        assert_eq!(x.materialize_f64(), vec![1.0, 2.0, 3.0]);
        assert_eq!(x.shape(), &[3]);
    }

    #[test]
    fn errorbar_builds_vertical_plot() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = errorbar_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[0.1, 0.2])),
        ]);
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert!(matches!(
            fig.plots().next().unwrap(),
            PlotElement::ErrorBar(_)
        ));
    }

    #[test]
    fn errorbar_supports_axes_target_and_capsize() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let ax = subplot_builtin(Value::Num(1.0), Value::Num(2.0), Value::Num(2.0)).unwrap();
        let _ = errorbar_builtin(vec![
            Value::Num(ax),
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[3.0, 4.0])),
            Value::Tensor(vec_tensor(&[0.2, 0.3])),
            Value::String("CapSize".into()),
            Value::Num(10.0),
        ]);
        let fig = clone_figure(current_figure_handle()).unwrap();
        assert_eq!(fig.plot_axes_indices()[0], 1);
    }

    #[test]
    fn errorbar_supports_both_direction_form() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = errorbar_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[3.0, 4.0])),
            Value::Tensor(vec_tensor(&[0.2, 0.3])),
            Value::Tensor(vec_tensor(&[0.2, 0.3])),
            Value::Tensor(vec_tensor(&[0.1, 0.2])),
            Value::Tensor(vec_tensor(&[0.1, 0.2])),
        ]);
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::ErrorBar(error) = fig.plots().next().unwrap() else {
            panic!("expected errorbar");
        };
        assert_eq!(
            error.orientation,
            runmat_plot::plots::errorbar::ErrorBarOrientation::Both
        );
        assert_eq!(error.x_neg, vec![0.1, 0.2]);
        assert_eq!(error.y_pos, vec![0.2, 0.3]);
    }

    #[test]
    fn errorbar_accepts_line_spec_before_name_value_pairs() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        errorbar_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[3.0, 4.0])),
            Value::Tensor(vec_tensor(&[0.2, 0.3])),
            Value::Tensor(vec_tensor(&[0.4, 0.5])),
            Value::String("o-".into()),
            Value::String("LineWidth".into()),
            Value::Num(1.5),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::ErrorBar(error) = fig.plots().next().unwrap() else {
            panic!("expected errorbar");
        };
        assert_eq!(
            error.orientation,
            runmat_plot::plots::errorbar::ErrorBarOrientation::Vertical
        );
        assert_eq!(error.line_style, runmat_plot::plots::LineStyle::Solid);
        assert_eq!(error.line_width, 1.5);
        let marker = error.marker.as_ref().expect("expected marker");
        assert_eq!(
            marker.kind,
            runmat_plot::plots::scatter::MarkerStyle::Circle
        );
    }

    #[test]
    fn errorbar_accepts_name_value_pairs_after_asymmetric_vertical_data() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        errorbar_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[3.0, 4.0])),
            Value::Tensor(vec_tensor(&[0.2, 0.3])),
            Value::Tensor(vec_tensor(&[0.4, 0.5])),
            Value::String("LineWidth".into()),
            Value::Num(1.5),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::ErrorBar(error) = fig.plots().next().unwrap() else {
            panic!("expected errorbar");
        };
        assert_eq!(
            error.orientation,
            runmat_plot::plots::errorbar::ErrorBarOrientation::Vertical
        );
        assert_eq!(error.line_width, 1.5);
    }

    #[test]
    fn errorbar_accepts_y_only_line_spec_before_name_value_pairs() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        errorbar_builtin(vec![
            Value::Tensor(vec_tensor(&[3.0, 4.0])),
            Value::Tensor(vec_tensor(&[0.2, 0.3])),
            Value::String("o-".into()),
            Value::String("LineWidth".into()),
            Value::Num(1.5),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::ErrorBar(error) = fig.plots().next().unwrap() else {
            panic!("expected errorbar");
        };
        assert_eq!(error.x, vec![1.0, 2.0]);
        assert_eq!(error.y, vec![3.0, 4.0]);
        assert_eq!(error.y_neg, vec![0.2, 0.3]);
        assert_eq!(error.y_pos, vec![0.2, 0.3]);
        assert_eq!(error.line_width, 1.5);
        let marker = error.marker.as_ref().expect("expected marker");
        assert_eq!(
            marker.kind,
            runmat_plot::plots::scatter::MarkerStyle::Circle
        );
    }

    #[test]
    fn errorbar_preserves_explicit_x_y_err_with_trailing_line_spec() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        errorbar_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[3.0, 4.0])),
            Value::Tensor(vec_tensor(&[0.2, 0.3])),
            Value::String("o-".into()),
            Value::String("LineWidth".into()),
            Value::Num(1.5),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::ErrorBar(error) = fig.plots().next().unwrap() else {
            panic!("expected errorbar");
        };
        assert_eq!(
            error.orientation,
            runmat_plot::plots::errorbar::ErrorBarOrientation::Vertical
        );
        assert_eq!(error.x, vec![1.0, 2.0]);
        assert_eq!(error.y, vec![3.0, 4.0]);
        assert_eq!(error.y_neg, vec![0.2, 0.3]);
        assert_eq!(error.y_pos, vec![0.2, 0.3]);
        assert_eq!(error.line_width, 1.5);
        let marker = error.marker.as_ref().expect("expected marker");
        assert_eq!(
            marker.kind,
            runmat_plot::plots::scatter::MarkerStyle::Circle
        );
    }

    #[test]
    fn errorbar_preserves_both_direction_form_with_trailing_style() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        errorbar_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[3.0, 4.0])),
            Value::Tensor(vec_tensor(&[0.2, 0.3])),
            Value::Tensor(vec_tensor(&[0.4, 0.5])),
            Value::Tensor(vec_tensor(&[0.1, 0.2])),
            Value::Tensor(vec_tensor(&[0.2, 0.3])),
            Value::String("o-".into()),
            Value::String("LineWidth".into()),
            Value::Num(1.5),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::ErrorBar(error) = fig.plots().next().unwrap() else {
            panic!("expected errorbar");
        };
        assert_eq!(
            error.orientation,
            runmat_plot::plots::errorbar::ErrorBarOrientation::Both
        );
        assert_eq!(error.x_neg, vec![0.1, 0.2]);
        assert_eq!(error.x_pos, vec![0.2, 0.3]);
        assert_eq!(error.line_width, 1.5);
        let marker = error.marker.as_ref().expect("expected marker");
        assert_eq!(
            marker.kind,
            runmat_plot::plots::scatter::MarkerStyle::Circle
        );
    }

    #[test]
    fn errorbar_handle_exposes_runtime_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = errorbar_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[0.1, 0.2])),
        ])
        .unwrap();
        let props = get_builtin(vec![Value::Num(handle)]).unwrap();
        let Value::Struct(st) = props else {
            panic!("expected struct");
        };
        assert_eq!(
            st.fields.get("Type"),
            Some(&Value::String("errorbar".into()))
        );
        assert!(st.fields.contains_key("CapSize"));
    }

    #[test]
    fn errorbar_handle_set_updates_properties() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let handle = errorbar_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[0.1, 0.2])),
        ])
        .unwrap();
        set_builtin(vec![
            Value::Num(handle),
            Value::String("CapSize".into()),
            Value::Num(12.0),
        ])
        .unwrap();
        let cap = get_builtin(vec![Value::Num(handle), Value::String("CapSize".into())]).unwrap();
        assert_eq!(cap, Value::Num(12.0));
    }

    #[test]
    fn errorbar_accepts_scalar_point() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _ = errorbar_builtin(vec![Value::Num(1.0), Value::Num(2.0), Value::Num(0.3)]).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::ErrorBar(plot) = fig.plots().next().unwrap() else {
            panic!("expected errorbar")
        };
        assert_eq!(plot.x, vec![1.0]);
        assert_eq!(plot.y, vec![2.0]);
    }

    #[test]
    fn errorbar_descriptor_includes_core_signatures() {
        let labels: Vec<&str> = ERRORBAR_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = errorbar(Y, E)"));
        assert!(labels.contains(&"h = errorbar(X, Y, E)"));
        assert!(labels.contains(&"h = errorbar(X, Y, YNeg, YPos, XNeg, XPos)"));
    }

    #[test]
    fn errorbar_missing_data_uses_stable_identifier() {
        let err = errorbar_builtin(vec![Value::Num(1.0)])
            .expect_err("expected errorbar argument validation error");
        assert_eq!(err.identifier(), ERRORBAR_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn errorbar_supports_documented_orientation_tokens() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        errorbar_builtin(vec![
            Value::Tensor(vec_tensor(&[1.0, 2.0])),
            Value::Tensor(vec_tensor(&[3.0, 4.0])),
            Value::Tensor(vec_tensor(&[0.2, 0.3])),
            Value::String("horizontal".into()),
        ])
        .unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::ErrorBar(error) = fig.plots().next().unwrap() else {
            panic!("expected errorbar");
        };
        assert_eq!(
            error.orientation,
            runmat_plot::plots::errorbar::ErrorBarOrientation::Horizontal
        );
        assert_eq!(error.x_neg, vec![0.2, 0.3]);
        assert_eq!(error.y_neg, vec![0.0, 0.0]);
    }

    #[test]
    fn errorbar_matrix_inputs_create_one_series_per_column() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let y = Tensor::new(vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0], vec![2, 3]).unwrap();
        let err = Tensor::new(vec![0.1; 6], vec![2, 3]).unwrap();
        errorbar_builtin(vec![Value::Tensor(y), Value::Tensor(err)]).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let plots: Vec<_> = fig.plots().collect();
        assert_eq!(plots.len(), 3);
        let PlotElement::ErrorBar(first) = plots[0] else {
            panic!("expected errorbar");
        };
        assert_eq!(first.x, vec![1.0, 2.0]);
        assert_eq!(first.y, vec![1.0, 2.0]);
    }

    #[test]
    fn errorbar_accepts_logical_arrays_and_defaults_to_no_marker() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let y = runmat_value::LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        let err = runmat_value::LogicalArray::new(vec![1, 1], vec![1, 2]).unwrap();
        errorbar_builtin(vec![Value::LogicalArray(y), Value::LogicalArray(err)]).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::ErrorBar(error) = fig.plots().next().unwrap() else {
            panic!("expected errorbar");
        };
        assert_eq!(error.y, vec![0.0, 1.0]);
        assert!(error.marker.is_none());
    }

    fn assert_invalid_errorbar_calls_preserve_state(assert_unchanged: impl Fn()) {
        let error = errorbar_builtin(vec![Value::Complex(1.0, 1.0), Value::Num(0.2)])
            .expect_err("complex data");
        assert_eq!(
            error.identifier(),
            ERRORBAR_ERROR_INVALID_ARGUMENT.identifier
        );
        assert_unchanged();

        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error =
            errorbar_builtin(vec![Value::Num(1.0), Value::Num(0.2)]).expect_err("excess outputs");
        assert_eq!(
            error.identifier(),
            runmat_builtins::COORDINATE_ERROR_TOO_MANY_OUTPUTS.identifier
        );
        assert_unchanged();
    }

    #[test]
    fn errorbar_rejects_complex_and_excess_outputs_without_creating_a_figure() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        state::reset_plot_state();
        assert!(state::figure_handles().is_empty());
        assert!(state::current_figure_handle_if_exists().is_none());
        assert_invalid_errorbar_calls_preserve_state(|| {
            assert!(state::figure_handles().is_empty());
            assert!(state::current_figure_handle_if_exists().is_none());
        });
    }

    #[test]
    fn errorbar_rejects_complex_and_excess_outputs_without_mutating_an_existing_figure() {
        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        state::reset_plot_state();
        errorbar_builtin(vec![Value::Num(1.0), Value::Num(0.2)])
            .expect("a valid errorbar creates the sentinel plot");
        let handle = current_figure_handle();
        let handles = state::figure_handles();
        let revision = state::current_figure_revision(handle);
        assert_eq!(clone_figure(handle).unwrap().plots().count(), 1);
        assert!(revision.is_some());
        assert_invalid_errorbar_calls_preserve_state(|| {
            assert_eq!(state::figure_handles(), handles);
            assert_eq!(state::current_figure_handle_if_exists(), Some(handle));
            assert_eq!(state::current_figure_revision(handle), revision);
            assert_eq!(clone_figure(handle).unwrap().plots().count(), 1);
        });
    }

    #[test]
    fn errorbar_integer_capabilities_cover_every_documented_role() {
        assert_eq!(ERRORBAR_INTEGER_CAPABILITIES.len(), 4);
        for capability in ERRORBAR_INTEGER_CAPABILITIES {
            for input in capability.inputs {
                assert_eq!(
                    input.classes,
                    crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES
                );
                assert_eq!(
                    input.availability,
                    BuiltinIntegerInputAvailability::Documented
                );
            }
        }
    }

    #[test]
    fn errorbar_host_boundary_accepts_all_integer_classes() {
        let storages = vec![
            IntegerStorage::I8(vec![1, 2]),
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I32(vec![1, 2]),
            IntegerStorage::I64(vec![1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![1, 2]),
        ];
        let parsed = parse_errorbar_style_args(&[]).unwrap();
        for storage in storages {
            let y = Tensor::new_integer(storage, vec![1, 2]).unwrap();
            let plots = build_errorbar_host_plots(
                vec_tensor(&[1.0, 2.0]),
                y,
                None,
                None,
                vec_tensor(&[1.0, 1.0]),
                vec_tensor(&[1.0, 1.0]),
                &parsed,
                "Data",
            )
            .unwrap();
            assert_eq!(plots[0].y, vec![1.0, 2.0]);
        }
    }

    #[test]
    fn errorbar_resident_integer_never_enters_floating_gpu_geometry() {
        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let make = |element| {
            NumericInput::Gpu(
                owner
                    .allocate_zeroed(
                        vec![1, 2],
                        element,
                        runmat_accelerate_api::GpuTensorStorage::Real,
                    )
                    .unwrap()
                    .handle,
            )
        };
        let x = make(runmat_accelerate_api::NumericElementType::F64);
        let y = make(runmat_accelerate_api::NumericElementType::U64);
        let yn = make(runmat_accelerate_api::NumericElementType::F64);
        let yp = make(runmat_accelerate_api::NumericElementType::F64);
        assert!(!errorbar_gpu_inputs_eligible(&x, &y, None, None, &yn, &yp));
    }
}
