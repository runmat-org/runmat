//! MATLAB-compatible `scatter` builtin.

use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{SCATTER_ERROR_INTERNAL, SCATTER_ERROR_INVALID_ARGUMENT};

use glam::{Vec3, Vec4};
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_macros::runtime_builtin;
use runmat_plot::core::BoundingBox;
use runmat_plot::gpu::scatter2::{
    Scatter2GpuInputs, Scatter2GpuParams, ScatterAttributeBuffer, ScatterColorBuffer,
    ScatterColorLayout, ScatterSizeUnit,
};
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::scatter::{MarkerStyle, ScatterGpuStyle};
use runmat_plot::plots::surface::ColorMap;
use runmat_plot::plots::LineStyle;
use runmat_plot::plots::NumericPlotData;
use runmat_plot::plots::ScatterPlot;
#[cfg(test)]
use runmat_value::Tensor;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use std::convert::TryFrom;

use super::common::{gather_tensor_from_gpu, numeric_plot_data_pair};
use super::op_common::line_inputs::NumericInput as ScatterInput;
use super::op_common::{apply_axes_target, split_leading_axes_arguments};
use super::perf::scatter_target_points;
use super::plotting_error;
use super::point::{
    convert_rgb_color_matrix, convert_rgb_color_matrix_async, convert_scalar_color_values_async,
    convert_size_vector, convert_size_vector_async, default_marker_diameter_px,
    map_scalar_values_to_colors, marker_area_points2_to_diameter_px, validate_gpu_color_matrix,
    validate_gpu_vector_length, PointArgs, PointColorArg, PointGpuColor, PointSizeArg,
};
use super::state::{render_active_plot, PlotRenderOptions};
use super::style::{LineStyleParseOptions, MarkerColor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::scatter")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "scatter",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    // Plotting is a sink, but can consume gpuArray inputs zero-copy when a shared WGPU context exists.
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "2-D scatter rendering terminates fusion graphs; gpuArray inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::scatter")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "scatter",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "scatter performs I/O and terminates fusion graphs.",
};

const BUILTIN_NAME: &str = "scatter";

fn scatter_error_with_detail(
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

fn map_scatter_invalid_argument(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    scatter_error_with_detail(&SCATTER_ERROR_INVALID_ARGUMENT, err.message)
}

fn map_scatter_internal(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        return err;
    }
    scatter_error_with_detail(&SCATTER_ERROR_INTERNAL, err.message)
}

#[runtime_builtin(
    name = "scatter",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::scatter"
)]
pub async fn scatter_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<f64> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::SCATTER_DESCRIPTOR,
        &runmat_builtins::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
        "scatter",
    )?;
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "scatter")?;
    // Keep the original rooted owner alive even when helpers extract native data.
    let _arguments = args.fork();
    work.finish(scatter_impl(args, &work).await)
}

async fn scatter_impl(
    args: crate::call::BuiltinArguments,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<f64> {
    let (axes_target, mut args) =
        split_leading_axes_arguments(args, BUILTIN_NAME).map_err(map_scatter_invalid_argument)?;
    if args.len() < 2 {
        return Err(scatter_error_with_detail(
            &SCATTER_ERROR_INVALID_ARGUMENT,
            "expected X and Y data after axes handle",
        ));
    }
    let x = args.pop_front().expect("checked coordinate count");
    let y = args.pop_front().expect("checked coordinate count");
    let rest = args;
    let style_args = PointArgs::parse_arguments(rest.view(), LineStyleParseOptions::scatter())
        .map_err(map_scatter_invalid_argument)?;
    let x_input =
        ScatterInput::from_value(x, BUILTIN_NAME).map_err(map_scatter_invalid_argument)?;
    let y_input =
        ScatterInput::from_value(y, BUILTIN_NAME).map_err(map_scatter_invalid_argument)?;
    let point_count = x_input.len();
    let mut resolved_style = resolve_scatter_style_async(point_count, &style_args, "scatter")
        .await
        .map_err(map_scatter_invalid_argument)?;
    let plot = if !resolved_style.requires_cpu {
        if let (Some(x_gpu), Some(y_gpu)) = (x_input.gpu_handle(), y_input.gpu_handle()) {
            build_scatter_gpu_plot(x_gpu, y_gpu, &resolved_style)?
        } else {
            None
        }
    } else {
        None
    };
    let mut plot = match plot {
        Some(plot) => plot,
        None => {
            let x_tensor = x_input
                .into_tensor_async("scatter")
                .await
                .map_err(map_scatter_invalid_argument)?;
            let y_tensor = y_input
                .into_tensor_async("scatter")
                .await
                .map_err(map_scatter_invalid_argument)?;
            ensure_host_marker_metadata_async(&mut resolved_style, point_count)
                .await
                .map_err(map_scatter_invalid_argument)?;
            let (x_data, y_data) = numeric_plot_data_pair(x_tensor, y_tensor, "scatter")
                .map_err(map_scatter_invalid_argument)?;
            build_scatter_plot_numeric(x_data, y_data, &mut resolved_style)
                .map_err(map_scatter_invalid_argument)?
        }
    };
    if let Some(visible) = style_args.style.appearance.visible {
        plot.set_visible(visible);
    }
    let opts = PlotRenderOptions::default();
    work.check()?;
    apply_axes_target(axes_target, BUILTIN_NAME).map_err(map_scatter_invalid_argument)?;
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let mut plot = Some(plot);
    work.check()?;
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let plot_index = figure.add_scatter_plot_on_axes(
            plot.take().expect("scatter plot consumed exactly once"),
            axes,
        );
        *plot_index_slot.borrow_mut() = Some((axes, plot_index));
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_scatter_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        if crate::builtins::plotting::state::presentation_unavailable(&err) {
            return Ok(handle);
        }
        return Err(map_scatter_internal(err));
    }
    Ok(handle)
}

fn build_scatter_plot(
    x: Vec<f64>,
    y: Vec<f64>,
    style: &mut ScatterResolvedStyle,
) -> BuiltinResult<ScatterPlot> {
    if x.len() != y.len() {
        return Err(scatter_err(
            "scatter: X and Y inputs must share the same length",
        ));
    }
    if x.is_empty() {
        return Err(scatter_err("scatter: inputs cannot be empty"));
    }

    ensure_host_marker_metadata(style, x.len())?;

    let mut scatter = ScatterPlot::new(x, y)
        .map_err(|err| scatter_err(format!("scatter: {err}")))?
        .with_style(style.uniform_color, style.marker_size, style.marker_style)
        .with_label(style.label.clone());
    scatter.colormap = style.colormap.clone();
    scatter.set_edge_color(style.edge_color);
    scatter.set_edge_thickness(style.edge_thickness);
    scatter.set_filled(style.filled);
    scatter.set_edge_color_from_vertex(style.marker_edge_flat);

    if let Some(sizes) = style.per_point_sizes.take() {
        scatter.set_sizes(sizes);
    }
    if let Some(colors) = style.per_point_colors.take() {
        scatter.set_colors(colors);
    }
    if let Some(values) = style.color_values.take() {
        scatter.set_color_values(values, style.color_limits.take());
    }
    Ok(scatter)
}

fn build_scatter_plot_numeric(
    x: NumericPlotData,
    y: NumericPlotData,
    style: &mut ScatterResolvedStyle,
) -> BuiltinResult<ScatterPlot> {
    if x.len() != y.len() {
        return Err(scatter_err(
            "scatter: X and Y inputs must share the same length",
        ));
    }
    if x.is_empty() {
        return Err(scatter_err("scatter: inputs cannot be empty"));
    }

    ensure_host_marker_metadata(style, x.len())?;

    let mut scatter = ScatterPlot::from_numeric_data(x, y)
        .map_err(|err| scatter_err(format!("scatter: {err}")))?
        .with_style(style.uniform_color, style.marker_size, style.marker_style)
        .with_label(style.label.clone());
    scatter.colormap = style.colormap.clone();
    scatter.set_edge_color(style.edge_color);
    scatter.set_edge_thickness(style.edge_thickness);
    scatter.set_filled(style.filled);
    scatter.set_edge_color_from_vertex(style.marker_edge_flat);

    if let Some(sizes) = style.per_point_sizes.take() {
        scatter.set_sizes(sizes);
    }
    if let Some(colors) = style.per_point_colors.take() {
        scatter.set_colors(colors);
    }
    if let Some(values) = style.color_values.take() {
        scatter.set_color_values(values, style.color_limits.take());
    }
    Ok(scatter)
}

pub(super) async fn build_scatter_plot_from_args_async(
    x: Vec<f64>,
    y: Vec<f64>,
    rest: &[Value],
    context: &'static str,
) -> BuiltinResult<ScatterPlot> {
    let style_args = PointArgs::parse(
        rest.to_vec(),
        LineStyleParseOptions {
            builtin_name: context,
            forbid_leading_numeric: true,
            forbid_interleaved_numeric: true,
            accepts_handle_visibility: false,
        },
    )?;
    let mut resolved_style = resolve_scatter_style_async(x.len(), &style_args, context).await?;
    build_scatter_plot(x, y, &mut resolved_style)
}

fn scatter_err(message: impl Into<String>) -> RuntimeError {
    plotting_error(BUILTIN_NAME, message)
}

const DEFAULT_LINE_WIDTH: f32 = 1.0;
const DEFAULT_SCATTER_LABEL: &str = "Data";

fn default_color() -> Vec4 {
    Vec4::new(0.85, 0.2, 0.2, 0.95)
}

#[derive(Clone, Debug)]
struct ScatterResolvedStyle {
    uniform_color: Vec4,
    edge_color: Vec4,
    edge_thickness: f32,
    marker_style: MarkerStyle,
    marker_size: f32,
    filled: bool,
    per_point_sizes: Option<Vec<f32>>,
    per_point_colors: Option<Vec<Vec4>>,
    color_values: Option<Vec<f64>>,
    color_limits: Option<(f64, f64)>,
    gpu_sizes: Option<GpuTensorHandle>,
    gpu_colors: Option<PointGpuColor>,
    colormap: ColorMap,
    marker_face_flat: bool,
    marker_edge_flat: bool,
    requires_cpu: bool,
    label: String,
}

fn base_scatter_style(args: &PointArgs) -> ScatterResolvedStyle {
    let mut style = ScatterResolvedStyle {
        uniform_color: default_color(),
        edge_color: default_color(),
        edge_thickness: DEFAULT_LINE_WIDTH,
        marker_style: MarkerStyle::Circle,
        marker_size: default_marker_diameter_px(),
        filled: args.filled,
        per_point_sizes: None,
        per_point_colors: None,
        color_values: None,
        color_limits: None,
        gpu_sizes: None,
        gpu_colors: None,
        colormap: ColorMap::Parula,
        marker_face_flat: false,
        marker_edge_flat: false,
        requires_cpu: false,
        label: DEFAULT_SCATTER_LABEL.to_string(),
    };
    if let Some(label) = args.style.label.clone() {
        style.label = label;
    }
    style
}

fn apply_scatter_line_appearance(
    style: &mut ScatterResolvedStyle,
    args: &PointArgs,
) -> BuiltinResult<()> {
    let appearance = &args.style.appearance;
    style.uniform_color = appearance.color;
    style.edge_color = appearance.color;
    style.edge_thickness = appearance.line_width.max(0.1);

    if let PointColorArg::Uniform(color) = &args.color {
        style.uniform_color = *color;
        style.edge_color = *color;
    }

    if appearance.marker.is_none() {
        style.edge_color = style.uniform_color;
    }

    if let Some(marker) = appearance.marker.as_ref() {
        style.marker_style = marker.kind.to_plot_marker();
        if let Some(size) = marker.size {
            style.marker_size = size.max(0.1);
        }
        if matches!(marker.edge_color, MarkerColor::Flat) {
            style.marker_edge_flat = true;
        } else {
            style.edge_color =
                resolve_marker_color(&marker.edge_color, style.edge_color, style.uniform_color);
        }
        match &marker.face_color {
            MarkerColor::Flat => {
                style.marker_face_flat = true;
                style.filled = true;
            }
            MarkerColor::None => {
                style.filled = false;
            }
            _ => {
                let face_color = resolve_marker_color(
                    &marker.face_color,
                    style.uniform_color,
                    style.uniform_color,
                );
                if matches!(marker.face_color, MarkerColor::Color(_) | MarkerColor::Auto) {
                    style.uniform_color = face_color;
                }
            }
        }
    }

    if let PointSizeArg::Scalar(size) = &args.size {
        style.marker_size = marker_area_points2_to_diameter_px(*size as f64);
    }
    Ok(())
}

fn finalize_scatter_style(
    style: &mut ScatterResolvedStyle,
    args: &PointArgs,
    context: &'static str,
) -> BuiltinResult<()> {
    // MATLAB leaves markers unfilled unless 'filled' or a MarkerFaceColor asks; with color
    // data, the edge takes each point's color ('flat') unless MarkerEdgeColor is given.
    let edge_color_given = args
        .style
        .appearance
        .marker
        .as_ref()
        .is_some_and(|marker| !matches!(marker.edge_color, MarkerColor::Auto));
    if (style.per_point_colors.is_some() || style.gpu_colors.is_some()) && !edge_color_given {
        style.marker_edge_flat = true;
    }

    if style.marker_face_flat {
        if style.per_point_colors.is_none() && style.gpu_colors.is_none() {
            return Err(scatter_err(format!(
                "{context}: MarkerFaceColor 'flat' requires per-point color data (C argument)"
            )));
        }
        style.filled = true;
    }

    if style.marker_edge_flat && style.per_point_colors.is_none() && style.gpu_colors.is_none() {
        return Err(scatter_err(format!(
            "{context}: MarkerEdgeColor 'flat' requires per-point color data (C argument)"
        )));
    }

    if args.style.line_style_explicit
        && !matches!(
            args.style.appearance.line_style,
            LineStyle::Solid | LineStyle::None
        )
    {
        style.requires_cpu = true;
    }
    style.requires_cpu |= args.style.requires_cpu_fallback;
    if args.style.line_style_order.is_some() {
        style.requires_cpu = true;
    }
    Ok(())
}

#[cfg(test)]
fn resolve_scatter_style(
    point_count: usize,
    args: &PointArgs,
    context: &'static str,
) -> BuiltinResult<ScatterResolvedStyle> {
    futures::executor::block_on(resolve_scatter_style_async(point_count, args, context))
}

async fn resolve_scatter_style_async(
    point_count: usize,
    args: &PointArgs,
    context: &'static str,
) -> BuiltinResult<ScatterResolvedStyle> {
    let mut style = base_scatter_style(args);
    apply_scatter_line_appearance(&mut style, args)?;

    if let Some(value) = args.size.value() {
        match value {
            Value::GpuTensor(handle) => {
                validate_gpu_vector_length(handle, point_count, context)?;
                style.gpu_sizes = Some(handle.clone());
            }
            _ => {
                style.per_point_sizes =
                    Some(convert_size_vector_async(value, point_count, context).await?);
            }
        }
    }

    match &args.color {
        PointColorArg::ScalarValues(value) => {
            let scalars = convert_scalar_color_values_async(value, point_count, context).await?;
            let (colors, limits) = map_scalar_values_to_colors(&scalars, style.colormap.clone());
            style.color_values = Some(scalars);
            style.per_point_colors = Some(colors);
            style.color_limits = Some(limits);
        }
        PointColorArg::RgbMatrix(value) => match value {
            Value::GpuTensor(handle) => {
                let components = validate_gpu_color_matrix(handle, point_count, context)?;
                style.gpu_colors = Some(PointGpuColor {
                    handle: handle.clone(),
                    components,
                });
            }
            _ => {
                style.per_point_colors =
                    Some(convert_rgb_color_matrix_async(value, point_count, context).await?);
            }
        },
        _ => {}
    }

    finalize_scatter_style(&mut style, args, context)?;
    Ok(style)
}

fn resolve_marker_color(marker_color: &MarkerColor, fallback: Vec4, default_base: Vec4) -> Vec4 {
    match marker_color {
        MarkerColor::Auto => fallback,
        MarkerColor::None => Vec4::new(default_base.x, default_base.y, default_base.z, 0.0),
        MarkerColor::Flat => fallback,
        MarkerColor::Color(color) => *color,
    }
}

fn build_scatter_gpu_plot(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    style: &ScatterResolvedStyle,
) -> BuiltinResult<Option<ScatterPlot>> {
    let count = super::gpu_helpers::GpuBufferExports::element_count;
    let point_count = count(x, BUILTIN_NAME)?;
    if point_count == 0 {
        return Err(scatter_err("scatter: empty input tensor"));
    }
    u32::try_from(point_count)
        .map_err(|_| scatter_err("scatter: point count exceeds supported range"))?;
    if count(y, BUILTIN_NAME)? != point_count {
        return Err(scatter_err(
            "scatter: X and Y inputs must have identical lengths",
        ));
    }
    if runmat_accelerate_api::handle_precision(x) != runmat_accelerate_api::handle_precision(y) {
        return Ok(None);
    }
    if let Some(input) = &style.gpu_sizes {
        if count(input, BUILTIN_NAME)? != point_count {
            return Err(scatter_err(format!(
                "scatter: marker size array must have {point_count} elements"
            )));
        }
        if runmat_accelerate_api::handle_precision(input) != Some(ProviderPrecision::F32) {
            return Ok(None);
        }
    }
    if let Some(color) = &style.gpu_colors {
        let expected = point_count
            .checked_mul(color.components.stride() as usize)
            .ok_or_else(|| scatter_err("scatter: color element count overflow"))?;
        if count(&color.handle, BUILTIN_NAME)? != expected {
            return Err(scatter_err(format!(
                "scatter: color array must contain {expected} elements"
            )));
        }
        if runmat_accelerate_api::handle_precision(&color.handle) != Some(ProviderPrecision::F32) {
            return Ok(None);
        }
    }
    if runmat_accelerate_api::handle_integer_type(x).is_some()
        || runmat_accelerate_api::handle_integer_type(y).is_some()
    {
        return Ok(None);
    }
    let mut inputs = vec![x, y];
    inputs.extend(style.gpu_sizes.iter());
    inputs.extend(style.gpu_colors.iter().map(|color| &color.handle));
    let Some((context, mut gpu_exports)) =
        super::gpu_helpers::GpuBufferExports::prepare(&inputs, BUILTIN_NAME)?
    else {
        return Ok(None);
    };
    let result = (|| {
        let x_ref = gpu_exports
            .export(x, BUILTIN_NAME)?
            .ok_or_else(|| scatter_err("scatter: unable to export GPU X data"))?;
        let y_ref = gpu_exports
            .export(y, BUILTIN_NAME)?
            .ok_or_else(|| scatter_err("scatter: unable to export GPU Y data"))?;

        if x_ref.len == 0 {
            return Err(scatter_err("scatter: empty input tensor"));
        }
        if x_ref.len != y_ref.len {
            return Err(scatter_err(
                "scatter: X and Y inputs must have identical lengths",
            ));
        }
        if x_ref.precision != y_ref.precision {
            return Err(scatter_err(
                "scatter: X and Y gpuArrays must have matching precision",
            ));
        }
        let point_count = x_ref.len;
        let len_u32 = u32::try_from(point_count)
            .map_err(|_| scatter_err("scatter: point count exceeds supported range"))?;
        let scalar = ScalarType::from_is_f64(x_ref.precision == ProviderPrecision::F64);
        let lod_stride = scatter_lod_stride(len_u32);

        let size_buffer = build_size_buffer(style, point_count, &mut gpu_exports)?;
        let has_sizes = size_buffer.has_data();
        let color_buffer = build_color_buffer(style, point_count, &mut gpu_exports)?;
        let has_colors = color_buffer.has_data();
        let color_layout = ScatterColorLayout::from_source(&color_buffer);

        let inputs = Scatter2GpuInputs {
            x_buffer: x_ref.clone().into(),
            y_buffer: y_ref.clone().into(),
            len: len_u32,
            scalar,
        };
        let params = Scatter2GpuParams {
            color: style.uniform_color,
            point_size: style.marker_size,
            size_unit: ScatterSizeUnit::from_source(&size_buffer),
            sizes: size_buffer,
            colors: color_buffer,
            color_layout,
            lod_stride,
        };

        let gpu_vertices = context
            .access
            .with_shared(|emission| {
                runmat_plot::gpu::scatter2::pack_vertices_from_xy(emission, &inputs, &params)
            })
            .map_err(|source| {
                super::plotting_error_with_source(
                    BUILTIN_NAME,
                    format!("{}: GPU emission unavailable: {source}", BUILTIN_NAME),
                    source,
                )
            })?
            .map_err(|source| {
                super::plotting_error_with_source(
                    BUILTIN_NAME,
                    format!("scatter: failed to build GPU vertices: {source}"),
                    source,
                )
            })?;
        gpu_exports.record_completed_effect();
        let drawn_points = gpu_vertices.vertex_count;

        let bounds = build_gpu_bounds(x, y, &gpu_exports.protected_handles())?;
        let gpu_style = ScatterGpuStyle {
            color: style.uniform_color,
            edge_color: style.edge_color,
            edge_thickness: style.edge_thickness,
            marker_size: style.marker_size,
            marker_style: style.marker_style,
            filled: style.filled,
            has_per_point_sizes: has_sizes,
            has_per_point_colors: has_colors,
            edge_from_vertex_colors: style.marker_edge_flat,
        };

        let mut scatter =
            ScatterPlot::from_gpu_buffer(gpu_vertices, drawn_points, bounds, gpu_style)
                .with_gpu_source_inputs(inputs)
                .with_label(style.label.clone());
        scatter.colormap = style.colormap.clone();
        scatter.set_edge_color_from_vertex(style.marker_edge_flat);
        if lod_stride == 1 {
            if let Some(values) = style.color_values.as_ref() {
                scatter.color_values = Some(values.clone());
            }
        }
        scatter.color_limits = style.color_limits;
        Ok(scatter)
    })();
    gpu_exports.complete(result, BUILTIN_NAME).map(Some)
}

fn scatter_lod_stride(point_count: u32) -> u32 {
    let target = scatter_target_points().max(1);
    if point_count <= target {
        1
    } else {
        point_count.div_ceil(target)
    }
}

fn build_gpu_bounds(
    x: &GpuTensorHandle,
    y: &GpuTensorHandle,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<BoundingBox> {
    let (min_x, max_x) = futures::executor::block_on(super::gpu_helpers::axis_bounds_protecting(
        x,
        protected,
        BUILTIN_NAME,
    ))?;
    let (min_y, max_y) = futures::executor::block_on(super::gpu_helpers::axis_bounds_protecting(
        y,
        protected,
        BUILTIN_NAME,
    ))
    .map_err(|error| super::gpu_helpers::preparation_after_effect(error, BUILTIN_NAME))?;
    Ok(BoundingBox::new(
        Vec3::new(min_x, min_y, 0.0),
        Vec3::new(max_x, max_y, 0.0),
    ))
}

fn build_size_buffer(
    style: &ScatterResolvedStyle,
    point_count: usize,
    gpu_exports: &mut super::gpu_helpers::GpuBufferExports,
) -> BuiltinResult<ScatterAttributeBuffer> {
    if let Some(handle) = style.gpu_sizes.as_ref() {
        let exported = gpu_exports
            .export(handle, BUILTIN_NAME)?
            .ok_or_else(|| scatter_err("scatter: unable to export GPU marker sizes"))?;
        if exported.len != point_count {
            return Err(scatter_err(format!(
                "scatter: marker size array must have {point_count} elements (got {})",
                exported.len
            )));
        }
        if exported.precision != ProviderPrecision::F32 {
            return Err(scatter_err(
                "scatter: GPU marker sizes must be single-precision (cast before plotting)",
            ));
        }
        return Ok(ScatterAttributeBuffer::Gpu(exported.into()));
    }

    if let Some(sizes) = style.per_point_sizes.as_ref() {
        if sizes.is_empty() {
            Ok(ScatterAttributeBuffer::None)
        } else {
            Ok(ScatterAttributeBuffer::Host(sizes.clone()))
        }
    } else {
        Ok(ScatterAttributeBuffer::None)
    }
}

fn build_color_buffer(
    style: &ScatterResolvedStyle,
    point_count: usize,
    gpu_exports: &mut super::gpu_helpers::GpuBufferExports,
) -> BuiltinResult<ScatterColorBuffer> {
    if let Some(gpu_color) = style.gpu_colors.as_ref() {
        let exported = gpu_exports
            .export(&gpu_color.handle, BUILTIN_NAME)?
            .ok_or_else(|| scatter_err("scatter: unable to export GPU color data"))?;
        let expected = point_count * gpu_color.components.stride() as usize;
        if exported.len != expected {
            return Err(scatter_err(format!(
                "scatter: color array must contain {} elements (got {})",
                expected, exported.len
            )));
        }
        if exported.precision != ProviderPrecision::F32 {
            return Err(scatter_err(
                "scatter: GPU color arrays must be single-precision (cast before plotting)",
            ));
        }
        return Ok(ScatterColorBuffer::Gpu {
            buffer: exported.into(),
            components: gpu_color.components.stride(),
        });
    }

    if let Some(colors) = style.per_point_colors.as_ref() {
        if colors.is_empty() {
            Ok(ScatterColorBuffer::None)
        } else {
            let data = colors.iter().map(|c| c.to_array()).collect();
            Ok(ScatterColorBuffer::Host(data))
        }
    } else {
        Ok(ScatterColorBuffer::None)
    }
}

fn ensure_host_marker_metadata(
    style: &mut ScatterResolvedStyle,
    point_count: usize,
) -> BuiltinResult<()> {
    if style.per_point_sizes.is_none() {
        if let Some(handle) = style.gpu_sizes.clone() {
            let tensor = gather_tensor_from_gpu(handle, "scatter")?;
            let value = Value::Tensor(tensor);
            style.per_point_sizes = Some(convert_size_vector(&value, point_count, "scatter")?);
        }
    }
    if style.per_point_colors.is_none() {
        if let Some(gpu_color) = style.gpu_colors.as_ref() {
            let tensor = gather_tensor_from_gpu(gpu_color.handle.clone(), "scatter")?;
            let value = Value::Tensor(tensor);
            style.per_point_colors =
                Some(convert_rgb_color_matrix(&value, point_count, "scatter")?);
        }
    }
    Ok(())
}

async fn ensure_host_marker_metadata_async(
    style: &mut ScatterResolvedStyle,
    point_count: usize,
) -> BuiltinResult<()> {
    if style.per_point_sizes.is_none() {
        if let Some(handle) = style.gpu_sizes.clone() {
            let tensor = super::common::gather_tensor_from_gpu_async(handle, "scatter").await?;
            let value = Value::Tensor(tensor);
            style.per_point_sizes =
                Some(convert_size_vector_async(&value, point_count, "scatter").await?);
        }
    }
    if style.per_point_colors.is_none() {
        if let Some(gpu_color) = style.gpu_colors.as_ref() {
            let tensor =
                super::common::gather_tensor_from_gpu_async(gpu_color.handle.clone(), "scatter")
                    .await?;
            let value = Value::Tensor(tensor);
            style.per_point_colors =
                Some(convert_rgb_color_matrix_async(&value, point_count, "scatter").await?);
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::point::{PointArgs, PointColorArg, PointSizeArg};
    use super::super::style::{
        LineAppearance, LineStyleParseOptions, MarkerAppearance, MarkerColor, MarkerKind,
        ParsedLineStyle,
    };
    use super::*;
    use crate::builtins::plotting::state::current_axes_handle_for_figure;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, configure_subplot, current_figure_handle,
        reset_hold_state_for_run,
    };
    use crate::RuntimeError;
    use futures::executor::block_on;
    use runmat_builtins::SCATTER_DESCRIPTOR;
    use runmat_plot::plots::PlotElement;
    use runmat_value::Value;

    fn setup_plot_tests() {
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
    }

    fn scatter_builtin(x: Value, y: Value, rest: Vec<Value>) -> BuiltinResult<f64> {
        block_on(super::scatter_builtin(None, {
            let mut args = vec![x, y];
            args.extend(rest);
            crate::builtins::common::test_support::builtin_arguments(args)
        }))
    }

    fn test_style() -> ScatterResolvedStyle {
        ScatterResolvedStyle {
            uniform_color: default_color(),
            edge_color: default_color(),
            edge_thickness: DEFAULT_LINE_WIDTH,
            marker_style: MarkerStyle::Circle,
            marker_size: default_marker_diameter_px(),
            filled: false,
            per_point_sizes: None,
            per_point_colors: None,
            color_values: None,
            color_limits: None,
            gpu_sizes: None,
            gpu_colors: None,
            colormap: ColorMap::Parula,
            marker_face_flat: false,
            marker_edge_flat: false,
            requires_cpu: false,
            label: DEFAULT_SCATTER_LABEL.to_string(),
        }
    }

    fn tensor_from(data: &[f64]) -> Tensor {
        Tensor::new(data.to_vec(), vec![data.len()]).expect("scatter test vector")
    }

    fn assert_plotting_unavailable(err: &RuntimeError) {
        assert!(
            crate::builtins::plotting::state::presentation_unavailable(err),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scatter_requires_equal_lengths() {
        setup_plot_tests();
        assert!(build_scatter_plot(vec![1.0], vec![], &mut test_style()).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scatter_builtin_returns_or_reports_backend_status() {
        setup_plot_tests();
        let out = scatter_builtin(
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Vec::new(),
        );
        if let Err(flow) = out {
            assert_plotting_unavailable(&flow);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scatter_resolves_marker_style_from_arguments() {
        setup_plot_tests();
        let rest = vec![
            Value::Num(12.0),
            Value::String("r".into()),
            Value::String("filled".into()),
            Value::String("Marker".into()),
            Value::String("s".into()),
        ];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter()).unwrap();
        let style = resolve_scatter_style(3, &args, "scatter").expect("style");
        assert!(style.filled);
        assert_eq!(style.marker_style, MarkerStyle::Square);
        assert!(
            (style.marker_size - marker_area_points2_to_diameter_px(12.0)).abs() < 1e-5,
            "size was {}",
            style.marker_size
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scatter_single_rgb_row_sets_uniform_face_and_edge_color() {
        setup_plot_tests();
        let rest = vec![
            Value::Num(49.0),
            Value::Tensor(Tensor::new(vec![0.9, 0.2, 0.2], vec![1, 3]).expect("RGB row")),
            Value::String("filled".into()),
            Value::String("Marker".into()),
            Value::String("o".into()),
        ];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter()).unwrap();
        let style = resolve_scatter_style(4, &args, "scatter").expect("style");
        assert!((style.uniform_color.x - 0.9).abs() < 1e-6);
        assert!((style.uniform_color.y - 0.2).abs() < 1e-6);
        assert!((style.uniform_color.z - 0.2).abs() < 1e-6);
        assert!((style.edge_color.x - 0.9).abs() < 1e-6);
        assert!((style.edge_color.y - 0.2).abs() < 1e-6);
        assert!((style.edge_color.z - 0.2).abs() < 1e-6);
        assert!(!style.marker_edge_flat);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scatter_supports_flat_marker_face_color() {
        setup_plot_tests();
        let rest = vec![
            Value::Tensor(tensor_from(&[5.0, 5.0])),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            Value::String("Marker".into()),
            Value::String("o".into()),
            Value::String("MarkerFaceColor".into()),
            Value::String("flat".into()),
        ];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter()).unwrap();
        let style = resolve_scatter_style(2, &args, "scatter").expect("style");
        assert!(style.marker_face_flat);
        assert!(style.per_point_colors.is_some() || style.gpu_colors.is_some());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scatter_applies_display_name() {
        setup_plot_tests();
        let rest = vec![
            Value::String("DisplayName".into()),
            Value::String("Series A".into()),
        ];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter()).unwrap();
        let mut style = resolve_scatter_style(2, &args, "scatter").expect("style");
        let plot = build_scatter_plot(vec![0.0, 1.0], vec![0.0, 1.0], &mut style).expect("plot");
        assert_eq!(plot.label.as_deref(), Some("Series A"));
    }

    #[test]
    fn scatter_accepts_leading_axes_handle() {
        let _guard = crate::builtins::plotting::tests::lock_plot_registry();
        setup_plot_tests();
        configure_subplot(1, 2, 1).unwrap();
        let fig_handle = current_figure_handle();
        let ax = current_axes_handle_for_figure(fig_handle).unwrap();
        let _ = scatter_builtin(
            Value::Num(ax),
            Value::Tensor(tensor_from(&[0.0, 1.0])),
            vec![Value::Tensor(tensor_from(&[1.0, 2.0]))],
        );
        let fig = clone_figure(fig_handle).unwrap();
        assert_eq!(fig.plot_axes_indices(), &[1]);
    }

    #[test]
    fn scatter_accepts_scalar_point() {
        let _guard = lock_plot_registry();
        setup_plot_tests();
        let _ = scatter_builtin(Value::Num(1.0), Value::Num(2.0), Vec::new()).unwrap();
        let fig = clone_figure(current_figure_handle()).unwrap();
        let PlotElement::Scatter(plot) = fig.plots().next().unwrap() else {
            panic!("expected scatter")
        };
        let (x, y) = plot.host_xy_f64().unwrap().unwrap();
        assert_eq!(x, vec![1.0]);
        assert_eq!(y, vec![2.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scatter_rejects_flat_marker_edge_color_without_color_data() {
        setup_plot_tests();
        let rest = vec![
            Value::Tensor(tensor_from(&[5.0, 5.0])),
            Value::String("Marker".into()),
            Value::String("o".into()),
            Value::String("MarkerEdgeColor".into()),
            Value::String("flat".into()),
        ];
        let args = PointArgs::parse(rest, LineStyleParseOptions::scatter()).unwrap();
        let err = resolve_scatter_style(2, &args, "scatter").unwrap_err();
        assert!(err.to_string().contains("requires per-point color data"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scatter_accepts_flat_marker_edge_color_when_colors_supplied() {
        setup_plot_tests();
        let appearance = LineAppearance {
            marker: Some(MarkerAppearance {
                kind: MarkerKind::Circle,
                size: None,
                edge_color: MarkerColor::Flat,
                face_color: MarkerColor::Auto,
            }),
            ..Default::default()
        };
        let args = PointArgs {
            size: PointSizeArg::Default,
            color: PointColorArg::ScalarValues(Value::Tensor(tensor_from(&[0.1, 0.5]))),
            filled: false,
            style: ParsedLineStyle {
                appearance,
                requires_cpu_fallback: false,
                line_style_explicit: false,
                color_explicit: false,
                line_style_order: None,
                label: None,
            },
        };
        let mut style = resolve_scatter_style(2, &args, "scatter").expect("style");
        assert!(style.marker_edge_flat);
        let plot = build_scatter_plot(vec![0.0, 1.0], vec![0.0, 1.0], &mut style).expect("plot");
        assert!(plot.edge_color_from_vertex_colors);
    }

    #[test]
    fn scatter_catalog_infers_host_double_handle() {
        use runmat_types::*;
        let numeric = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        });
        let input = ValueFact::proven(
            numeric.clone(),
            ShapeFact::from(vec![Some(3), Some(1)]),
            StorageFact::Dense,
        );
        let request = CallRequest {
            arguments: vec![input; 2],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let result =
            runmat_builtins::infer_catalog_call(&runmat_builtins::SCATTER_CATALOG_ENTRY, &request);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs.len(), 1);
        let output = &result.outputs[0];
        assert_eq!(output.kind, numeric);
        assert_eq!(output.shape, ShapeFact::Scalar);
        assert_eq!(output.storage, StorageFact::Dense);
        assert_eq!(output.residency, ResidencyFact::Host);
    }

    #[test]
    fn scatter_descriptor_signatures_cover_supported_forms() {
        let labels: Vec<&str> = SCATTER_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = scatter(X, Y)"));
        assert!(labels.contains(&"h = scatter(X, Y, S, C)"));
        assert!(labels.contains(&"h = scatter(X, Y, Name, Value, ...)"));
        assert!(labels.contains(&"h = scatter(ax, X, Y)"));
        assert!(labels.contains(&"h = scatter(ax, X, Y, Name, Value, ...)"));
    }

    #[test]
    fn scatter_missing_post_axes_input_uses_stable_identifier() {
        let _guard = lock_plot_registry();
        setup_plot_tests();
        configure_subplot(1, 2, 1).unwrap();
        let fig_handle = current_figure_handle();
        let ax = current_axes_handle_for_figure(fig_handle).unwrap();
        let err = scatter_builtin(Value::Num(ax), Value::Num(1.0), Vec::new())
            .expect_err("missing y after axes handle should fail");
        assert_eq!(err.identifier(), SCATTER_ERROR_INVALID_ARGUMENT.identifier);
    }
}
