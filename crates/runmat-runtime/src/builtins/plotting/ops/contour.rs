//! Helpers and builtins for contour construction (CPU + GPU).

use glam::{Vec2, Vec3, Vec4};
use log::warn;
use runmat_accelerate_api::{self, GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;
use runmat_plot::core::Vertex;
use runmat_plot::gpu::contour_fill;
use runmat_plot::gpu::ScalarType;
use runmat_plot::plots::contour::contour_bounds;
use runmat_plot::plots::{ColorMap, ContourFillPlot, ContourPlot};

use super::common::{numeric_vector, tensor_to_surface_grid, value_as_f64, SurfaceDataInput};
use super::op_common::surface_inputs::{extract_meshgrid_axes_from_xy_matrices, AxisSource};
use super::style::{parse_color_value, value_as_string, LineStyleParseOptions};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

use super::gpu_helpers::axis_bounds;
use super::plotting_error;
use super::state::{render_active_plot, PlotRenderOptions};
use super::surf::build_color_lut;
use crate::builtins::plotting::type_resolvers::handle_scalar_type;

use crate::BuiltinResult;

const BUILTIN_NAME: &str = "contour";
const DEFAULT_LEVELS: usize = 10;

#[derive(Clone, Debug, Default)]
pub(crate) enum ContourLevelSpec {
    #[default]
    Auto,
    Count(usize),
    Values(Vec<f64>),
    Step(f32),
}

impl ContourLevelSpec {
    fn resolve(&self, name: &'static str, min_z: f32, max_z: f32) -> BuiltinResult<Vec<f32>> {
        match self {
            ContourLevelSpec::Auto => Ok(evenly_spaced_levels(DEFAULT_LEVELS, min_z, max_z)),
            ContourLevelSpec::Count(count) => {
                if *count == 0 {
                    Err(plotting_error(
                        name,
                        format!("{name}: level count must be positive"),
                    ))
                } else {
                    Ok(evenly_spaced_levels(*count, min_z, max_z))
                }
            }
            ContourLevelSpec::Values(values) => {
                if values.is_empty() {
                    return Err(plotting_error(
                        name,
                        format!("{name}: level vector must contain at least one value"),
                    ));
                }
                let mut last = None;
                for &value in values {
                    if !value.is_finite() {
                        return Err(plotting_error(
                            name,
                            format!("{name}: level values must be finite"),
                        ));
                    }
                    if let Some(prev) = last {
                        if value <= prev {
                            return Err(plotting_error(
                                name,
                                format!("{name}: level values must be strictly increasing"),
                            ));
                        }
                    }
                    last = Some(value);
                }
                Ok(values.iter().map(|&v| v as f32).collect())
            }
            ContourLevelSpec::Step(step) => {
                if *step <= 0.0 || !step.is_finite() {
                    return Err(plotting_error(
                        name,
                        format!("{name}: LevelStep must be a positive, finite number"),
                    ));
                }
                let mut levels = Vec::new();
                let mut value = min_z;
                let max = if max_z <= min_z { min_z + 1.0 } else { max_z };
                while value <= max {
                    levels.push(value);
                    value += *step;
                }
                if levels.is_empty() {
                    levels.push(min_z);
                } else if *levels.last().unwrap() < max {
                    levels.push(max);
                }
                Ok(levels)
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) enum ContourLineColor {
    #[default]
    Auto,
    Color(Vec4),
    None,
}

#[derive(Clone)]
pub(crate) struct ContourArgs {
    pub name: &'static str,
    pub x_axis: Vec<f64>,
    pub y_axis: Vec<f64>,
    pub z_input: SurfaceDataInput,
    pub level_spec: ContourLevelSpec,
    pub line_color: ContourLineColor,
    pub line_width: f32,
}

pub(crate) fn parse_contour_args(
    name: &'static str,
    first: Value,
    rest: Vec<Value>,
) -> BuiltinResult<ContourArgs> {
    if rest.is_empty() {
        return from_implicit_args(name, first, None, &[]);
    }
    if is_option_token(&rest[0]) {
        return from_implicit_args(name, first, None, &rest);
    }
    match rest.len() {
        1 => from_implicit_args(name, first, Some(rest[0].clone()), &rest[1..]),
        2 => from_explicit_args(
            name,
            first,
            rest[0].clone(),
            rest[1].clone(),
            None,
            &rest[2..],
        ),
        3 => from_explicit_args(
            name,
            first,
            rest[0].clone(),
            rest[1].clone(),
            Some(rest[2].clone()),
            &rest[3..],
        ),
        _ => from_explicit_args(
            name,
            first,
            rest[0].clone(),
            rest[1].clone(),
            Some(rest[2].clone()),
            &rest[3..],
        ),
    }
}

fn from_implicit_args(
    name: &'static str,
    z_value: Value,
    level_value: Option<Value>,
    options: &[Value],
) -> BuiltinResult<ContourArgs> {
    let z_input = SurfaceDataInput::from_value(z_value, name)?;
    let (rows, cols) = z_input.grid_shape(name)?;
    if rows < 2 || cols < 2 {
        return Err(plotting_error(
            name,
            format!("{name}: Z must be at least 2x2"),
        ));
    }
    let x_axis = implicit_axis(rows);
    let y_axis = implicit_axis(cols);
    let level_spec = match level_value {
        Some(value) => parse_level_spec(value, name)?,
        None => ContourLevelSpec::Auto,
    };
    let mut args = ContourArgs {
        name,
        x_axis,
        y_axis,
        z_input,
        level_spec,
        line_color: ContourLineColor::default(),
        line_width: 1.0,
    };
    apply_contour_options(&mut args, options)?;
    Ok(args)
}

fn from_explicit_args(
    name: &'static str,
    x_value: Value,
    y_value: Value,
    z_value: Value,
    level_value: Option<Value>,
    options: &[Value],
) -> BuiltinResult<ContourArgs> {
    // Axis vectors can be supplied as gpuArray; gather them (small) while keeping Z on-device.
    let x_tensor = match &x_value {
        Value::GpuTensor(handle) => super::common::gather_tensor_from_gpu(handle.clone(), name)?,
        _ => {
            Tensor::try_from(&x_value).map_err(|e| plotting_error(name, format!("{name}: {e}")))?
        }
    };
    let y_tensor = match &y_value {
        Value::GpuTensor(handle) => super::common::gather_tensor_from_gpu(handle.clone(), name)?,
        _ => {
            Tensor::try_from(&y_value).map_err(|e| plotting_error(name, format!("{name}: {e}")))?
        }
    };
    let z_input = SurfaceDataInput::from_value(z_value, name)?;
    let (rows, cols) = z_input.grid_shape(name)?;
    let (x_axis, y_axis) = if x_tensor.rows == rows
        && x_tensor.cols == cols
        && y_tensor.rows == rows
        && y_tensor.cols == cols
    {
        extract_meshgrid_axes_from_xy_matrices(&x_tensor, &y_tensor, rows, cols, name)?
    } else {
        let x_axis = numeric_vector(x_tensor);
        let y_axis = numeric_vector(y_tensor);
        if x_axis.len() < 2 || y_axis.len() < 2 {
            return Err(plotting_error(
                name,
                format!("{name}: axis vectors must contain at least two elements"),
            ));
        }
        (x_axis, y_axis)
    };
    if rows != x_axis.len() || cols != y_axis.len() {
        return Err(plotting_error(
            name,
            format!(
                "{name}: Z must be {}x{} to match axis vectors",
                x_axis.len(),
                y_axis.len()
            ),
        ));
    }
    let level_spec = match level_value {
        Some(value) => parse_level_spec(value, name)?,
        None => ContourLevelSpec::Auto,
    };
    let mut args = ContourArgs {
        name,
        x_axis,
        y_axis,
        z_input,
        level_spec,
        line_color: ContourLineColor::default(),
        line_width: 1.0,
    };
    apply_contour_options(&mut args, options)?;
    Ok(args)
}

pub(crate) fn default_level_count() -> usize {
    DEFAULT_LEVELS
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::plotting::contour")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "contour",
    op_kind: GpuOpKind::PlotRender,
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    // Plotting is a sink, but can consume gpuArray inputs zero-copy when a shared WGPU context exists.
    // Avoid forcing implicit gathers.
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Contour rendering terminates fusion graphs; gpuArray inputs may remain on device when shared plotting context is installed.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::plotting::contour")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "contour",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "contour performs I/O and stops fusion.",
};

#[runtime_builtin(
    name = "contour",
    category = "plotting",
    summary = "Render MATLAB-compatible contour plots.",
    keywords = "contour,plotting,isolines",
    sink = true,
    suppress_auto_output = true,
    type_resolver(handle_scalar_type),
    builtin_path = "crate::builtins::plotting::contour"
)]
pub fn contour_builtin(first: Value, rest: Vec<Value>) -> crate::BuiltinResult<f64> {
    let mut call = Some(ContourCall::parse(BUILTIN_NAME, first, rest)?);
    let opts = PlotRenderOptions {
        title: "Contour Plot",
        x_label: "X",
        y_label: "Y",
        axis_equal: true,
        ..Default::default()
    };
    let plot_index_out = std::rc::Rc::new(std::cell::RefCell::new(None));
    let plot_index_slot = std::rc::Rc::clone(&plot_index_out);
    let figure_handle = crate::builtins::plotting::current_figure_handle();
    let render_result = render_active_plot(BUILTIN_NAME, opts, move |figure, axes| {
        let current = call.take().expect("contour call consumed once");
        let before = figure.plots().count();
        current.render(figure, axes)?;
        let after = figure.plots().count();
        if after > before {
            *plot_index_slot.borrow_mut() = Some((axes, after - 1));
        }
        Ok(())
    });
    let Some((axes, plot_index)) = *plot_index_out.borrow() else {
        return render_result.map(|_| f64::NAN);
    };
    let handle =
        crate::builtins::plotting::state::register_contour_handle(figure_handle, axes, plot_index);
    if let Err(err) = render_result {
        let lower = err.to_string().to_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(handle);
        }
        return Err(err);
    }
    Ok(handle)
}

struct ContourCall {
    args: ContourArgs,
    color_map: ColorMap,
    base_z: f32,
}

impl ContourCall {
    fn parse(name: &'static str, first: Value, rest: Vec<Value>) -> BuiltinResult<Self> {
        let args = parse_contour_args(name, first, rest)?;
        Ok(Self {
            args,
            color_map: ColorMap::Parula,
            base_z: 0.0,
        })
    }

    fn render(self, figure: &mut runmat_plot::plots::Figure, axes: usize) -> BuiltinResult<()> {
        let ContourCall {
            args,
            color_map,
            base_z,
        } = self;
        let ContourArgs {
            name,
            x_axis,
            y_axis,
            z_input,
            level_spec,
            line_color,
            line_width,
        } = args;

        if matches!(line_color, ContourLineColor::None) {
            return Ok(());
        }

        if let SurfaceDataInput::Gpu(handle) = &z_input {
            match build_contour_gpu_plot(
                name,
                &x_axis,
                &y_axis,
                handle,
                color_map,
                base_z,
                &level_spec,
                &line_color,
            ) {
                Ok(contour) => {
                    figure.add_contour_plot_on_axes(contour.with_line_width(line_width), axes);
                    return Ok(());
                }
                Err(err) => {
                    warn!("{name} GPU path unavailable: {err}");
                }
            }
        }

        let grid =
            tensor_to_surface_grid(z_input.into_tensor(name)?, x_axis.len(), y_axis.len(), name)?;
        if let ContourLineColor::None = line_color {
            return Ok(());
        }
        let contour = build_contour_plot(
            name,
            &x_axis,
            &y_axis,
            &grid,
            color_map,
            base_z,
            &level_spec,
            &line_color,
        )?
        .with_line_width(line_width);
        figure.add_contour_plot_on_axes(contour, axes);
        Ok(())
    }
}

fn parse_level_spec(value: Value, context: &str) -> BuiltinResult<ContourLevelSpec> {
    match value {
        Value::Tensor(tensor) => parse_tensor_levels(tensor, context),
        other => {
            if let Some(count) = value_as_f64(&other) {
                parse_scalar_level_count(count, context)
            } else {
                let tensor = Tensor::try_from(&other)
                    .map_err(|e| plotting_error(context, format!("{context}: {e}")))?;
                parse_tensor_levels(tensor, context)
            }
        }
    }
}

fn parse_scalar_level_count(value: f64, context: &str) -> BuiltinResult<ContourLevelSpec> {
    if !value.is_finite() {
        return Err(plotting_error(
            context,
            format!("{context}: level count must be finite"),
        ));
    }
    if value < 1.0 {
        return Err(plotting_error(
            context,
            format!("{context}: level count must be positive"),
        ));
    }
    let rounded = value.round();
    if rounded < 1.0 {
        return Err(plotting_error(
            context,
            format!("{context}: level count must be positive"),
        ));
    }
    Ok(ContourLevelSpec::Count(rounded as usize))
}

fn parse_tensor_levels(tensor: Tensor, context: &str) -> BuiltinResult<ContourLevelSpec> {
    if tensor.data.is_empty() {
        return Err(plotting_error(
            context,
            format!("{context}: level vector must be non-empty"),
        ));
    }
    if tensor.data.len() == 1 {
        return parse_scalar_level_count(tensor.data[0], context);
    }
    if tensor.data.iter().all(|value| *value == tensor.data[0]) {
        return Ok(ContourLevelSpec::Values(vec![tensor.data[0]]));
    }
    for pair in tensor.data.windows(2) {
        if pair[1] <= pair[0] {
            return Err(plotting_error(
                context,
                format!("{context}: level values must be strictly increasing"),
            ));
        }
    }
    Ok(ContourLevelSpec::Values(tensor.data))
}

fn apply_contour_options(args: &mut ContourArgs, options: &[Value]) -> BuiltinResult<()> {
    if options.is_empty() {
        return Ok(());
    }
    if !options.len().is_multiple_of(2) {
        return Err(plotting_error(
            args.name,
            format!("{}: name-value arguments must come in pairs", args.name),
        ));
    }
    let opts = LineStyleParseOptions::generic(args.name);
    let mut level_override_seen = !matches!(args.level_spec, ContourLevelSpec::Auto);
    let mut manual_requested = false;
    for pair in options.chunks_exact(2) {
        let Some(key) = value_as_string(&pair[0]) else {
            return Err(plotting_error(
                args.name,
                format!("{}: option names must be char arrays or strings", args.name),
            ));
        };
        let lower = key.trim().to_ascii_lowercase();
        match lower.as_str() {
            "levellist" | "levels" => {
                args.level_spec = parse_level_spec(pair[1].clone(), args.name)?;
                level_override_seen = true;
            }
            "levelstep" => {
                let step = value_as_f64(&pair[1]).ok_or_else(|| {
                    plotting_error(
                        args.name,
                        format!("{}: LevelStep must be numeric", args.name),
                    )
                })?;
                if !step.is_finite() || step <= 0.0 {
                    return Err(plotting_error(
                        args.name,
                        format!("{}: LevelStep must be a positive, finite number", args.name),
                    ));
                }
                args.level_spec = ContourLevelSpec::Step(step as f32);
                level_override_seen = true;
            }
            "linecolor" => {
                args.line_color = parse_line_color_option(&opts, &pair[1])?;
            }
            "linewidth" => {
                let width = value_as_f64(&pair[1]).ok_or_else(|| {
                    plotting_error(
                        args.name,
                        format!("{}: LineWidth must be numeric", args.name),
                    )
                })?;
                if !width.is_finite() || width <= 0.0 {
                    return Err(plotting_error(
                        args.name,
                        format!("{}: LineWidth must be a positive, finite number", args.name),
                    ));
                }
                args.line_width = width as f32;
            }
            "levellistmode" => {
                let Some(mode) = value_as_string(&pair[1]) else {
                    return Err(plotting_error(
                        args.name,
                        format!(
                            "{}: LevelListMode must be the string 'auto' or 'manual'",
                            args.name
                        ),
                    ));
                };
                let normalized = mode.trim().to_ascii_lowercase();
                match normalized.as_str() {
                    "auto" => {
                        args.level_spec = ContourLevelSpec::Auto;
                        level_override_seen = false;
                        manual_requested = false;
                    }
                    "manual" => {
                        manual_requested = true;
                    }
                    other => {
                        return Err(plotting_error(
                            args.name,
                            format!("{}: unsupported LevelListMode `{other}` (expected 'auto' or 'manual')", args.name),
                        ));
                    }
                }
            }
            other => {
                return Err(plotting_error(
                    args.name,
                    format!("{}: unsupported option `{other}`", args.name),
                ));
            }
        }
    }
    if manual_requested && !level_override_seen {
        return Err(plotting_error(
            args.name,
            format!(
                "{}: LevelListMode 'manual' requires LevelList or LevelStep in the same call",
                args.name
            ),
        ));
    }
    Ok(())
}

fn parse_line_color_option(
    opts: &LineStyleParseOptions,
    value: &Value,
) -> BuiltinResult<ContourLineColor> {
    if let Some(text) = value_as_string(value) {
        let lower = text.trim().to_ascii_lowercase();
        return match lower.as_str() {
            "auto" | "flat" => Ok(ContourLineColor::Auto),
            "none" => Ok(ContourLineColor::None),
            _ => {
                let color = parse_color_value(opts, value)?;
                Ok(ContourLineColor::Color(color))
            }
        };
    }
    let color = parse_color_value(opts, value)?;
    Ok(ContourLineColor::Color(color))
}

fn is_option_token(value: &Value) -> bool {
    value_as_string(value)
        .map(|s| is_contour_option_name(&s))
        .unwrap_or(false)
}

fn is_contour_option_name(token: &str) -> bool {
    matches!(
        token.trim().to_ascii_lowercase().as_str(),
        "levellist" | "levels" | "levelstep" | "linecolor" | "linewidth" | "levellistmode"
    )
}

pub(crate) fn build_contour_gpu_plot(
    name: &'static str,
    x_axis: &[f64],
    y_axis: &[f64],
    z: &GpuTensorHandle,
    color_map: ColorMap,
    base_z: f32,
    level_spec: &ContourLevelSpec,
    line_color: &ContourLineColor,
) -> BuiltinResult<ContourPlot> {
    build_contour_gpu_plot_with_axes(
        name,
        &AxisSource::Host(x_axis.to_vec()),
        &AxisSource::Host(y_axis.to_vec()),
        z,
        color_map,
        base_z,
        level_spec,
        line_color,
    )
}

pub(crate) fn build_contour_gpu_plot_with_axes(
    name: &'static str,
    x_axis: &AxisSource,
    y_axis: &AxisSource,
    z: &GpuTensorHandle,
    color_map: ColorMap,
    base_z: f32,
    level_spec: &ContourLevelSpec,
    line_color: &ContourLineColor,
) -> BuiltinResult<ContourPlot> {
    let context = super::gpu_helpers::ensure_shared_wgpu_context(name)?;
    let z_ref = runmat_accelerate_api::export_wgpu_buffer(z)
        .ok_or_else(|| plotting_error(name, format!("{name}: unable to export GPU Z data")))?;

    let (min_z, max_z) = axis_bounds(z, name)?;
    let levels = level_spec.resolve(name, min_z, max_z)?;
    if levels.is_empty() {
        return Err(plotting_error(
            name,
            format!("{name}: no contour levels available"),
        ));
    }

    let scalar = ScalarType::from_is_f64(z_ref.precision == ProviderPrecision::F64);
    let color_table = match line_color {
        ContourLineColor::Auto => build_color_lut(color_map, 512, 1.0),
        ContourLineColor::Color(color) => vec![color.to_array()],
        ContourLineColor::None => vec![[0.0, 0.0, 0.0, 0.0]],
    };
    let (min_x, max_x) = match x_axis {
        AxisSource::Host(v) => (
            v.iter().fold(f32::INFINITY, |acc, &v| acc.min(v as f32)),
            v.iter()
                .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v as f32)),
        ),
        AxisSource::Gpu(h) => axis_bounds(h, name)?,
    };
    let (min_y, max_y) = match y_axis {
        AxisSource::Host(v) => (
            v.iter().fold(f32::INFINITY, |acc, &v| acc.min(v as f32)),
            v.iter()
                .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v as f32)),
        ),
        AxisSource::Gpu(h) => axis_bounds(h, name)?,
    };
    let bounds = contour_bounds(min_x, max_x, min_y, max_y, base_z);

    let mut x_f32_storage: Vec<f32> = Vec::new();
    let x_axis_data = match x_axis {
        AxisSource::Gpu(h) => {
            let exported = runmat_accelerate_api::export_wgpu_buffer(h).ok_or_else(|| {
                plotting_error(name, format!("{name}: unable to export GPU X axis buffer"))
            })?;
            if exported.len as usize != x_axis.len() {
                return Err(plotting_error(
                    name,
                    format!("{name}: X axis length mismatch"),
                ));
            }
            if exported.precision != z_ref.precision {
                return Err(plotting_error(
                    name,
                    format!("{name}: X axis precision must match Z precision"),
                ));
            }
            runmat_plot::gpu::axis::AxisData::Buffer(exported.buffer.clone())
        }
        AxisSource::Host(v) => {
            if scalar == ScalarType::F32 {
                x_f32_storage = v.iter().map(|&val| val as f32).collect();
                runmat_plot::gpu::axis::AxisData::F32(x_f32_storage.as_slice())
            } else {
                runmat_plot::gpu::axis::AxisData::F64(v.as_slice())
            }
        }
    };
    let mut y_f32_storage: Vec<f32> = Vec::new();
    let y_axis_data = match y_axis {
        AxisSource::Gpu(h) => {
            let exported = runmat_accelerate_api::export_wgpu_buffer(h).ok_or_else(|| {
                plotting_error(name, format!("{name}: unable to export GPU Y axis buffer"))
            })?;
            if exported.len as usize != y_axis.len() {
                return Err(plotting_error(
                    name,
                    format!("{name}: Y axis length mismatch"),
                ));
            }
            if exported.precision != z_ref.precision {
                return Err(plotting_error(
                    name,
                    format!("{name}: Y axis precision must match Z precision"),
                ));
            }
            runmat_plot::gpu::axis::AxisData::Buffer(exported.buffer.clone())
        }
        AxisSource::Host(v) => {
            if scalar == ScalarType::F32 {
                y_f32_storage = v.iter().map(|&val| val as f32).collect();
                runmat_plot::gpu::axis::AxisData::F32(y_f32_storage.as_slice())
            } else {
                runmat_plot::gpu::axis::AxisData::F64(v.as_slice())
            }
        }
    };
    let _keep_alive = (&x_f32_storage, &y_f32_storage);

    let inputs = runmat_plot::gpu::contour::ContourGpuInputs {
        x_axis: x_axis_data,
        y_axis: y_axis_data,
        z_buffer: z_ref.buffer.clone(),
        color_table: &color_table,
        level_values: &levels,
        x_len: x_axis.len() as u32,
        y_len: y_axis.len() as u32,
        scalar,
    };

    let params = runmat_plot::gpu::contour::ContourGpuParams {
        min_z,
        max_z,
        base_z,
        level_count: levels.len() as u32,
    };

    let gpu_vertices = runmat_plot::gpu::contour::pack_contour_vertices(
        &context.device,
        &context.queue,
        &inputs,
        &params,
    )
    .map_err(|e| plotting_error(name, format!("{name}: failed to build GPU vertices: {e}")))?;

    let vertex_count = gpu_vertices.vertex_count;
    Ok(
        ContourPlot::from_gpu_buffer(gpu_vertices, vertex_count, base_z, bounds)
            .with_label("Contours"),
    )
}

pub(crate) fn build_contour_plot(
    name: &'static str,
    x_axis: &[f64],
    y_axis: &[f64],
    grid: &[Vec<f64>],
    color_map: ColorMap,
    base_z: f32,
    level_spec: &ContourLevelSpec,
    line_color: &ContourLineColor,
) -> BuiltinResult<ContourPlot> {
    let (min_z, max_z) = grid_extents(name, grid)?;
    let levels = level_spec.resolve(name, min_z, max_z)?;
    let color_table = match line_color {
        ContourLineColor::Auto => build_color_lut(color_map, 512, 1.0),
        ContourLineColor::Color(color) => vec![color.to_array()],
        ContourLineColor::None => vec![[0.0, 0.0, 0.0, 0.0]],
    };
    let mut vertices = Vec::new();

    for (level_idx, level) in levels.iter().enumerate() {
        let color = match line_color {
            ContourLineColor::Auto => sample_color(&color_table, level_idx, levels.len()),
            ContourLineColor::Color(color) => *color,
            ContourLineColor::None => Vec4::new(0.0, 0.0, 0.0, 0.0),
        };
        march_cells(x_axis, y_axis, grid, *level, color, base_z, &mut vertices);
    }

    let bounds = contour_bounds(
        x_axis
            .iter()
            .fold(f32::INFINITY, |acc, &v| acc.min(v as f32)),
        x_axis
            .iter()
            .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v as f32)),
        y_axis
            .iter()
            .fold(f32::INFINITY, |acc, &v| acc.min(v as f32)),
        y_axis
            .iter()
            .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v as f32)),
        base_z,
    );

    Ok(ContourPlot::from_vertices(vertices, base_z, bounds).with_label("Contours"))
}

pub(crate) fn build_contour_fill_gpu_plot(
    name: &'static str,
    x_axis: &[f64],
    y_axis: &[f64],
    z: &GpuTensorHandle,
    color_map: ColorMap,
    base_z: f32,
    level_spec: &ContourLevelSpec,
) -> BuiltinResult<ContourFillPlot> {
    let context = super::gpu_helpers::ensure_shared_wgpu_context(name)?;
    let z_ref = runmat_accelerate_api::export_wgpu_buffer(z)
        .ok_or_else(|| plotting_error(name, format!("{name}: unable to export GPU Z data")))?;
    let (min_z, max_z) = axis_bounds(z, name)?;
    let levels = ensure_fill_levels(name, level_spec, min_z, max_z)?;
    let palette = build_color_lut(color_map, palette_size(&levels), 0.95);
    let scalar = ScalarType::from_is_f64(z_ref.precision == ProviderPrecision::F64);
    let x_f32 = if scalar == ScalarType::F32 {
        Some(x_axis.iter().map(|&v| v as f32).collect::<Vec<f32>>())
    } else {
        None
    };
    let y_f32 = if scalar == ScalarType::F32 {
        Some(y_axis.iter().map(|&v| v as f32).collect::<Vec<f32>>())
    } else {
        None
    };
    let (min_x, max_x) = (
        x_axis
            .iter()
            .fold(f32::INFINITY, |acc, &v| acc.min(v as f32)),
        x_axis
            .iter()
            .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v as f32)),
    );
    let (min_y, max_y) = (
        y_axis
            .iter()
            .fold(f32::INFINITY, |acc, &v| acc.min(v as f32)),
        y_axis
            .iter()
            .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v as f32)),
    );
    let bounds = contour_bounds(min_x, max_x, min_y, max_y, base_z);

    let x_axis_data = if let Some(values) = x_f32.as_ref() {
        runmat_plot::gpu::axis::AxisData::F32(values.as_slice())
    } else {
        runmat_plot::gpu::axis::AxisData::F64(x_axis)
    };
    let y_axis_data = if let Some(values) = y_f32.as_ref() {
        runmat_plot::gpu::axis::AxisData::F32(values.as_slice())
    } else {
        runmat_plot::gpu::axis::AxisData::F64(y_axis)
    };

    let inputs = contour_fill::ContourFillGpuInputs {
        x_axis: x_axis_data,
        y_axis: y_axis_data,
        z_buffer: z_ref.buffer.clone(),
        color_table: &palette,
        level_values: &levels,
        x_len: x_axis.len() as u32,
        y_len: y_axis.len() as u32,
        scalar,
    };

    let params = contour_fill::ContourFillGpuParams { base_z, alpha: 1.0 };

    let gpu_vertices =
        contour_fill::pack_contour_fill_vertices(&context.device, &context.queue, &inputs, &params)
            .map_err(|e| {
                plotting_error(name, format!("{name}: failed to build GPU vertices: {e}"))
            })?;
    let vertex_count = gpu_vertices.vertex_count;
    Ok(
        ContourFillPlot::from_gpu_buffer(gpu_vertices, vertex_count, bounds)
            .with_label("Filled Contours"),
    )
}

pub(crate) fn build_contour_fill_plot(
    name: &'static str,
    x_axis: &[f64],
    y_axis: &[f64],
    grid: &[Vec<f64>],
    color_map: ColorMap,
    base_z: f32,
    level_spec: &ContourLevelSpec,
) -> BuiltinResult<ContourFillPlot> {
    let (min_z, max_z) = grid_extents(name, grid)?;
    let levels = ensure_fill_levels(name, level_spec, min_z, max_z)?;
    let palette_raw = build_color_lut(color_map, palette_size(&levels), 0.95);
    let palette: Vec<Vec4> = palette_raw.iter().map(|c| Vec4::from_array(*c)).collect();

    let nx = x_axis.len();
    let ny = y_axis.len();
    if nx < 2 || ny < 2 {
        return Err(plotting_error(
            name,
            format!("{name}: axis vectors must contain at least two elements"),
        ));
    }

    let band_count = levels.len() - 1;
    let mut vertices = Vec::with_capacity((nx - 1) * (ny - 1) * band_count * 12);
    for col in 0..ny - 1 {
        for row in 0..nx - 1 {
            let p0 = ScalarPoint2 {
                pos: Vec2::new(x_axis[row] as f32, y_axis[col] as f32),
                value: grid[row][col] as f32,
            };
            let p1 = ScalarPoint2 {
                pos: Vec2::new(x_axis[row + 1] as f32, y_axis[col] as f32),
                value: grid[row + 1][col] as f32,
            };
            let p2 = ScalarPoint2 {
                pos: Vec2::new(x_axis[row + 1] as f32, y_axis[col + 1] as f32),
                value: grid[row + 1][col + 1] as f32,
            };
            let p3 = ScalarPoint2 {
                pos: Vec2::new(x_axis[row] as f32, y_axis[col + 1] as f32),
                value: grid[row][col + 1] as f32,
            };
            for band_idx in 0..band_count {
                let lo = levels[band_idx];
                let hi = levels[band_idx + 1];
                let include_hi = band_idx + 1 == band_count;
                let color = palette[band_idx.min(palette.len() - 1)];
                let cell_tris = [[p0, p1, p2], [p0, p2, p3]];
                for tri in cell_tris {
                    triangulate_band_triangle(
                        tri,
                        lo,
                        hi,
                        include_hi,
                        color,
                        base_z,
                        &mut vertices,
                    );
                }
            }
        }
    }

    let bounds = contour_bounds(
        x_axis
            .iter()
            .fold(f32::INFINITY, |acc, &v| acc.min(v as f32)),
        x_axis
            .iter()
            .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v as f32)),
        y_axis
            .iter()
            .fold(f32::INFINITY, |acc, &v| acc.min(v as f32)),
        y_axis
            .iter()
            .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v as f32)),
        base_z,
    );

    Ok(ContourFillPlot::from_vertices(vertices, bounds).with_label("Filled Contours"))
}

fn grid_extents(name: &'static str, grid: &[Vec<f64>]) -> BuiltinResult<(f32, f32)> {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for row in grid {
        for &value in row {
            if value.is_finite() {
                min_v = min_v.min(value);
                max_v = max_v.max(value);
            }
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        Err(plotting_error(
            name,
            format!("{name}: unable to determine data range"),
        ))
    } else {
        Ok((min_v as f32, max_v as f32))
    }
}

fn evenly_spaced_levels(count: usize, min_z: f32, max_z: f32) -> Vec<f32> {
    if count <= 1 || (max_z - min_z).abs() < f32::EPSILON {
        return vec![min_z];
    }
    let mut levels = Vec::with_capacity(count);
    let step = (max_z - min_z) / (count - 1) as f32;
    for i in 0..count {
        levels.push(min_z + step * i as f32);
    }
    levels
}

fn sample_color(table: &[[f32; 4]], idx: usize, total_levels: usize) -> Vec4 {
    if table.is_empty() {
        return Vec4::ONE;
    }
    if total_levels <= 1 {
        let c = table[0];
        return Vec4::from_array(c);
    }
    let t = idx as f32 / (total_levels as f32 - 1.0);
    let scaled = t * (table.len() as f32 - 1.0);
    let lower = scaled.floor() as usize;
    let upper = scaled.ceil() as usize;
    if lower >= table.len() {
        return Vec4::from_array(*table.last().unwrap());
    }
    if upper >= table.len() {
        return Vec4::from_array(table[lower]);
    }
    let frac = scaled - lower as f32;
    let a = Vec4::from_array(table[lower]);
    let b = Vec4::from_array(table[upper]);
    a.lerp(b, frac)
}

fn march_cells(
    x_axis: &[f64],
    y_axis: &[f64],
    grid: &[Vec<f64>],
    level: f32,
    color: Vec4,
    base_z: f32,
    out: &mut Vec<Vertex>,
) {
    let nx = x_axis.len();
    let ny = y_axis.len();
    for col in 0..ny - 1 {
        for row in 0..nx - 1 {
            let z00 = grid[row][col] as f32;
            let z10 = grid[row + 1][col] as f32;
            let z11 = grid[row + 1][col + 1] as f32;
            let z01 = grid[row][col + 1] as f32;

            let corners = [
                Vec2::new(x_axis[row] as f32, y_axis[col] as f32),
                Vec2::new(x_axis[row + 1] as f32, y_axis[col] as f32),
                Vec2::new(x_axis[row + 1] as f32, y_axis[col + 1] as f32),
                Vec2::new(x_axis[row] as f32, y_axis[col + 1] as f32),
            ];
            let values = [z00, z10, z11, z01];

            let case_index = ((z00 > level) as u32)
                | (((z10 > level) as u32) << 1)
                | (((z11 > level) as u32) << 2)
                | (((z01 > level) as u32) << 3);

            let mut segments: [Vec2; 4] = [Vec2::ZERO; 4];
            let mut segment_count = 0u32;

            match case_index {
                0 | 15 => {}
                1 | 14 => add_segment(
                    3,
                    0,
                    &corners,
                    &values,
                    level,
                    &mut segments,
                    &mut segment_count,
                ),
                2 | 13 => add_segment(
                    0,
                    1,
                    &corners,
                    &values,
                    level,
                    &mut segments,
                    &mut segment_count,
                ),
                3 | 12 => add_segment(
                    3,
                    1,
                    &corners,
                    &values,
                    level,
                    &mut segments,
                    &mut segment_count,
                ),
                4 | 11 => add_segment(
                    1,
                    2,
                    &corners,
                    &values,
                    level,
                    &mut segments,
                    &mut segment_count,
                ),
                5 | 10 => add_ambiguous_segments(
                    case_index,
                    &corners,
                    &values,
                    level,
                    &mut segments,
                    &mut segment_count,
                ),
                6 | 9 => add_segment(
                    0,
                    2,
                    &corners,
                    &values,
                    level,
                    &mut segments,
                    &mut segment_count,
                ),
                7 | 8 => add_segment(
                    3,
                    2,
                    &corners,
                    &values,
                    level,
                    &mut segments,
                    &mut segment_count,
                ),
                _ => {}
            }

            for idx in 0..segment_count {
                let start = segments[(idx * 2) as usize];
                let end = segments[(idx * 2 + 1) as usize];
                push_segment(start, end, color, base_z, out);
            }
        }
    }
}

fn add_segment(
    edge_a: u32,
    edge_b: u32,
    corners: &[Vec2; 4],
    values: &[f32; 4],
    level: f32,
    io_segments: &mut [Vec2; 4],
    io_count: &mut u32,
) {
    if *io_count >= 2 {
        return;
    }
    let idx = *io_count * 2;
    io_segments[idx as usize] = interpolate_edge(edge_a, corners, values, level);
    io_segments[idx as usize + 1] = interpolate_edge(edge_b, corners, values, level);
    *io_count += 1;
}

fn add_ambiguous_segments(
    case_index: u32,
    corners: &[Vec2; 4],
    values: &[f32; 4],
    level: f32,
    io_segments: &mut [Vec2; 4],
    io_count: &mut u32,
) {
    let f00 = values[0] - level;
    let f10 = values[1] - level;
    let f11 = values[2] - level;
    let f01 = values[3] - level;
    let q = f00 * f11 - f10 * f01;

    let use_default = q > 0.0 || (q.abs() <= 1e-6 && case_index == 5);
    match (case_index, use_default) {
        (5, true) | (10, true) => {
            add_segment(3, 2, corners, values, level, io_segments, io_count);
            add_segment(0, 1, corners, values, level, io_segments, io_count);
        }
        (5, false) | (10, false) => {
            add_segment(3, 0, corners, values, level, io_segments, io_count);
            add_segment(1, 2, corners, values, level, io_segments, io_count);
        }
        _ => {}
    }
}

fn interpolate_edge(edge: u32, corners: &[Vec2; 4], values: &[f32; 4], level: f32) -> Vec2 {
    let (a_idx, b_idx) = match edge {
        0 => (0, 1),
        1 => (1, 2),
        2 => (2, 3),
        _ => (3, 0),
    };
    let a = corners[a_idx];
    let b = corners[b_idx];
    let va = values[a_idx];
    let vb = values[b_idx];
    let delta = vb - va;
    let t = if delta.abs() <= 1e-6 {
        0.5
    } else {
        ((level - va) / delta).clamp(0.0, 1.0)
    };
    a + (b - a) * t
}

fn push_segment(start: Vec2, end: Vec2, color: Vec4, base_z: f32, out: &mut Vec<Vertex>) {
    let z = Vec3::new(0.0, 0.0, 1.0);
    out.push(Vertex {
        position: Vec3::new(start.x, start.y, base_z).to_array(),
        color: color.to_array(),
        normal: z.to_array(),
        tex_coords: [0.0, 0.0],
    });
    out.push(Vertex {
        position: Vec3::new(end.x, end.y, base_z).to_array(),
        color: color.to_array(),
        normal: z.to_array(),
        tex_coords: [0.0, 0.0],
    });
}

fn implicit_axis(len: usize) -> Vec<f64> {
    (0..len).map(|idx| (idx + 1) as f64).collect()
}

fn ensure_fill_levels(
    name: &'static str,
    level_spec: &ContourLevelSpec,
    min_z: f32,
    max_z: f32,
) -> BuiltinResult<Vec<f32>> {
    let mut levels = level_spec.resolve(name, min_z, max_z)?;
    if levels.len() < 2 {
        let second = if (max_z - min_z).abs() < f32::EPSILON {
            min_z + 1.0
        } else {
            max_z
        };
        levels = vec![min_z, second];
    }
    Ok(levels)
}

fn palette_size(levels: &[f32]) -> usize {
    levels.len().saturating_sub(1).max(1)
}

fn push_fill_triangle(vertices: &mut Vec<Vertex>, positions: [Vec3; 3], colors: [Vec4; 3]) {
    let normal = Vec3::new(0.0, 0.0, 1.0);
    for i in 0..3 {
        vertices.push(Vertex {
            position: positions[i].to_array(),
            color: colors[i].to_array(),
            normal: normal.to_array(),
            tex_coords: [0.0, 0.0],
        });
    }
}

#[derive(Clone, Copy, Debug)]
struct ScalarPoint2 {
    pos: Vec2,
    value: f32,
}

fn interpolate_scalar_point(a: ScalarPoint2, b: ScalarPoint2, threshold: f32) -> ScalarPoint2 {
    let delta = b.value - a.value;
    let t = if delta.abs() <= 1e-6 {
        0.5
    } else {
        ((threshold - a.value) / delta).clamp(0.0, 1.0)
    };
    ScalarPoint2 {
        pos: a.pos + (b.pos - a.pos) * t,
        value: threshold,
    }
}

fn clip_polygon_lower(poly: &[ScalarPoint2], threshold: f32) -> Vec<ScalarPoint2> {
    clip_polygon(poly, |v| v >= threshold, threshold)
}

fn clip_polygon_upper(poly: &[ScalarPoint2], threshold: f32, inclusive: bool) -> Vec<ScalarPoint2> {
    if inclusive {
        clip_polygon(poly, |v| v <= threshold, threshold)
    } else {
        clip_polygon(poly, |v| v < threshold, threshold)
    }
}

fn clip_polygon<F>(poly: &[ScalarPoint2], inside: F, threshold: f32) -> Vec<ScalarPoint2>
where
    F: Fn(f32) -> bool,
{
    if poly.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut prev = *poly.last().unwrap();
    let mut prev_inside = inside(prev.value);
    for &curr in poly {
        let curr_inside = inside(curr.value);
        if curr_inside != prev_inside {
            out.push(interpolate_scalar_point(prev, curr, threshold));
        }
        if curr_inside {
            out.push(curr);
        }
        prev = curr;
        prev_inside = curr_inside;
    }
    out
}

fn triangulate_band_triangle(
    tri: [ScalarPoint2; 3],
    lo: f32,
    hi: f32,
    include_hi: bool,
    color: Vec4,
    base_z: f32,
    out: &mut Vec<Vertex>,
) {
    let poly = clip_polygon_upper(&clip_polygon_lower(&tri, lo), hi, include_hi);
    if poly.len() < 3 {
        return;
    }
    let p0 = poly[0].pos;
    for idx in 1..poly.len() - 1 {
        push_fill_triangle(
            out,
            [
                Vec3::new(p0.x, p0.y, base_z),
                Vec3::new(poly[idx].pos.x, poly[idx].pos.y, base_z),
                Vec3::new(poly[idx + 1].pos.x, poly[idx + 1].pos.y, base_z),
            ],
            [color, color, color],
        );
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::plotting::tests::ensure_plot_test_env;
    use runmat_builtins::NumericDType;
    use runmat_builtins::{ResolveContext, Type};

    fn setup_plot_tests() {
        ensure_plot_test_env();
    }

    fn tensor_from(data: &[f64], rows: usize, cols: usize) -> Tensor {
        let mut shape = vec![rows];
        if cols > 1 {
            shape.push(cols);
        }
        Tensor {
            data: data.to_vec(),
            shape,
            rows,
            cols,
            dtype: NumericDType::F64,
        }
    }

    fn assert_contour_vertices_within_bounds(vertices: &[Vertex], x_axis: &[f64], y_axis: &[f64]) {
        assert_eq!(vertices.len() % 2, 0);
        let min_x = x_axis.iter().copied().fold(f64::INFINITY, f64::min) as f32;
        let max_x = x_axis.iter().copied().fold(f64::NEG_INFINITY, f64::max) as f32;
        let min_y = y_axis.iter().copied().fold(f64::INFINITY, f64::min) as f32;
        let max_y = y_axis.iter().copied().fold(f64::NEG_INFINITY, f64::max) as f32;
        for vertex in vertices {
            assert!(vertex.position[0].is_finite());
            assert!(vertex.position[1].is_finite());
            assert!(vertex.position[2].is_finite());
            assert!(vertex.position[0] >= min_x - 1e-4 && vertex.position[0] <= max_x + 1e-4);
            assert!(vertex.position[1] >= min_y - 1e-4 && vertex.position[1] <= max_y + 1e-4);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn explicit_axes_must_match_grid() {
        setup_plot_tests();
        let x = Value::Tensor(tensor_from(&[0.0, 1.0], 2, 1));
        let y = Value::Tensor(tensor_from(&[0.0, 1.0, 2.0], 3, 1));
        let z = Value::Tensor(tensor_from(&[0.0, 1.0, 2.0, 3.0], 2, 2));
        let result = parse_contour_args("contour", x, vec![y, z]);
        assert!(result.is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn implicit_axes_respect_tensor_shape() {
        setup_plot_tests();
        let z = Value::Tensor(tensor_from(&[0.0, 1.0, 2.0, 3.0], 2, 2));
        let args = parse_contour_args("contour", z, Vec::new()).unwrap();
        assert_eq!(args.x_axis, vec![1.0, 2.0]);
        assert_eq!(args.y_axis, vec![1.0, 2.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn explicit_axes_plus_scalar_level_count_parse_correctly() {
        setup_plot_tests();
        let x = Value::Tensor(tensor_from(&[0.0, 1.0], 2, 1));
        let y = Value::Tensor(tensor_from(&[0.0, 1.0], 2, 1));
        let z = Value::Tensor(tensor_from(&[0.0, 1.0, 1.0, 0.0], 2, 2));
        let args = parse_contour_args("contour", x, vec![y, z, Value::Num(12.0)]).unwrap();
        assert_eq!(args.x_axis, vec![0.0, 1.0]);
        assert_eq!(args.y_axis, vec![0.0, 1.0]);
        match args.level_spec {
            ContourLevelSpec::Count(count) => assert_eq!(count, 12),
            other => panic!("expected scalar level count, found {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn explicit_meshgrid_axes_parse_correctly() {
        setup_plot_tests();
        let x = Value::Tensor(tensor_from(&[10.0, 10.0, 20.0, 20.0], 2, 2));
        let y = Value::Tensor(tensor_from(&[1.0, 2.0, 1.0, 2.0], 2, 2));
        let z = Value::Tensor(tensor_from(&[0.0, 1.0, 1.0, 0.0], 2, 2));
        let args = parse_contour_args("contour", x, vec![y, z]).unwrap();
        assert_eq!(args.x_axis, vec![1.0, 2.0]);
        assert_eq!(args.y_axis, vec![10.0, 20.0]);
    }

    #[test]
    fn interpolate_edge_handles_descending_values() {
        let corners = [
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(2.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let values = [2.0, 0.0, 0.0, 2.0];
        let point = interpolate_edge(0, &corners, &values, 1.0);
        assert!((point.x - 1.0).abs() < 1e-6);
        assert!(point.y.abs() < 1e-6);
    }

    #[test]
    fn ambiguous_case_uses_asymptotic_decider() {
        let corners = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 1.0),
        ];
        let values = [2.0, 0.8, 2.0, 0.0];
        let mut segments = [Vec2::ZERO; 4];
        let mut count = 0;
        add_ambiguous_segments(5, &corners, &values, 1.0, &mut segments, &mut count);
        assert_eq!(count, 2);
        assert!(segments[0].x.abs() < 1e-6 || (segments[0].y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn contour_cpu_matrix_handles_nonuniform_axes_fixture() {
        let x_axis = vec![-3.0, -1.0, 0.5, 2.0];
        let y_axis = vec![-2.0, -0.25, 1.5, 3.0];
        let grid = vec![
            vec![0.0, 0.3, 0.9, 1.2],
            vec![-0.4, 0.2, 0.8, 1.0],
            vec![-0.8, -0.1, 0.4, 0.9],
            vec![-1.0, -0.5, 0.1, 0.6],
        ];
        let mut plot = build_contour_plot(
            "contour",
            &x_axis,
            &y_axis,
            &grid,
            ColorMap::Parula,
            0.0,
            &ContourLevelSpec::Values(vec![-0.5, 0.0, 0.5, 1.0]),
            &ContourLineColor::Auto,
        )
        .expect("contour plot");
        let render = plot.render_data();
        assert!(!render.vertices.is_empty());
        assert_contour_vertices_within_bounds(&render.vertices, &x_axis, &y_axis);
    }

    #[test]
    fn contour_cpu_saddle_fixture_emits_finite_segments() {
        let x_axis = vec![0.0, 1.0, 2.0];
        let y_axis = vec![0.0, 1.0, 2.0];
        let grid = vec![
            vec![1.0, -1.0, 1.0],
            vec![-1.0, 1.0, -1.0],
            vec![1.0, -1.0, 1.0],
        ];
        let mut plot = build_contour_plot(
            "contour",
            &x_axis,
            &y_axis,
            &grid,
            ColorMap::Parula,
            0.0,
            &ContourLevelSpec::Values(vec![0.0]),
            &ContourLineColor::Auto,
        )
        .expect("contour plot");
        let render = plot.render_data();
        assert!(!render.vertices.is_empty());
        assert_contour_vertices_within_bounds(&render.vertices, &x_axis, &y_axis);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn level_vector_must_increase() {
        setup_plot_tests();
        let bad_levels = Value::Tensor(tensor_from(&[0.0, 0.0], 1, 2));
        let repeated = parse_level_spec(bad_levels, "contour").unwrap();
        match repeated {
            ContourLevelSpec::Values(values) => assert_eq!(values, vec![0.0]),
            other => panic!("expected repeated single contour level, found {other:?}"),
        }

        let good_levels = Value::Tensor(tensor_from(&[0.0, 1.0, 2.0], 1, 3));
        let spec = parse_level_spec(good_levels, "contour").unwrap();
        match spec {
            ContourLevelSpec::Values(values) => assert_eq!(values, vec![0.0, 1.0, 2.0]),
            _ => panic!("expected explicit levels"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn level_step_option_generates_sequence() {
        setup_plot_tests();
        let z = Value::Tensor(tensor_from(&[0.0, 1.0, 2.0, 3.0], 2, 2));
        let args = parse_contour_args(
            "contour",
            z,
            vec![Value::String("LevelStep".into()), Value::Num(0.5)],
        )
        .unwrap();
        match args.level_spec {
            ContourLevelSpec::Step(step) => assert!((step - 0.5).abs() < f32::EPSILON),
            other => panic!("expected LevelStep, found {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn line_color_option_parses_literal() {
        setup_plot_tests();
        let z = Value::Tensor(tensor_from(&[0.0, 1.0, 2.0, 3.0], 2, 2));
        let args = parse_contour_args(
            "contour",
            z,
            vec![Value::String("LineColor".into()), Value::String("r".into())],
        )
        .unwrap();
        match args.line_color {
            ContourLineColor::Color(color) => {
                assert!((color.x - 1.0).abs() < f32::EPSILON);
                assert!((color.y).abs() < f32::EPSILON);
            }
            other => panic!("expected explicit color, found {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linewidth_option_parses() {
        setup_plot_tests();
        let z = Value::Tensor(tensor_from(&[0.0, 1.0, 2.0, 3.0], 2, 2));
        let args = parse_contour_args(
            "contour",
            z,
            vec![Value::String("LineWidth".into()), Value::Num(2.0)],
        )
        .unwrap();
        assert!((args.line_width - 2.0).abs() < f32::EPSILON);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn level_list_option_accepts_explicit_vector() {
        setup_plot_tests();
        let z = Value::Tensor(tensor_from(&[0.0, 1.0, 2.0, 3.0], 2, 2));
        let args = parse_contour_args(
            "contour",
            z,
            vec![
                Value::String("LevelList".into()),
                Value::Tensor(tensor_from(&[0.0, 0.5, 1.0], 1, 3)),
            ],
        )
        .unwrap();
        match args.level_spec {
            ContourLevelSpec::Values(values) => {
                assert_eq!(values, vec![0.0, 0.5, 1.0]);
            }
            other => panic!("expected explicit level list, found {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn line_color_none_suppresses_overlays() {
        setup_plot_tests();
        let z = Value::Tensor(tensor_from(&[0.0, 1.0, 2.0, 3.0], 2, 2));
        let args = parse_contour_args(
            "contour",
            z,
            vec![
                Value::String("LineColor".into()),
                Value::String("none".into()),
            ],
        )
        .unwrap();
        match args.line_color {
            ContourLineColor::None => {}
            other => panic!("expected LineColor none, found {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn level_list_mode_manual_requires_explicit_levels() {
        setup_plot_tests();
        let z = Value::Tensor(tensor_from(&[0.0, 1.0, 2.0, 3.0], 2, 2));
        let err = parse_contour_args(
            "contour",
            z.clone(),
            vec![
                Value::String("LevelListMode".into()),
                Value::String("manual".into()),
            ],
        );
        assert!(err.is_err());

        let ok = parse_contour_args(
            "contour",
            z,
            vec![
                Value::String("LevelListMode".into()),
                Value::String("manual".into()),
                Value::String("LevelStep".into()),
                Value::Num(0.25),
            ],
        );
        assert!(ok.is_ok());
    }

    #[test]
    fn contour_type_is_numeric_handle() {
        assert_eq!(
            handle_scalar_type(&[Type::tensor()], &ResolveContext::new(Vec::new())),
            Type::Num
        );
    }

    #[test]
    fn contour_returns_handle() {
        setup_plot_tests();
        let handle = contour_builtin(
            Value::Tensor(tensor_from(&[0.0, 1.0, 1.0, 0.0], 2, 2)),
            Vec::new(),
        )
        .expect("contour should return handle");
        assert!(handle.is_finite());
    }
}
